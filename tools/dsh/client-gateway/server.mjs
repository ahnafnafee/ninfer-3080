import { createServer } from 'node:http';
import { timingSafeEqual, randomBytes } from 'node:crypto';
import { readFile, writeFile, mkdir, copyFile } from 'node:fs/promises';
import { openSync, closeSync } from 'node:fs';
import { join, dirname, resolve } from 'node:path';
import { homedir } from 'node:os';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { loadRuntime, clientCatalog } from './runtime.mjs';
import { RequestError, translateRequest, responseEvents } from './responses.mjs';

const host = '127.0.0.1';
const port = 18021;
const argument = name => { const index = process.argv.indexOf(name); return index < 0 ? undefined : process.argv[index + 1]; };
const cliHome = resolve(process.env.CODEX_HOME ?? join(homedir(), '.codex'));
// AppData writes from a packaged desktop app can be redirected into its private
// store, leaving ordinary terminal clients unable to read the generated catalog.
const dataDirectory = resolve(argument('--data-dir') ?? process.env.DSH_CLIENT_GATEWAY_HOME ?? join(cliHome, 'dsh-client-gateway'));
const dshHome = argument('--dsh-home') ?? process.env.DSH_HOME ?? join(homedir(), '.dsh');
const tokenFile = join(dataDirectory, 'auth-token');
const json = (response, status, body) => { response.writeHead(status, { 'content-type': 'application/json' }); response.end(JSON.stringify(body)); };

export function gatewayServer(runtime, token, reloadRuntime) {
  const authorized = request => {
    const actual = Buffer.from(request.headers.authorization ?? '');
    const expected = Buffer.from(`Bearer ${token}`);
    return actual.length === expected.length && timingSafeEqual(actual, expected);
  };
  return createServer(async (request, response) => {
    if (!authorized(request)) return json(response, 401, { error: { code: 'invalid_api_key', message: 'Gateway authentication required' } });
    if (request.method === 'POST' && request.url === '/reload' && reloadRuntime) {
      try {
        const next = await reloadRuntime();
        if (next.failures.length) throw new Error('Some configured models could not be loaded');
        runtime = next;
        return json(response, 200, { models: runtime.routes.size });
      } catch (error) { return json(response, 503, { error: { code: 'catalog_reload_failed', message: error.message } }); }
    }
    if (request.method === 'GET' && request.url === '/health') return json(response, 200, { service: 'dsh-client-gateway', models: runtime.routes.size, pid: process.pid });
    if (request.method === 'GET' && request.url === '/v1/models') return json(response, 200, { object: 'list', data: [...runtime.routes.values()].map(route => ({ id: route.id, object: 'model', owned_by: route.provider, context_window: route.info.context.contextWindow })) });
    if (request.method !== 'POST' || request.url !== '/v1/responses') return json(response, 404, { error: { code: 'not_found', message: 'Supported endpoints: GET /v1/models and POST /v1/responses' } });
    const controller = new AbortController();
    response.on('close', () => { if (!response.writableEnded) controller.abort(); });
    let heartbeat;
    try {
      const buffers = []; let bytes = 0;
      for await (const buffer of request) {
        bytes += buffer.length;
        if (bytes > 32 * 1024 * 1024) throw new RequestError('Request exceeds 32 MiB', 'request_too_large', 413);
        buffers.push(buffer);
      }
      let body;
      try { body = JSON.parse(Buffer.concat(buffers).toString('utf8')); }
      catch { throw new RequestError('Invalid JSON request'); }
      const route = runtime.routes.get(body.model);
      if (!route) throw new RequestError(`Unknown configured DSH model: ${body.model}`, 'model_not_found', 404);
      const { options, toolMap } = translateRequest(body, route, controller.signal);
      const events = responseEvents(route.adapter.stream(options), route.id, toolMap);
      if (body.stream) {
        response.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
        heartbeat = setInterval(() => { if (!response.destroyed) response.write(': keepalive\n\n'); }, 15000);
        for await (const event of events) {
          if (!response.write(`event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`)) await once(response, 'drain', { signal: controller.signal });
        }
        response.end();
      } else {
        let terminal;
        for await (const event of events) if (event.response) terminal = event.response;
        json(response, terminal?.status === 'failed' ? 502 : 200, terminal);
      }
    } catch (error) {
      if (response.destroyed) return;
      const failure = { code: error.code ?? 'gateway_error', message: error.message };
      if (response.headersSent) { response.write(`event: error\ndata: ${JSON.stringify({ type: 'error', ...failure })}\n\n`); response.end(); }
      else json(response, error.status ?? 500, { error: failure });
    } finally { clearInterval(heartbeat); }
  });
}

async function install() {
  const runtime = await loadRuntime(dshHome);
  if (runtime.failures.length) throw new Error('Catalog is incomplete: ' + JSON.stringify(runtime.failures));
  await mkdir(dataDirectory, { recursive: true });
  await mkdir(cliHome, { recursive: true });
  let token;
  try { token = await readFile(tokenFile, 'utf8'); }
  catch (error) { if (error.code !== 'ENOENT') throw error; token = randomBytes(32).toString('hex'); await writeFile(tokenFile, token, { mode: 0o600 }); }
  const catalogPath = join(dataDirectory, 'models.json');
  const profilePath = join(cliHome, 'dsh.config.toml');
  const q = JSON.stringify;
  const defaultRoute = runtime.routes.get(runtime.defaultModel) ?? runtime.routes.values().next().value;
  const effort = defaultRoute.info.reasoning?.defaultEffort ?? 'off';
  const helperArgs = [fileURLToPath(import.meta.url), '--auth-token', '--data-dir', dataDirectory, '--dsh-home', dshHome];
  const selectedEffort = effort === 'off' ? 'none' : effort;
  const profile = `# DSH model routes; select a model using /model after launching codex -p dsh.\nmodel = ${q(defaultRoute.id)}\nmodel_provider = "dsh"\nmodel_reasoning_effort = ${q(selectedEffort)}\nmodel_catalog_json = ${q(catalogPath)}\nweb_search = "disabled"\n\n[agents]\ndefault_subagent_model = ${q(defaultRoute.id)}\ndefault_subagent_reasoning_effort = ${q(selectedEffort)}\n\n[model_providers.dsh]\nname = "DSH models"\nbase_url = "http://${host}:${port}/v1"\nwire_api = "responses"\nrequires_openai_auth = false\nsupports_websockets = false\nrequest_max_retries = 0\nstream_max_retries = 0\nstream_idle_timeout_ms = 600000\n\n[model_providers.dsh.auth]\ncommand = ${q(process.execPath)}\nargs = [${helperArgs.map(q).join(', ')}]\ntimeout_ms = 30000\nrefresh_interval_ms = 300000\n`;
  // Back up only files this installer owns. The base CLI config is untouched.
  for (const path of [catalogPath, profilePath]) {
    try { await copyFile(path, `${path}.backup-${Date.now()}`); }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
  }
  await writeFile(catalogPath, JSON.stringify(clientCatalog(runtime), null, 2) + '\n');
  await writeFile(profilePath, profile);
  try {
    const response = await fetch(`http://${host}:${port}/reload`, { method: 'POST', headers: { authorization: `Bearer ${token.trim()}` }, signal: AbortSignal.timeout(15000) });
    if (!response.ok) console.warn(`Restart the gateway to load its updated catalog (HTTP ${response.status}).`);
  } catch (error) { if (error.cause?.code !== 'ECONNREFUSED') throw error; }
  console.log(JSON.stringify({ profile: profilePath, models: runtime.routes.size, default: defaultRoute.id, catalog: catalogPath }));
}

async function ensureServer(token) {
  const health = async () => {
    try {
      const response = await fetch(`http://${host}:${port}/health`, { headers: { authorization: `Bearer ${token}` }, signal: AbortSignal.timeout(1500) });
      if (!response.ok) throw new Error(`Port ${port} is occupied by a service that rejected gateway authentication`);
      const body = await response.json();
      if (body.service !== 'dsh-client-gateway') throw new Error(`Port ${port} is occupied by another service`);
      return true;
    } catch (error) { if (error.cause?.code === 'ECONNREFUSED' || error.name === 'TimeoutError') return false; throw error; }
  };
  if (await health()) return;
  const log = openSync(join(dataDirectory, 'gateway.log'), 'a');
  const child = spawn(process.execPath, [fileURLToPath(import.meta.url), '--serve', '--data-dir', dataDirectory, '--dsh-home', dshHome], { detached: true, windowsHide: true, stdio: ['ignore', log, log], cwd: dirname(fileURLToPath(import.meta.url)) });
  child.unref(); closeSync(log);
  for (let attempt = 0; attempt < 100; attempt++) {
    await new Promise(resolve => setTimeout(resolve, 200));
    if (await health()) return;
  }
  throw new Error('Gateway did not start; inspect its local gateway.log');
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  if (process.argv.includes('--install')) await install();
  else if (process.argv.includes('--auth-token')) {
    const token = (await readFile(tokenFile, 'utf8')).trim();
    await ensureServer(token);
    // This entry point is the CLI credential helper. Never run it in a terminal
    // or copy its stdout into diagnostics; the client consumes it privately.
    process.stdout.write(token);
  } else if (process.argv.includes('--serve')) {
    const token = (await readFile(tokenFile, 'utf8')).trim();
    const runtime = await loadRuntime(dshHome);
    if (runtime.failures.length) console.error(JSON.stringify({ catalogWarnings: runtime.failures }));
    const server = gatewayServer(runtime, token, () => loadRuntime(dshHome));
    server.listen(port, host, () => console.log(JSON.stringify({ service: 'dsh-client-gateway', models: runtime.routes.size, pid: process.pid, port })));
  }
}
