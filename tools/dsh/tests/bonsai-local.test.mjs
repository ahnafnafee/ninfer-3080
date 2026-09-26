import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mkdtemp, realpath, rm } from 'node:fs/promises';
import { createServer } from 'node:http';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { tmpdir } from 'node:os';
import { pathToFileURL } from 'node:url';
import { test } from 'node:test';

const requireDsh = createRequire(join(dirname(process.execPath), 'node_modules/@deepseek-ai/dsh/package.json'));
const moduleOf = name => import(pathToFileURL(requireDsh.resolve(name)));
const yaml = requireDsh('js-yaml');
const schema = yaml.DEFAULT_SCHEMA.extend(new yaml.Type('tag:yaml.org,2002:js', { kind: 'scalar' }));
const readYaml = name => {
  try { return yaml.load(readFileSync(new URL(`../${name}`, import.meta.url), 'utf8'), { schema }); }
  catch (error) { throw new Error(`${name}: ${error.reason ?? 'YAML parse failure'} at line ${(error.mark?.line ?? 0) + 1}`); }
};
const settings = readYaml('settings.example.yaml');
const preset = readYaml('bonsai2-heretic-3080/agent.cordis.yml');
const patch = readYaml('host.cordis.patch.example.yml');
const compact = preset.find(row => row.id === 'compaction').config.find(row => row.id === 'compaction-basic').config;
const nativeRows = patch.flatMap(operation => operation.insert ?? []).filter(row => row.id === 'native-ninfer');
const { Context } = await moduleOf('@deepseek-ai/cordis');
const { default: LlmRuntime, createUserMessage } = await moduleOf('@deepseek-ai/dsh-llm');
const { default: CompactWithoutTools } = await import('../bonsai2-heretic-3080/compact-without-tools.mjs');

test('saved Bonsai routes use the exact-count adapter and medium reasoning without duplicate provider ownership', () => {
  assert.equal(nativeRows.length, 1, 'one host-level native adapter owns both local routes');
  const route = settings['agent-default-model'];
  assert.equal(route.provider, 'qwen-3080');
  assert.equal(route.model, nativeRows[0].config.model);
  assert.equal(route.reasoningEffort, 'medium');
  assert.equal(settings['llm-pi-ai']?.providers?.['qwen-3080'], undefined);
  assert.equal(settings['llm-pi-ai']?.providers?.['qwen-3080-summary'], undefined);
  assert.equal(compact.summarizationProvider, 'qwen-3080-summary');
  assert.equal(compact.summarizationModel, route.model);
  assert.equal(compact.maxTokens, nativeRows[0].config.summaryMaxTokens);
  assert.ok(nativeRows[0].config.minOutputTokens >= 4096, 'reserve room beyond the server 2048-token thinking budget');
  assert.ok(nativeRows[0].config.minOutputTokens + nativeRows[0].config.reserveTokens < 65536 * (1 - compact.thresholdRatio));
});

test('configured host plugin and Bonsai compaction request fit the active window with thinking and tools disabled for summaries', async () => {
  const requests = [];
  const model = nativeRows[0].config.model;
  const server = createServer(async (req, res) => {
    const buffers = [];
    for await (const buffer of req) buffers.push(buffer);
    const body = buffers.length ? JSON.parse(Buffer.concat(buffers).toString()) : undefined;
    requests.push({ path: req.url, body });
    res.setHeader('content-type', 'application/json');
    if (req.url === '/v1/models') res.end(JSON.stringify({ data: [{ id: model, context_window: 8192 }] }));
    else if (req.url === '/v1/responses/input_tokens') res.end(JSON.stringify({ input_tokens: 600 }));
    else {
      assert.equal(req.url, '/v1/responses');
      res.setHeader('content-type', 'text/event-stream');
      const events = [
        { type: 'response.output_text.delta', output_index: 0, content_index: 0, delta: 'Goal: repair Bonsai context handling. Next: verify the configured native adapter.' },
        { type: 'response.output_text.done', output_index: 0, content_index: 0, text: 'Goal: repair Bonsai context handling. Next: verify the configured native adapter.' },
        { type: 'response.completed', response: { status: 'completed', usage: { input_tokens: 600, output_tokens: 20, total_tokens: 620 } } },
      ];
      res.end(events.map(event => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`).join(''));
    }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const workspace = await mkdtemp(join(tmpdir(), 'bonsai-local-test-'));
  const ctx = new Context();
  try {
    await ctx.plugin(LlmRuntime);
    const plugin = await import('../bonsai2-heretic-3080/native-ninfer.mjs');
    await ctx.plugin(plugin, { ...nativeRows[0].config, baseURL: `http://127.0.0.1:${server.address().port}/v1` });
    const agent = { options: { provider: 'qwen-3080', model }, session: { id: 'bonsai-compaction-check', header: { cwd: workspace }, requestHeader: () => ({ config: { provider: 'qwen-3080', model } }) } };
    const result = await CompactWithoutTools.prototype.summarize.call({ ctx, config: { modelPolicies: [], ...compact } }, {
      messages: [createUserMessage({ content: [{ type: 'text', text: 'We found that failed summaries allowed an old conversation to exceed the context. Preserve the repair and next step.' }], source: { kind: 'user' } })],
      tools: [{ name: 'unused_catalog_tool', description: 'A large catalog belongs only on ordinary calls.', parameters: { type: 'object' } }],
    }, agent, AbortSignal.timeout(5000));
    assert.ok(result.summary.some(block => block.text.includes('Goal: repair Bonsai')));
    const generated = requests.filter(request => request.path === '/v1/responses');
    const counted = requests.filter(request => request.path === '/v1/responses/input_tokens');
    assert.equal(generated.length, 1);
    assert.equal(counted.length, 1);
    assert.equal(generated[0].body.reasoning.effort, 'none');
    assert.equal(generated[0].body.max_output_tokens, compact.maxTokens);
    assert.ok(!generated[0].body.tools);
    assert.deepEqual(generated[0].body.input, counted[0].body.input);
  } finally {
    await ctx.fiber.dispose();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    assert.equal(dirname(await realpath(workspace)), await realpath(tmpdir()));
    await rm(workspace, { recursive: true, force: true });
  }
});
