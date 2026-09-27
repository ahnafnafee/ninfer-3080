import assert from 'node:assert/strict';
import { test } from 'node:test';
import { once } from 'node:events';
import { translateRequest, responseEvents } from './responses.mjs';
import { gatewayServer } from './server.mjs';
import { clientCatalog } from './runtime.mjs';

const route = { id: 'fixture/model', provider: 'fixture', model: 'model', info: {
  name: 'Fixture', context: { contextWindow: 65536 }, defaultMaxTokens: 8192,
  reasoning: { efforts: [{ id: 'off' }, { id: 'low' }], defaultEffort: 'low' },
} };
async function collect(source) { const result = []; for await (const event of source) result.push(structuredClone(event)); return result; }
async function* chunks(block, finish = { kind: 'stop' }) {
  yield { type: 'block-start', index: 0, blockType: block.type };
  if (block.type === 'text') yield { type: 'text-delta', index: 0, text: block.text };
  yield { type: 'block-end', index: 0, block };
  yield { type: 'usage', usage: { inputTokens: 12, cacheReadTokens: 3, outputTokens: 8, reasoningTokens: 2 } };
  yield { type: 'finish', reason: finish };
}

test('all exact routes enter the catalog with their own context and efforts', () => {
  const models = clientCatalog({ routes: new Map([[route.id, route]]), defaultModel: route.id }).models;
  assert.equal(models[0].slug, route.id);
  assert.equal(models[0].context_window, 65536);
  assert.deepEqual(models[0].supported_reasoning_levels.map(x => x.effort), ['none', 'low']);
  assert.deepEqual(models[0].input_modalities, ['text']);
  assert.equal(models[0].priority, 0);
});

test('instructions, function calls, tool evidence and custom patches survive replay', () => {
  const input = { instructions: 'top instructions', reasoning: { effort: 'none' }, tools: [
    { type: 'function', name: 'read', parameters: { type: 'object' } },
    { type: 'custom', name: 'apply_patch' },
  ], input: [
    { role: 'developer', content: 'developer instructions' },
    { role: 'user', content: [{ type: 'input_text', text: 'fix it' }] },
    { type: 'reasoning', summary: [{ type: 'summary_text', text: 'inspect first' }] },
    { type: 'function_call', name: 'read', call_id: 'c1', arguments: '{"path":"x"}' },
    { type: 'function_call_output', call_id: 'c1', output: 'original source' },
    { type: 'custom_tool_call', name: 'apply_patch', call_id: 'c2', input: '*** Begin Patch\n*** End Patch' },
    { type: 'custom_tool_call_output', call_id: 'c2', output: 'done' },
  ] };
  const before = structuredClone(input);
  const { options, toolMap } = translateRequest(input, route);
  assert.equal(options.system, 'top instructions\n\ndeveloper instructions');
  assert.equal(options.reasoningEffort, 'off');
  assert.equal(options.messages[2].content[0].content[0].text, 'original source');
  assert.deepEqual(JSON.parse(options.messages[3].content[0].arguments), { input: '*** Begin Patch\n*** End Patch' });
  assert.equal(toolMap.get('apply_patch').type, 'custom');
  assert.deepEqual(input, before);
});

test('unsupported inputs and efforts fail explicitly instead of silently discarding data', () => {
  for (const body of [
    { input: [], previous_response_id: 'other' },
    { input: [{ role: 'user', content: [{ type: 'input_image', image_url: 'anything' }] }] },
    { input: [], reasoning: { effort: 'xhigh' } },
    { input: [], tools: [{ type: 'web_search' }] },
  ]) assert.throws(() => translateRequest(body, route));
});

test('text and reasoning preserve terminal status and cached usage', async () => {
  const events = await collect(responseEvents(chunks({ type: 'text', text: 'ready' }), route.id, new Map()));
  assert.equal(events.filter(x => x.type === 'response.output_text.delta').map(x => x.delta).join(''), 'ready');
  const terminal = events.at(-1);
  assert.equal(terminal.type, 'response.completed');
  assert.equal(terminal.response.usage.input_tokens, 15);
  assert.equal(terminal.response.usage.input_tokens_details.cached_tokens, 3);
  assert.equal(terminal.response.usage.total_tokens, 23);
  assert.equal(terminal.response.output[0].content[0].text, 'ready');
  const reasoning = await collect(responseEvents(chunks({ type: 'reasoning', text: 'thought' }), route.id, new Map()));
  assert.equal(reasoning.at(-1).response.output[0].summary[0].text, 'thought');
});

test('custom and namespaced tools round-trip through function-only adapters', async () => {
  const body = { input: [], tools: [{ type: 'namespace', name: 'functions', tools: [{ type: 'custom', name: 'apply_patch' }] }] };
  const { options, toolMap } = translateRequest(body, route);
  assert.equal(options.tools[0].name, 'functions__apply_patch');
  const events = await collect(responseEvents(chunks({ type: 'tool-call', name: 'functions__apply_patch', id: 'c1', arguments: '{"input":"patch text"}' }, { kind: 'tool-calls' }), route.id, toolMap));
  const call = events.at(-1).response.output[0];
  assert.equal(call.type, 'custom_tool_call'); assert.equal(call.namespace, 'functions');
  assert.equal(call.name, 'apply_patch'); assert.equal(call.input, 'patch text');
  assert.equal(call.call_id, 'c1');
});

test('truncation and context errors never become completed responses', async () => {
  for (const [finish, type] of [
    [{ kind: 'max-tokens' }, 'response.incomplete'],
    [{ kind: 'error', failure: { code: 'CONTEXT_WINDOW_EXCEEDED', message: 'too large' } }, 'response.failed'],
  ]) {
    const events = await collect(responseEvents(chunks({ type: 'text', text: 'partial' }, finish), route.id, new Map()));
    assert.equal(events.at(-1).type, type);
    assert.ok(!events.some(e => e.type === 'response.completed'));
    const tools = new Map([['write', { type: 'function', name: 'write' }]]);
    const failedCall = await collect(responseEvents(chunks({ type: 'tool-call', name: 'write', id: 'c', arguments: '{}' }, finish), route.id, tools));
    assert.ok(!failedCall.some(e => e.type === 'response.output_item.done'), 'failed generations expose no executable tool call');
  }
});

test('HTTP authentication, routing, SSE and disconnect cancellation work end to end', async t => {
  let sawAbort = false;
  const adapter = { async *stream(options) {
    assert.equal(options.provider, 'fixture'); assert.equal(options.model, 'model');
    if (options.messages[0]?.content[0]?.text === 'wait') {
      await new Promise(resolve => options.signal.addEventListener('abort', () => { sawAbort = true; resolve(); }, { once: true }));
      yield { type: 'finish', reason: { kind: 'aborted', failure: { code: 'ABORTED', message: 'aborted' } } }; return;
    }
    yield* chunks({ type: 'text', text: 'ready' });
  } };
  const server = gatewayServer({ routes: new Map([[route.id, { ...route, adapter }]]) }, 'test-only');
  server.listen(0, '127.0.0.1'); await once(server, 'listening');
  t.after(() => { server.closeAllConnections(); server.close(); });
  const base = `http://127.0.0.1:${server.address().port}`;
  assert.equal((await fetch(base + '/v1/models')).status, 401);
  const headers = { authorization: 'Bearer test-only', 'content-type': 'application/json' };
  const models = await (await fetch(base + '/v1/models', { headers })).json();
  assert.equal(models.data[0].id, route.id);
  assert.equal((await fetch(base + '/v1/responses', { method: 'POST', headers, body: JSON.stringify({ model: 'missing', input: 'x' }) })).status, 404);
  const response = await fetch(base + '/v1/responses', { method: 'POST', headers, body: JSON.stringify({ model: route.id, input: 'x', stream: true }) });
  const text = await response.text(); assert.match(text, /event: response.completed/); assert.match(text, /ready/);
  const cancel = new AbortController();
  const pending = await fetch(base + '/v1/responses', { method: 'POST', headers, body: JSON.stringify({ model: route.id, input: 'wait', stream: true }), signal: cancel.signal });
  await pending.body.getReader().read(); cancel.abort();
  for (let i = 0; i < 20 && !sawAbort; i++) await new Promise(resolve => setTimeout(resolve, 10));
  assert.equal(sawAbort, true);
});

test('catalog reload is authenticated and preserves working routes on failure', async t => {
  let fail = false;
  const initial = { routes: new Map([[route.id, route]]) };
  const next = { routes: new Map([['fixture/second', { ...route, id: 'fixture/second' }]]), failures: [] };
  const server = gatewayServer(initial, 'test-only', async () => {
    if (fail) return { ...initial, failures: [{ message: 'offline' }] };
    return next;
  });
  server.listen(0, '127.0.0.1'); await once(server, 'listening');
  t.after(() => { server.closeAllConnections(); server.close(); });
  const base = `http://127.0.0.1:${server.address().port}`;
  assert.equal((await fetch(base + '/reload', { method: 'POST' })).status, 401);
  const headers = { authorization: 'Bearer test-only' };
  assert.equal((await fetch(base + '/reload', { method: 'POST', headers })).status, 200);
  fail = true;
  assert.equal((await fetch(base + '/reload', { method: 'POST', headers })).status, 503);
  const models = await (await fetch(base + '/v1/models', { headers })).json();
  assert.equal(models.data[0].id, 'fixture/second');
});
