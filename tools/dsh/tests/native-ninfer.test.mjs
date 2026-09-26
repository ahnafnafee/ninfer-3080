import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createServer } from 'node:http';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { NativeNinferAdapter, responseInput, responseRequest, responseChunks, sseEvents, apply } from '../bonsai2-heretic-3080/native-ninfer.mjs';

const requireDsh = createRequire(join(dirname(process.execPath), 'node_modules/@deepseek-ai/dsh/package.json'));
const { Context } = await import(pathToFileURL(requireDsh.resolve('@deepseek-ai/cordis')));
const { default: LlmRuntime, createUserMessage } = await import(pathToFileURL(requireDsh.resolve('@deepseek-ai/dsh-llm')));
const model = 'bonsai2-heretic';
const options = () => ({ provider: 'qwen-3080', model, messages: [createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text: 'Explain the next repair.' }] })] });
const collect = async iterable => { const output = []; for await (const value of iterable) output.push(value); return output; };
test('short session titles reserve their output for visible text', () => {
  assert.equal(responseRequest({ ...options(), purpose: 'session-title', maxTokens: 64 }).reasoning.effort, 'none');
  assert.equal(responseRequest(options()).reasoning.effort, 'medium');
});
async function* events(values) { yield* values; }
const textEvents = (status = 'completed') => [
  { type: 'response.reasoning_text.delta', output_index: 0, content_index: 0, delta: 'Check ' },
  { type: 'response.reasoning_text.delta', output_index: 0, content_index: 0, delta: 'it.' },
  { type: 'response.reasoning_text.done', output_index: 0, content_index: 0, text: 'Check it.' },
  { type: 'response.output_text.delta', output_index: 1, content_index: 0, delta: 'Done.' },
  { type: 'response.output_text.done', output_index: 1, content_index: 0, text: 'Done.' },
  { type: `response.${status}`, response: { status, usage: { input_tokens: 40, input_tokens_details: { cached_tokens: 10 }, output_tokens: 8, output_tokens_details: { reasoning_tokens: 4 }, total_tokens: 48 } } },
];

async function mockNinfer(testCase) {
  const state = { capacity: 8192, count: 1000, generation: textEvents(), requests: [], countError: null };
  const server = createServer(async (request, response) => {
    const buffers = [];
    for await (const part of request) buffers.push(part);
    const body = buffers.length ? JSON.parse(Buffer.concat(buffers).toString()) : undefined;
    state.requests.push({ path: request.url, headers: request.headers, body });
    response.setHeader('content-type', 'application/json');
    if (request.url === '/v1/models') response.end(JSON.stringify({ data: [{ id: model, context_window: state.capacity }] }));
    else if (request.url === '/v1/responses/input_tokens') {
      if (state.countError) { response.statusCode = 400; response.end(JSON.stringify({ error: state.countError })); }
      else response.end(JSON.stringify({ input_tokens: state.count }));
    } else if (request.url === '/v1/responses') {
      response.setHeader('content-type', 'text/event-stream');
      for (const event of state.generation) response.write(`event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`);
      response.end();
    } else { response.statusCode = 404; response.end('{}'); }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const config = { baseURL: `http://127.0.0.1:${server.address().port}/v1` };
  try { await testCase(state, config); }
  finally { server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); }
}

test('native route discovers current capacity and counts identical tool/prompt input before generation', async () => {
  await mockNinfer(async (state, config) => {
    const adapter = new NativeNinferAdapter(config);
    const metadata = await adapter.resolveModel('qwen-3080', model);
    assert.equal(metadata.context.contextWindow, 8192);
    assert.equal(metadata.reasoning.defaultEffort, 'medium');
    const request = { ...options(), maxTokens: 32768, system: 'Keep the user instructions.', tools: [{ name: 'read', description: 'Read a file.', parameters: { type: 'object', properties: { path: { type: 'string' } }, required: ['path'] } }] };
    const chunks = await collect(adapter.stream(request));
    const counted = state.requests.find(call => call.path.endsWith('/input_tokens'));
    const generated = state.requests.find(call => call.path === '/v1/responses');
    for (const key of Object.keys(counted.body)) assert.deepEqual(generated.body[key], counted.body[key]);
    assert.equal(generated.body.max_output_tokens, 4096, 'the pi-ai 4096 safety deduction must not shrink this request');
    assert.equal(generated.body.reasoning.effort, 'medium');
    assert.equal(generated.body.store, false);
    assert.ok(state.requests.every(call => call.headers['user-agent']));
    assert.deepEqual(chunks.find(chunk => chunk.type === 'usage').usage, { inputTokens: 30, cacheReadTokens: 10, outputTokens: 8, reasoningTokens: 4, totalTokens: 48 });
    assert.deepEqual(chunks.at(-1), { type: 'finish', reason: { kind: 'stop' } });
    state.capacity = 4096;
    assert.equal((await adapter.resolveModel('qwen-3080', model)).context.contextWindow, 4096, 'server profile changes must not retain stale 64K metadata');
    state.capacity = 65536;
    assert.equal((await adapter.prepareCall('qwen-3080', model)).model.context.contextWindow, 65536, 'switching to 64K must immediately publish the loaded capacity');
  });
});

test('capacity boundary reserves answer space, caps output honestly, and requests compaction before an HTTP generation error', async () => {
  await mockNinfer(async (state, config) => {
    const adapter = new NativeNinferAdapter(config);
    state.count = 8192 - 256 - 4096;
    const chunks = await collect(adapter.stream(options()));
    assert.equal(chunks.at(-1).reason.kind, 'stop');
    assert.equal(state.requests.find(call => call.path === '/v1/responses').body.max_output_tokens, 4096);
    state.requests.length = 0;
    state.count++;
    const rejected = await collect(adapter.stream(options()));
    assert.equal(rejected.at(-1).reason.failure.code, 'CONTEXT_WINDOW_EXCEEDED');
    assert.ok(!state.requests.some(call => call.path === '/v1/responses'));
    state.countError = { code: 'context_length_exceeded', message: 'prepared prompt exceeds Engine max_context 8192' };
    const oversized = await collect(adapter.stream(options()));
    assert.equal(oversized.at(-1).reason.failure.code, 'CONTEXT_WINDOW_EXCEEDED');
  });
});

test('summary route disables thinking and reserves the complete summary output budget', async () => {
  await mockNinfer(async (state, config) => {
    const adapter = new NativeNinferAdapter(config);
    const request = { ...options(), provider: 'qwen-3080-summary', reasoningEffort: 'high' };
    const chunks = await collect(adapter.stream(request));
    assert.equal(chunks.at(-1).reason.kind, 'stop');
    const generation = state.requests.find(call => call.path === '/v1/responses').body;
    assert.equal(generation.max_output_tokens, 768);
    assert.equal(generation.reasoning.effort, 'none');
    assert.ok(!generation.tools);
    state.count = 8192 - 256 - 767;
    assert.equal((await collect(adapter.stream(request))).at(-1).reason.failure.code, 'CONTEXT_WINDOW_EXCEEDED');
  });
});

test('complete tool calls remain correlated with tool results and preserve reasoning in replay', () => {
  const input = responseInput([
    { role: 'user', content: [{ type: 'text', text: 'Read it.' }] },
    { role: 'assistant', content: [{ type: 'reasoning', text: 'Need the file.' }, { type: 'tool-call', id: 'call_1', name: 'read', arguments: '{"path":"README.md"}' }] },
    { role: 'user', content: [{ type: 'tool-result', toolCallId: 'call_1', content: [{ type: 'text', text: 'The file contents.' }] }] },
    { role: 'assistant', content: [{ type: 'tool-call', id: 'call_2', name: 'write', arguments: '{"path":' }] },
    { role: 'user', content: [{ type: 'text', text: 'continue' }] },
  ]);
  assert.equal(input[1].type, 'reasoning');
  assert.deepEqual(input[2], { type: 'function_call', call_id: 'call_1', name: 'read', arguments: '{"path":"README.md"}' });
  assert.equal(input[3].call_id, 'call_1');
  assert.equal(input[4].type, 'message');
  assert.match(input[4].content, /Unexecuted tool call write/);
  assert.equal(input.at(-1).content, 'continue');
});

test('length-limited tool JSON is never executed and prior output remains visible', async () => {
  for (const argumentsText of ['{"path":', '{"path":"result.txt"}']) {
    const chunks = await collect(responseChunks(events([
      ...textEvents().slice(0, -1),
      { type: 'response.output_item.done', output_index: 2, item: { type: 'function_call', call_id: 'call_1', name: 'write', arguments: argumentsText, status: 'incomplete' } },
      { type: 'response.incomplete', response: { status: 'incomplete', incomplete_details: { reason: 'max_output_tokens' } } },
    ])));
    assert.equal(chunks.at(-1).reason.kind, 'max-tokens');
    assert.ok(!chunks.some(chunk => chunk.blockType === 'tool-call' || chunk.block?.type === 'tool-call'));
    assert.ok(chunks.some(chunk => chunk.block?.text === 'Done.'));
    assert.ok(chunks.some(chunk => chunk.block?.text.includes('Unexecuted tool call write')));
  }
});

test('completed tools produce the complete raw JSON and proper tool-calls finish', async () => {
  const chunks = await collect(responseChunks(events([
    { type: 'response.output_item.done', output_index: 0, item: { type: 'function_call', call_id: 'call_7', name: 'read', arguments: '{"path":"README.md"}', status: 'completed' } },
    { type: 'response.completed', response: { status: 'completed' } },
  ])));
  assert.deepEqual(chunks.find(chunk => chunk.type === 'block-end').block, { type: 'tool-call', id: 'call_7', name: 'read', arguments: '{"path":"README.md"}' });
  assert.equal(chunks.at(-1).reason.kind, 'tool-calls');
});

test('SSE parsing handles UTF-8 and CRLF boundaries and refuses missing terminal events', async () => {
  const data = Buffer.from(': keep-alive\r\n\r\nevent: delta\r\ndata: {"type":"test","text":"λ"}\r\n\r\n');
  const stream = new ReadableStream({ start(controller) { for (const byte of data) controller.enqueue(new Uint8Array([byte])); controller.close(); } });
  assert.deepEqual(await collect(sseEvents(stream)), [{ type: 'test', text: 'λ' }]);
  await assert.rejects(collect(responseChunks(events(textEvents().slice(0, -1)))), /before its terminal/);
});

test('native adapter registers through the actual DSH LLM service and binds defaults', async () => {
  await mockNinfer(async (_state, config) => {
    const ctx = new Context();
    try {
      await ctx.plugin(LlmRuntime);
      apply(ctx, config);
      const prepared = await ctx.llm.prepareCall({ provider: 'qwen-3080', model });
      assert.equal(prepared.config.reasoningEffort, 'medium');
      assert.equal(prepared.config.maxTokens, 4096);
      assert.equal(prepared.context.contextWindow, 8192);
      const chunks = await collect(prepared.stream({ ...options(), ...prepared.config }));
      assert.equal(chunks.at(-1).reason.kind, 'stop');
    } finally { await ctx.fiber.dispose(); }
  });
});
