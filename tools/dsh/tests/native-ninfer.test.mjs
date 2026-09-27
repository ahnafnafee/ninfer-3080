import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createServer } from 'node:http';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { NativeNinferAdapter, responseInput, responseRequest, responseChunks, sseEvents, apply } from '../bonsai2-heretic-3080/native-ninfer.mjs';
import { recoveryNotice } from '../bonsai2-heretic-3080/loop-guard.mjs';

const requireDsh = createRequire(join(dirname(process.execPath), 'node_modules/@deepseek-ai/dsh/package.json'));
const { Context } = await import(pathToFileURL(requireDsh.resolve('@deepseek-ai/cordis')));
const { default: LlmRuntime, createUserMessage } = await import(pathToFileURL(requireDsh.resolve('@deepseek-ai/dsh-llm')));
const model = 'bonsai2-heretic';
const options = () => ({ provider: 'qwen-3080', model, messages: [createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text: 'Explain the next repair.' }] })] });
const collect = async iterable => { const output = []; for await (const value of iterable) output.push(value); return output; };
test('short session titles reserve their output for visible text', () => {
  assert.equal(responseRequest({ ...options(), purpose: 'session-title', maxTokens: 64 }).reasoning.effort, 'none');
  assert.equal(responseRequest(options()).reasoning.effort, 'low');
});

test('loop recovery removes prior reasoning but preserves evidence, constraints and later reasoning', async () => {
  await mockNinfer(async (state, config) => {
    const beforeRecovery = [
      ...options().messages,
      { role: 'assistant', content: [{ type: 'reasoning', text: 'Repeat the mistaken offset calculation.' }, { type: 'tool-call', id: 'read_bytes', name: 'pwsh', arguments: '{"command":"read bytes"}' }] },
      { role: 'user', content: [{ type: 'tool-result', toolCallId: 'read_bytes', isError: false, content: [{ type: 'text', text: 'Byte 9897 is 0d; byte 9899 is 7d.' }] }] },
    ];
    const messages = [...beforeRecovery, recoveryNotice(),
      { role: 'assistant', content: [{ type: 'reasoning', text: 'Measure encoded bytes independently.' }, { type: 'text', text: 'The source file remains unchanged.' }] },
    ];
    const saved = structuredClone(messages);
    const chunks = await collect(new NativeNinferAdapter(config).stream({ ...options(), messages }));
    const body = state.requests.find(call => call.path === '/v1/responses').body;
    assert.deepEqual(body.input.filter(item => item.type === 'reasoning'), [{ type: 'reasoning', content: [{ type: 'reasoning_text', text: 'Measure encoded bytes independently.' }] }]);
    for (const item of responseInput(beforeRecovery).filter(item => item.type !== 'reasoning')) assert.ok(body.input.some(value => JSON.stringify(value) === JSON.stringify(item)));
    assert.equal(chunks.at(-1).replayState.response.omittedReasoningBlocks, 1);
    assert.deepEqual(state.requests.find(call => call.path.endsWith('/input_tokens')).body.input, body.input);
    assert.deepEqual(messages, saved);
    const ordinaryUserText = { ...recoveryNotice(), source: { kind: 'user' } };
    assert.equal(responseInput([...beforeRecovery, ordinaryUserText]).filter(item => item.type === 'reasoning').length, 1, 'ordinary text cannot masquerade as a recovery event');
  });
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
      const countError = typeof state.countError === 'function' ? state.countError(body) : state.countError;
      if (countError) { response.statusCode = 400; response.end(JSON.stringify({ error: countError })); }
      else response.end(JSON.stringify({ input_tokens: typeof state.count === 'function' ? state.count(body) : state.count }));
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
    assert.equal(metadata.reasoning.defaultEffort, 'low');
    const request = { ...options(), maxTokens: 32768, system: 'Keep the user instructions.', tools: [{ name: 'read', description: 'Read a file.', parameters: { type: 'object', properties: { path: { type: 'string' } }, required: ['path'] } }] };
    const chunks = await collect(adapter.stream(request));
    const counted = state.requests.find(call => call.path.endsWith('/input_tokens'));
    const generated = state.requests.find(call => call.path === '/v1/responses');
    for (const key of Object.keys(counted.body)) assert.deepEqual(generated.body[key], counted.body[key]);
    assert.equal(generated.body.max_output_tokens, 4096, 'the pi-ai 4096 safety deduction must not shrink this request');
    assert.equal(generated.body.reasoning.effort, 'low');
    assert.equal(generated.body.temperature, 0.2);
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

test('exact preflight reclaims older reasoning before rejecting a verified long tool conversation', async () => {
  await mockNinfer(async (state, config) => {
    state.capacity = 65536;
    state.count = body => body.input.filter(item => item.type === 'reasoning').length > 1 ? 62627 : 54000;
    const messages = [
      { role: 'user', content: [{ type: 'text', text: 'Keep every requirement and original source.' }] },
      { role: 'assistant', content: [{ type: 'reasoning', text: 'Old exploratory thinking.' }, { type: 'tool-call', id: 'call_1', name: 'write', arguments: '{"content":"$numFiles = 3"}' }] },
      { role: 'user', content: [{ type: 'tool-result', toolCallId: 'call_1', content: [{ type: 'text', text: 'Wrote the source.' }] }] },
      { role: 'assistant', content: [{ type: 'reasoning', text: 'Newest verification reasoning.' }, { type: 'tool-call', id: 'call_2', name: 'pwsh', arguments: '{"command":"node verify.cjs"}' }] },
      { role: 'user', content: [{ type: 'tool-result', toolCallId: 'call_2', content: [{ type: 'text', text: '126 checks passed.' }] }] },
    ];
    const before = structuredClone(messages);
    const chunks = await collect(new NativeNinferAdapter({ ...config, maxTokens: 8192 }).stream({ ...options(), messages }));
    assert.equal(chunks.at(-1).reason.kind, 'stop');
    const counts = state.requests.filter(call => call.path.endsWith('/input_tokens'));
    assert.equal(chunks.at(-1).replayState.response.omittedReasoningBlocks, 1);
    const generation = state.requests.find(call => call.path === '/v1/responses').body;
    assert.equal(counts.length, 2);
    assert.deepEqual(generation.input, counts.at(-1).body.input, 'count the actual shortened prompt');
    assert.deepEqual(generation.input.filter(item => item.type !== 'reasoning'), responseInput(messages).filter(item => item.type !== 'reasoning'));
    assert.deepEqual(generation.input.filter(item => item.type === 'reasoning'), [{ type: 'reasoning', content: [{ type: 'reasoning_text', text: 'Newest verification reasoning.' }] }]);
    assert.deepEqual(messages, before, 'durable caller history is never rewritten');
    state.requests.length = 0;
    state.countError = body => body.input.filter(item => item.type === 'reasoning').length > 1
      ? { code: 'context_length_exceeded', message: 'prepared prompt exceeds Engine max_context 65536' } : null;
    const recovered = await collect(new NativeNinferAdapter({ ...config, maxTokens: 8192 }).stream({ ...options(), messages }));
    assert.equal(recovered.at(-1).reason.kind, 'stop', 'also recover when the full prompt cannot be counted');
    assert.deepEqual(state.requests.find(call => call.path === '/v1/responses').body.input, generation.input);
    state.requests.length = 0;
    state.countError = null;
    state.count = 1000;
    const roomy = await collect(new NativeNinferAdapter(config).stream({ ...options(), messages }));
    assert.equal(roomy.at(-1).replayState?.response?.omittedReasoningBlocks, undefined);
    assert.deepEqual(state.requests.find(call => call.path === '/v1/responses').body.input, responseInput(messages), 'reasoning is retained when it fits');
  });
});

test('explicit reasoning and temperature choices override coding defaults', async () => {
  await mockNinfer(async (state, config) => {
    const adapter = new NativeNinferAdapter({ ...config, temperature: 0.3 });
    await collect(adapter.stream({ ...options(), reasoningEffort: 'medium', temperature: 0 }));
    const request = state.requests.find(call => call.path === '/v1/responses').body;
    assert.equal(request.reasoning.effort, 'medium');
    assert.equal(request.temperature, 0);
    assert.equal(responseRequest({ ...options(), reasoningEffort: 'off' }).reasoning.effort, 'none');
  });
  for (const temperature of [NaN, Infinity, -1, 3]) assert.throws(() => new NativeNinferAdapter({ temperature }), /temperature/);
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
      assert.equal(prepared.config.reasoningEffort, 'low');
      assert.equal(prepared.config.maxTokens, 4096);
      assert.equal(prepared.context.contextWindow, 8192);
      const chunks = await collect(prepared.stream({ ...options(), ...prepared.config }));
      assert.equal(chunks.at(-1).reason.kind, 'stop');
    } finally { await ctx.fiber.dispose(); }
  });
});
