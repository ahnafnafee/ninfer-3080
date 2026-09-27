import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { responseChunks } from '../bonsai2-heretic-3080/native-ninfer.mjs';
const requireDsh = createRequire(join(dirname(process.execPath), 'node_modules/@deepseek-ai/dsh/package.json'));
const { AssistantStreamAccumulator } = await import(pathToFileURL(requireDsh.resolve('@deepseek-ai/dsh-llm')));
const stats = await import('../bonsai2-heretic-3080/ninfer-session-stats.mjs');
let definition;
stats.apply({ sessionProjections: { register(value) { definition = value; } } });

async function replay(timings) {
  // The server emits parsed tool JSON in one burst after four seconds of work.
  const stream = new AssistantStreamAccumulator();
  let finish;
  for await (const chunk of responseChunks((async function* () {
    yield { type: 'response.output_item.done', output_index: 0, item: { type: 'function_call', call_id: 'call_1', name: 'pwsh', arguments: '{"command":"numFiles = 1"}', status: 'completed' } };
    yield { type: 'response.completed', response: { usage: { input_tokens: 100, output_tokens: 556 }, ...(timings ? { timings } : {}) } };
  })())) {
    stream.push({ time: 4999, chunk });
    if (chunk.type === 'finish') finish = chunk;
  }
  let state = definition.apply(definition.init(), { type: 'step/start', time: 1000, data: { turn: 1, step: 1 } });
  return definition.apply(state, { type: 'assistant/message', time: 5000, data: {
    turn: 1, step: 1, stream: stream.snapshot(), usage: { outputTokens: 556 },
    message: { role: 'assistant', content: [{ type: 'tool-call' }], source: { provider: 'qwen-3080', model: 'bonsai2-heretic', replayState: finish.replayState } },
  } });
}

test('old buffered tool-only records do not report hundreds of thousands of tokens/sec', async () => {
  const state = await replay();
  assert.equal(state.llmMs, 4000);
  assert.equal(state.decodeTokens, 0, 'no recoverable decode measurement in the old stream');
  assert.equal(state.decodeMs, 0);
  assert.equal(state.ttftSteps, 0, 'buffer flush is not first-token latency');
});

test('tool-only replies use server generation intervals instead of buffer-flush duration', async () => {
  const state = await replay({ predicted_n: 556, predicted_ms: 3500, ttft_ms: 500 });
  assert.equal(state.decodeMs, 3500);
  assert.equal(state.decodeTokens, 555, 'N tokens span N-1 decode intervals');
  assert.equal(state.ttftMs, 500);
  assert.ok(Math.abs(state.decodeTokens * 1000 / state.decodeMs - 158.5714) < 0.001);
});

test('missing, inconsistent, or nonfinite server timings never become decode measurements', async () => {
  for (const timing of [ { predicted_n: 555, predicted_ms: 3500, ttft_ms: 500 },
    { predicted_n: 556, predicted_ms: 0, ttft_ms: 500 },
    { predicted_n: 556, predicted_ms: Infinity, ttft_ms: 500 },
    { predicted_n: 556, predicted_ms: 3500, ttft_ms: -1 } ]) {
    assert.equal((await replay(timing)).decodeTokens, 0);
  }
});

test('other providers retain the stock timing fold', () => {
  let state = definition.apply(definition.init(), { type: 'step/start', time: 1000, data: { turn: 1, step: 1 } });
  state = definition.apply(state, { type: 'assistant/message', time: 5000, data: {
    turn: 1, step: 1, stream: [{ type: 'text-chunks', time0: 1500, index: 0, dt: [], texts: ['hello'] }],
    usage: { outputTokens: 556 }, message: { source: { provider: 'another-provider' } },
  } });
  assert.equal(state.decodeMs, 3500);
  assert.equal(state.decodeTokens, 556);
});
