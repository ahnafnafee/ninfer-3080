import assert from 'node:assert/strict';
import { mkdtemp, realpath, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { after, test } from 'node:test';
import CompactWithoutTools from '../bonsai2-heretic-3080/compact-without-tools.mjs';

const requireDsh = createRequire(join(dirname(process.execPath), 'node_modules/@deepseek-ai/dsh/package.json'));
const moduleOf = (name) => import(pathToFileURL(requireDsh.resolve(name)));
const { Context } = await moduleOf('@deepseek-ai/cordis');
const { Session } = await moduleOf('@deepseek-ai/dsh-session');
const { default: SessionProjections } = await moduleOf('@deepseek-ai/dsh-session-projection');
const { default: TokenMeter } = await moduleOf('@deepseek-ai/dsh-token-meter');
const { default: ToolResultPruner } = await moduleOf('@deepseek-ai/dsh-compaction-tool-result-pruner');
const { createUserMessage, createSystemMessage, createAssistantMessage, createToolResultMessage, CONTEXT_WINDOW_EXCEEDED_CODE } = await moduleOf('@deepseek-ai/dsh-llm');
const workspace = await mkdtemp(join(tmpdir(), 'bonsai-compaction-recovery-'));
after(async () => {
  assert.equal(dirname(await realpath(workspace)), await realpath(tmpdir()));
  await rm(workspace, { recursive: true, force: true });
});

async function setup({ capacity = 8192, compactConfig = {}, summaryResponse = 'The user wants main.cpp repaired. Earlier work inspected files; preserve pending tests and continue with the latest request.' } = {}) {
  const ctx = new Context();
  const calls = [];
  ctx.provide('llm', {
    resolveModelInfo: async () => ({ context: { contextWindow: capacity } }),
    imageRequestPricing: () => undefined,
    fileRequestText: () => undefined,
    async *stream(call) {
      calls.push(call);
      yield { type: 'text-delta', index: 0, text: summaryResponse };
      yield { type: 'finish', reason: { kind: 'stop' } };
    },
  });
  await ctx.plugin(SessionProjections);
  await ctx.plugin(TokenMeter);
  const engine = new CompactWithoutTools(ctx, {
    thresholdRatio: 0.45, retainTokens: 256, maxTokens: 768,
    summarizationProvider: 'qwen-3080-summary', summarizationModel: 'bonsai2-heretic',
    compactionRetries: 2, maxOverflowRetries: 1,
    ...compactConfig,
  });
  const session = Session.create('context-recovery-fixture', undefined, { version: 3, id: 'context-recovery-fixture', createdAt: Date.now(), isSeeded: false, cwd: workspace });
  const agent = { options: { provider: 'qwen-3080', model: 'bonsai2-heretic' }, session };
  const user = (text) => session.append('user/message', createUserMessage({ content: [{ type: 'text', text }], source: { kind: 'user' } }), { surfaceOp: 'append' });
  const header = (tools) => session.append('request/header', { header: { config: agent.options, ...(tools ? { tools } : {}) }, reason: 'change' });
  session.append('system/message', { turn: 0, step: 0, message: createSystemMessage('You are a coding assistant.', 'test') }, { surfaceOp: 'append' });
  session.append('turn/start', { turn: 1 });
  return { ctx, engine, session, agent, calls, user, header };
}

function appendToolTurn(session, agent, outputs, closed = true) {
  session.append('step/start', { turn: 1, step: 1 });
  const assistant = createAssistantMessage({
    content: outputs.map((_output, index) => ({ type: 'tool-call', id: `file-${index}`, name: 'pwsh', arguments: '{"command":"Get-Content main.cpp"}' })),
    source: agent.options,
  });
  const call = session.append('assistant/message', { turn: 1, step: 1, message: assistant, stream: [] }, { surfaceOp: 'append' });
  const results = outputs.map((output, index) => session.append('tool/result', {
    turn: 1, step: 1,
    message: createToolResultMessage({ callId: `file-${index}`, content: [{ type: 'text', text: output }], isError: false }),
  }, { surfaceOp: 'append' }));
  if (closed) session.append('step/end', { turn: 1, step: 1 });
  return { call, results };
}

test('recoverable exact-preflight error compacts restored oversized history after the tool header refreshes', async () => {
  const { ctx, session, agent, calls, user, header } = await setup();
  try {
    header([{ name: 'old_tool', description: 'OLD CATALOG '.repeat(20000), parameters: { type: 'object' } }]);
    for (let index = 0; index < 12; index += 1) user(`Historical message ${index}: ` + 'Keep source paths and pending repairs. '.repeat(800));
    const latest = user('Continue repairing main.cpp and verify the result.');
    const originals = session.snapshotEvents();
    assert.ok(ctx.tokenMeter.measure(session).totalTokens > 65536);
    header([{ name: 'pwsh', description: 'Run a command.', parameters: { type: 'object', properties: { command: { type: 'string' } } } }]);
    const generation = session.surface.replaceGeneration;
    const action = await ctx.waterfall('agent/request-error', { agent, failure: { code: CONTEXT_WINDOW_EXCEEDED_CODE, message: 'Prompt leaves less than the output reservation.' }, signal: AbortSignal.timeout(5000) }, () => undefined);
    assert.deepEqual(action, { kind: 'retry' });
    assert.ok(session.surface.replaceGeneration > generation);
    assert.ok(ctx.tokenMeter.measure(session).totalTokens < 2867);
    assert.equal(session.deriveMessages().at(-1).content[0].text, latest.data.content[0].text, 'latest user request stays verbatim');
    assert.deepEqual(session.snapshotEvents(0, originals.length), originals, 'compaction must preserve the original append-only history');
    assert.ok(calls.length > 20);
    assert.ok(calls.every((call) => Buffer.byteLength(call.messages[0].content[0].text, 'utf8') <= 4096));
    const repeated = await ctx.waterfall('agent/request-error', { agent, failure: { code: CONTEXT_WINDOW_EXCEEDED_CODE, message: 'Still overfull.' }, signal: AbortSignal.timeout(5000) }, () => undefined);
    assert.equal(repeated, undefined, 'overflow recovery is bounded per unsuccessful request');
  } finally {
    await ctx.fiber.dispose();
  }
});

test('compaction retains an entire latest tool turn when its result crosses the tail budget', async () => {
  const { ctx, engine, session, agent, user, header } = await setup();
  try {
    header();
    user('Old history. '.repeat(3000));
    session.append('step/start', { turn: 1, step: 1 });
    const assistant = createAssistantMessage({ content: [{ type: 'tool-call', id: 'file-1', name: 'pwsh', arguments: '{"command":"Get-Content main.cpp"}' }], source: agent.options });
    const callEvent = session.append('assistant/message', { turn: 1, step: 1, message: assistant, stream: [] }, { surfaceOp: 'append' });
    const resultEvent = session.append('tool/result', { turn: 1, step: 1, message: createToolResultMessage({ callId: 'file-1', content: [{ type: 'text', text: 'File contents. '.repeat(100) }], isError: false }) }, { surfaceOp: 'append' });
    session.append('step/end', { turn: 1, step: 1 });
    const result = await engine.compactIfNeeded(agent, 'pressure', AbortSignal.timeout(5000));
    assert.ok(result);
    assert.ok(session.surface.nodes.includes(callEvent.seq));
    assert.ok(session.surface.nodes.includes(resultEvent.seq));
    assert.ok(!result.shadowedSeqs.includes(callEvent.seq));
    assert.ok(!result.shadowedSeqs.includes(resultEvent.seq));
  } finally {
    await ctx.fiber.dispose();
  }
});

test('an irreducible single oversized user message does not enter an endless recovery loop', async () => {
  const { ctx, session, agent, calls, user, header } = await setup();
  try {
    header();
    user('One oversized incoming prompt. '.repeat(5000));
    const snapshot = session.snapshotEvents();
    const action = await ctx.waterfall('agent/request-error', { agent, failure: { code: CONTEXT_WINDOW_EXCEEDED_CODE, message: 'Prompt is too large.' }, signal: AbortSignal.timeout(5000) }, () => undefined);
    assert.equal(action, undefined);
    assert.equal(calls.length, 0);
    assert.deepEqual(session.snapshotEvents(), snapshot);
  } finally {
    await ctx.fiber.dispose();
  }
});

test('normal 45% pressure prunes a huge newest tool result before spending tokens on a summary', async (t) => {
  const { ctx, engine, session, agent, calls, user, header } = await setup();
  try {
    header([{ name: 'pwsh', description: 'Run a command.', parameters: { type: 'object', properties: { command: { type: 'string' } } } }]);
    new ToolResultPruner(ctx, { thresholdChars: 3072, headChars: 1536, tailChars: 512 });
    user('Inspect main.cpp and report the compiler failure.');
    const output = 'HEAD: main.cpp\n' + 'File contents. '.repeat(5000) + '\nTAIL: compilation failed at main.cpp:42';
    const { call, results } = appendToolTurn(session, agent, [output]);
    assert.ok(ctx.tokenMeter.measure(session).totalTokens > Math.floor(8192 * 0.45));
    const result = await engine.compactIfNeeded(agent, 'pressure', AbortSignal.timeout(5000));
    assert.equal(result, null, 'deterministic pruning alone should bring this turn below pressure');
    assert.equal(calls.length, 0);
    assert.ok(ctx.tokenMeter.measure(session).totalTokens < Math.floor(8192 * 0.45));
    assert.ok(session.surface.nodes.includes(call.seq));
    assert.ok(!session.surface.nodes.includes(results[0].seq));
    const pruned = session.deriveMessages().at(-1).content[0];
    assert.equal(pruned.toolCallId, 'file-0');
    assert.ok(pruned.content[0].text.startsWith('HEAD: main.cpp'));
    assert.ok(pruned.content[0].text.endsWith('TAIL: compilation failed at main.cpp:42'));
    assert.ok(pruned.content[0].text.includes('tool result middle pruned'));
    assert.equal(session.eventAt(results[0].seq).data.message.content[0].content[0].text, output, 'original tool output remains in the durable log');
    if (process.env.BONSAI_LIVE_TOKEN_COUNT === '1') {
      const { responseRequest } = await import('../bonsai2-heretic-3080/native-ninfer.mjs');
      const body = responseRequest({ ...agent.options, messages: session.deriveMessages(), tools: session.requestHeader().tools });
      const response = await fetch('http://127.0.0.1:18020/v1/responses/input_tokens', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body), signal: AbortSignal.timeout(5000) });
      const counted = await response.json();
      assert.equal(response.status, 200, JSON.stringify(counted));
      const { input_tokens } = counted;
      assert.ok(input_tokens + 4096 + 256 <= 8192, `pruned request needs ${input_tokens} input tokens`);
      t.diagnostic(`NInfer counted ${input_tokens} prompt tokens after deterministic pruning; 4096 output plus 256 reserve fit.`);
    }
  } finally {
    await ctx.fiber.dispose();
  }
});

test('context recovery summarizes a completed parallel tool turn while keeping the latest human request verbatim', async (t) => {
  const { ctx, session, agent, calls, user, header } = await setup();
  try {
    header();
    new ToolResultPruner(ctx, { thresholdChars: 3072, headChars: 1536, tailChars: 512 });
    const latestUser = user('Inspect main.cpp and report all compile errors.');
    const { call, results } = appendToolTurn(session, agent, Array.from({ length: 8 }, (_, index) => `FILE ${index}\n` + '🦙錯誤'.repeat(2000) + `\nTAIL ${index}`));
    const action = await ctx.waterfall('agent/request-error', { agent, failure: { code: CONTEXT_WINDOW_EXCEEDED_CODE, message: 'Latest parallel results fill the context.' }, signal: AbortSignal.timeout(5000) }, () => undefined);
    assert.deepEqual(action, { kind: 'retry' });
    assert.ok(session.surface.nodes.includes(latestUser.seq), 'the latest human request must remain verbatim');
    assert.ok(!session.surface.nodes.includes(call.seq), 'completed tool calls must be replaced together with their results');
    assert.ok(session.deriveMessages().every((message) => message.content.every((block) => block.type !== 'tool-call' && block.type !== 'tool-result')));
    assert.ok(calls.every((request) => Buffer.byteLength(request.messages[0].content[0].text, 'utf8') <= 4096));
    for (const result of results) assert.ok(session.eventAt(result.seq).data.message.content[0].content[0].text.includes('🦙錯誤'), 'full original results remain available in the durable log');
    if (process.env.BONSAI_LIVE_TOKEN_COUNT === '1') {
      const { responseRequest } = await import('../bonsai2-heretic-3080/native-ninfer.mjs');
      const body = responseRequest({ ...agent.options, messages: session.deriveMessages() });
      const response = await fetch('http://127.0.0.1:18020/v1/responses/input_tokens', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body), signal: AbortSignal.timeout(5000) });
      const counted = await response.json();
      assert.equal(response.status, 200, JSON.stringify(counted));
      assert.ok(counted.input_tokens + 4096 + 256 <= 8192);
      t.diagnostic(`NInfer counted ${counted.input_tokens} prompt tokens after parallel tool compaction.`);
      const largest = calls.reduce((selected, request) => Buffer.byteLength(request.messages[0].content[0].text, 'utf8') > Buffer.byteLength(selected.messages[0].content[0].text, 'utf8') ? request : selected);
      const summaryResponse = await fetch('http://127.0.0.1:18020/v1/responses/input_tokens', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(responseRequest(largest)), signal: AbortSignal.timeout(5000) });
      const summaryCount = await summaryResponse.json();
      assert.equal(summaryResponse.status, 200, JSON.stringify(summaryCount));
      assert.ok(summaryCount.input_tokens + 768 + 256 <= 8192);
      t.diagnostic(`Largest Unicode summary fragment counted ${summaryCount.input_tokens} prompt tokens.`);
    }
  } finally {
    await ctx.fiber.dispose();
  }
});

test('context overflow never summarizes a tool turn before its step has closed', async () => {
  const { ctx, session, agent, user, header } = await setup();
  try {
    header();
    const latestUser = user('Read main.cpp.');
    const { call, results } = appendToolTurn(session, agent, ['🦙錯誤'.repeat(3000)], false);
    const action = await ctx.waterfall('agent/request-error', { agent, failure: { code: CONTEXT_WINDOW_EXCEEDED_CODE, message: 'Step is still open.' }, signal: AbortSignal.timeout(5000) }, () => undefined);
    assert.equal(action, undefined);
    assert.ok(session.surface.nodes.includes(latestUser.seq));
    assert.ok(session.surface.nodes.includes(call.seq));
    assert.ok(session.surface.nodes.includes(results[0].seq));
    assert.equal(session.surface.replaceGeneration, 0);
  } finally {
    await ctx.fiber.dispose();
  }
});

test('context overflow never summarizes an unmatched tool call even if the step is marked closed', async () => {
  const { ctx, session, agent, user, header } = await setup();
  try {
    header();
    user('Read main.cpp.');
    session.append('step/start', { turn: 1, step: 1 });
    const message = createAssistantMessage({ content: [
      { type: 'tool-call', id: 'ready', name: 'pwsh', arguments: '{}' },
      { type: 'tool-call', id: 'pending', name: 'pwsh', arguments: '{}' },
    ], source: agent.options });
    const call = session.append('assistant/message', { turn: 1, step: 1, message, stream: [] }, { surfaceOp: 'append' });
    const result = session.append('tool/result', { turn: 1, step: 1, message: createToolResultMessage({ callId: 'ready', content: [{ type: 'text', text: '🦙錯誤'.repeat(3000) }], isError: false }) }, { surfaceOp: 'append' });
    session.append('step/end', { turn: 1, step: 1 });
    const action = await ctx.waterfall('agent/request-error', { agent, failure: { code: CONTEXT_WINDOW_EXCEEDED_CODE, message: 'Tool result is missing.' }, signal: AbortSignal.timeout(5000) }, () => undefined);
    assert.equal(action, undefined);
    assert.ok(session.surface.nodes.includes(call.seq));
    assert.ok(session.surface.nodes.includes(result.seq));
    assert.equal(session.surface.replaceGeneration, 0);
  } finally {
    await ctx.fiber.dispose();
  }
});

async function longPreviousTurn() {
  const fixture = await setup({ capacity: 65536, compactConfig: { thresholdRatio: 0.70, retainTokens: 2048, maxTokens: 1024 } });
  const { session, agent, user, header } = fixture;
  header();
  const original = user('Original user reference: ' + 'requirements and source context; '.repeat(6500));
  session.append('step/start', { turn: 1, step: 1 });
  session.append('assistant/message', {
    turn: 1, step: 1,
    message: createAssistantMessage({ content: [{ type: 'text', text: 'I have read the reference and can implement the requested change.' }], source: agent.options }),
    stream: [],
  }, { surfaceOp: 'append' });
  session.append('step/end', { turn: 1, step: 1 });
  session.append('turn/end', { turn: 1, reason: { kind: 'completed' } });
  session.append('turn/start', { turn: 2 });
  return { ...fixture, original };
}

test('a new accepted human follow-up allows proactive compaction of an old 50K-token user message', async () => {
  const { ctx, session, agent, calls, original } = await longPreviousTurn();
  try {
    assert.ok(ctx.tokenMeter.measure(session).totalTokens > 65536 * 0.70);
    const pending = createUserMessage({ content: [{ type: 'text', text: 'Implement the change now and preserve the exact identifier critical_function.' }], source: { kind: 'user' } });
    const expected = { kind: 'enter', messages: [pending] };
    const decision = await ctx.waterfall('agent/pre-step', { agent, messages: [pending], turn: 2, step: 2, signal: AbortSignal.timeout(5000) }, () => expected);
    assert.equal(decision, expected);
    assert.deepEqual(decision.messages, [pending]);
    assert.ok(session.surface.replaceGeneration > 0, 'the old oversized user node must no longer block pressure compaction');
    assert.ok(ctx.tokenMeter.measure(session).totalTokens < 65536 * 0.70);
    assert.ok(calls.length > 1);
    assert.equal(session.eventAt(original.seq), original, 'the full original user input stays in the durable log');
    assert.ok(!session.deriveMessages().some((message) => message.id === pending.id), 'only the loop may admit pending input');
    session.append('user/message', pending, { surfaceOp: 'append' });
    assert.deepEqual(session.deriveMessages().at(-1), pending);
  } finally {
    await ctx.fiber.dispose();
  }
});

test('old latest human input is not compacted without an accepted direct human follow-up', async () => {
  for (const scenario of ['empty', 'rejected', 'plugin-only', 'unclaimed']) {
    const { ctx, session, agent, calls, original } = await longPreviousTurn();
    try {
      const human = createUserMessage({ content: [{ type: 'text', text: 'Next request.' }], source: { kind: 'user' } });
      const plugin = createUserMessage({ content: [{ type: 'text', text: 'A tool or background job updated.' }], source: { kind: 'plugin', plugin: 'test' } });
      const claimed = scenario === 'rejected' ? [human] : scenario === 'plugin-only' ? [plugin] : [];
      const expected = scenario === 'rejected' ? { kind: 'reject' } : { kind: 'enter', messages: scenario === 'unclaimed' ? [human] : claimed };
      const decision = await ctx.waterfall('agent/pre-step', { agent, messages: claimed, turn: 2, step: 2, signal: AbortSignal.timeout(5000) }, () => expected);
      assert.equal(decision, expected);
      assert.equal(session.surface.replaceGeneration, 0, scenario);
      assert.ok(session.surface.nodes.includes(original.seq), scenario);
      assert.equal(calls.length, 0, scenario);
    } finally {
      await ctx.fiber.dispose();
    }
  }
});

test('a pending human follow-up never permits compaction of unfinished historical tool work', async () => {
  for (const unfinished of ['open-step', 'missing-result']) {
    const { ctx, session, agent, calls, user, header } = await setup({ capacity: 65536, compactConfig: { thresholdRatio: 0.70, retainTokens: 2048, maxTokens: 1024 } });
    try {
      header();
      const original = user('Old reference. '.repeat(15000));
      if (unfinished === 'open-step') {
        appendToolTurn(session, agent, ['Small completed result within an open step.'], false);
      } else {
        session.append('step/start', { turn: 1, step: 1 });
        session.append('assistant/message', {
          turn: 1, step: 1,
          message: createAssistantMessage({ content: [
            { type: 'tool-call', id: 'first', name: 'pwsh', arguments: '{}' },
            { type: 'tool-call', id: 'unanswered', name: 'pwsh', arguments: '{}' },
          ], source: agent.options }),
          stream: [],
        }, { surfaceOp: 'append' });
        session.append('tool/result', { turn: 1, step: 1, message: createToolResultMessage({ callId: 'first', content: [{ type: 'text', text: 'Only one result has arrived.' }], isError: false }) }, { surfaceOp: 'append' });
        session.append('step/end', { turn: 1, step: 1 });
      }
      const pending = createUserMessage({ content: [{ type: 'text', text: 'Continue implementing the change.' }], source: { kind: 'user' } });
      const expected = { kind: 'enter', messages: [pending] };
      const decision = await ctx.waterfall('agent/pre-step', { agent, messages: [pending], turn: 2, step: 2, signal: AbortSignal.timeout(5000) }, () => expected);
      assert.equal(decision, expected);
      assert.equal(calls.length, 0, unfinished);
      assert.equal(session.surface.replaceGeneration, 0, unfinished);
      assert.ok(session.surface.nodes.includes(original.seq), unfinished);
    } finally {
      await ctx.fiber.dispose();
    }
  }
});

test('a fake tool-call checkpoint never replaces the live session history', async () => {
  const { ctx, engine, session, agent, calls, user, header } = await setup({ summaryResponse: '[tool call call-example: pwsh]\n{"command":"Get-Content source.cpp"}' });
  try {
    header();
    const first = user('Exact original requirement: preserve all flags. ' + 'Original context. '.repeat(1000));
    const last = user('Continue the requested repair.');
    const originalEvents = session.snapshotEvents();
    await assert.rejects(engine.compactRegion(first.seq, last.seq, agent, AbortSignal.timeout(5000)), /tool invocation instead of a checkpoint/);
    assert.equal(calls.length, 2);
    assert.equal(session.surface.replaceGeneration, 0);
    assert.ok(session.surface.nodes.includes(first.seq));
    assert.ok(session.surface.nodes.includes(last.seq));
    assert.deepEqual(session.snapshotEvents(0, originalEvents.length), originalEvents);
    assert.equal(session.snapshotEvents().filter((event) => event.type === 'compaction/summary').length, 0);
  } finally {
    await ctx.fiber.dispose();
  }
});
