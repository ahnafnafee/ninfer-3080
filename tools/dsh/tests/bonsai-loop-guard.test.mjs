import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { pathToFileURL } from 'node:url';
const requireDsh = createRequire(join(dirname(process.execPath), 'node_modules/@deepseek-ai/dsh/package.json'));
const { Context } = await import(pathToFileURL(requireDsh.resolve('@deepseek-ai/cordis')));
const { default: Tools, defineContentToolFixture } = await import(pathToFileURL(requireDsh.resolve('@deepseek-ai/dsh-tools')));
const guard = await import('../bonsai2-heretic-3080/loop-guard.mjs');

async function setup() {
  const ctx = new Context();
  ctx.provide('systemPrompt', { tools() {} });
  await ctx.plugin(Tools);
  await ctx.plugin(guard);
  let executions = 0, output = "[stderr]\nThe term 'numFiles' is not recognized";
  for (const name of ['pwsh', 'job_output']) ctx.tools.register(defineContentToolFixture({
    name, description: 'Fixture tool', parameters: { command: { type: 'string' } },
    async execute() { executions++; return [{ type: 'text', text: output }]; },
  }));
  const agent = { options: { provider: 'qwen-3080', model: 'bonsai2-heretic' }, session: { snapshotEvents: () => [] } };
  let id = 0;
  return { ctx, agent, executions: () => executions, output: value => { output = value; },
    call: (command = 'numFiles = 1', name = 'pwsh') => ctx.tools.execute({ name, arguments: { command }, callId: `call_${++id}`, agent, signal: AbortSignal.timeout(5000) }),
    step: messages => ctx.waterfall('agent/pre-step', { agent, messages }, () => ({ kind: 'enter', messages })),
  };
}

test('unchanged successful-looking PowerShell failures stop before a fourth execution', async () => {
  const f = await setup();
  try {
    for (let i = 0; i < 3; i++) assert.equal((await f.call()).isError, false);
    const blocked = await f.call();
    assert.equal(f.executions(), 3, 'the fourth identical command must not run');
    assert.equal(blocked.isError, true);
    assert.match(blocked.content[0].text, /unchanged|progress/i);
    const recovery = await f.step([{ source: { kind: 'plugin', plugin: 'compaction' } }]);
    assert.equal(recovery.kind, 'enter');
    assert.ok(recovery.messages.some(guard.isRecoveryNotice));
    assert.equal((await f.call()).isError, true, 'the recovery attempt cannot execute the same command');
    assert.equal(f.executions(), 3);
    assert.equal((await f.step([{ source: { kind: 'plugin', plugin: 'compaction' } }])).kind, 'reject', 'compaction/reminders cannot grant another recovery');
    assert.equal((await f.step([{ source: { kind: 'user' } }])).kind, 'enter');
    assert.equal((await f.call()).isError, false, 'a new user turn explicitly permits another attempt');
  } finally { await f.ctx.fiber.dispose(); }
});

test('changing output and background-job polling remain usable', async () => {
  const f = await setup();
  try {
    for (let i = 0; i < 6; i++) { f.output(`progress ${i}`); assert.equal((await f.call()).isError, false); }
    for (let i = 0; i < 6; i++) assert.equal((await f.call('poll', 'job_output')).isError, false);
    assert.equal(f.executions(), 12);
  } finally { await f.ctx.fiber.dispose(); }
});

test('alternating unchanged calls are bounded and an independent agent keeps its own count', async () => {
  const f = await setup();
  try {
    for (let i = 0; i < 3; i++) { await f.call('a'); await f.call('b'); }
    assert.equal((await f.call('a')).isError, true);
    const other = { ...f.agent };
    const result = await f.ctx.tools.execute({ name: 'pwsh', arguments: { command: 'a' }, callId: 'other', agent: other, signal: AbortSignal.timeout(5000) });
    assert.equal(result.isError, false);
  } finally { await f.ctx.fiber.dispose(); }
});

test('a resumed agent recovers repetition evidence from tool pairs across compaction', async () => {
  const f = await setup();
  try {
    const history = [];
    for (let i = 0; i < 3; i++) history.push(
      { type: 'tool/call', data: { name: 'pwsh', callId: `old-${i}`, arguments: '{"command":"numFiles = 1"}' } },
      { type: 'tool/result', data: { message: { source: { callId: `old-${i}` }, content: [{ type: 'tool-result', isError: false, content: [{ type: 'text', text: 'same output' }] }] } } },
      { type: 'user/message', data: { source: { kind: 'plugin', plugin: 'compaction' } } },
    );
    f.agent.session.snapshotEvents = () => history;
    assert.equal((await f.call()).isError, true);
    assert.equal(f.executions(), 0);
  } finally { await f.ctx.fiber.dispose(); }
});

test('the actual DSH agent loop bounds a failed recovery to five completions', { timeout: 10000 }, async () => {
  const moduleOf = name => import(pathToFileURL(requireDsh.resolve(`@deepseek-ai/${name}`)));
  const { default: Llm, LlmAdapter, createUserMessage } = await moduleOf('dsh-llm');
  const ctx = new Context();
  let requests = 0, executions = 0;
  class RepeatingAdapter extends LlmAdapter {
    providerInfo() { return { id: 'qwen-3080', name: 'fixture' }; }
    async resolveModel(provider, id) { return { provider, id, name: 'fixture', inputModalities: ['text'], context: { contextWindow: 65536 }, defaultMaxTokens: 8192 }; }
    async *stream() {
      assert.ok(++requests <= 5, 'one recovery must not become an unlimited stream of denied calls');
      yield { type: 'block-end', index: 0, block: { type: 'tool-call', id: `call_${requests}`, name: 'pwsh', arguments: '{"command":"numFiles = 1"}' } };
      yield { type: 'finish', reason: { kind: 'tool-calls' } };
    }
  }
  try {
    for (const name of ['dsh-agent', 'dsh-session', 'dsh-session-projection', 'dsh-system-prompt']) await ctx.plugin((await moduleOf(name)).default);
    await ctx.plugin(Llm);
    ctx.llm.registerAdapter(['qwen-3080'], new RepeatingAdapter());
    await ctx.plugin(Tools);
    ctx.tools.register(defineContentToolFixture({ name: 'pwsh', description: 'fixture', parameters: { command: { type: 'string' } },
      async execute() { executions++; return [{ type: 'text', text: "[stderr] The term 'numFiles' is not recognized" }]; } }));
    await ctx.plugin(guard);
    await ctx.plugin((await moduleOf('dsh-agent-loop')).default);
    const agent = await ctx.agentLoop.create('loop-regression', { provider: 'qwen-3080', model: 'bonsai2-heretic' });
    agent.followup(createUserMessage({ content: [{ type: 'text', text: 'Inspect the fixture.' }], source: { kind: 'user' } }));
    await agent.whenIdle();
    assert.equal(executions, 3, JSON.stringify(agent.session.snapshotEvents().filter(event => event.type === 'turn/end')));
    assert.equal(requests, 5);
    const events = agent.session.snapshotEvents();
    assert.equal(events.findLast(event => event.type === 'turn/end').data.reason.kind, 'blocked');
    assert.equal(events.filter(event => event.type === 'tool/result').length, 5, 'each denial closes its tool-call pair');
  } finally { await ctx.fiber.dispose(); }
});

test('resume preserves the consumed recovery and unchanged-result count after a denial', async () => {
  const f = await setup();
  try {
    const history = [];
    for (let i = 0; i < 4; i++) history.push(
      { type: 'tool/call', data: { name: 'pwsh', callId: `old-${i}`, arguments: '{"command":"numFiles = 1"}' } },
      { type: 'tool/result', data: { message: { source: { callId: `old-${i}` }, content: [{ type: 'tool-result', isError: i === 3, content: [{ type: 'text', text: i === 3 ? 'Error: Blocked a repeated-tool call: duplicate' : 'same output' }] }] } } },
    );
    history.push({ type: 'user/message', data: guard.recoveryNotice() });
    history.push({ type: 'user/message', data: { source: { kind: 'plugin', plugin: 'compaction' } } });
    f.agent.session.snapshotEvents = () => history;
    assert.equal((await f.call()).isError, true);
    assert.equal(f.executions(), 0);
    assert.equal((await f.step([])).kind, 'reject', 'resume cannot grant a second recovery');
  } finally { await f.ctx.fiber.dispose(); }
});

test('the actual DSH loop gets one recovery step and can verify a changed command', { timeout: 10000 }, async () => {
  const moduleOf = name => import(pathToFileURL(requireDsh.resolve(`@deepseek-ai/${name}`)));
  const { default: Llm, LlmAdapter, createUserMessage } = await moduleOf('dsh-llm');
  const ctx = new Context();
  const commands = [];
  let requests = 0;
  class RecoveringAdapter extends LlmAdapter {
    providerInfo() { return { id: 'qwen-3080', name: 'fixture' }; }
    async resolveModel(provider, id) { return { provider, id, name: 'fixture', inputModalities: ['text'], context: { contextWindow: 65536 }, defaultMaxTokens: 8192 }; }
    async *stream(options) {
      assert.ok(++requests <= 6);
      if (requests === 5) {
        assert.ok(options.messages.some(message => message.source.plugin === guard.name && /recovery/i.test(message.source.summary ?? '')));
      }
      if (requests < 6) {
        yield { type: 'block-end', index: 0, block: { type: 'tool-call', id: `call_${requests}`, name: 'pwsh', arguments: JSON.stringify({ command: requests <= 4 ? 'read unchanged bytes' : 'measure encoded byte length' }) } };
        yield { type: 'finish', reason: { kind: 'tool-calls' } };
      } else {
        yield { type: 'block-end', index: 0, block: { type: 'text', text: 'The byte-length check resolved the discrepancy.' } };
        yield { type: 'finish', reason: { kind: 'stop' } };
      }
    }
  }
  try {
    for (const name of ['dsh-agent', 'dsh-session', 'dsh-session-projection', 'dsh-system-prompt']) await ctx.plugin((await moduleOf(name)).default);
    await ctx.plugin(Llm);
    ctx.llm.registerAdapter(['qwen-3080'], new RecoveringAdapter());
    await ctx.plugin(Tools);
    ctx.tools.register(defineContentToolFixture({ name: 'pwsh', description: 'fixture', parameters: { command: { type: 'string' } },
      async execute(args) { commands.push(args.command); return [{ type: 'text', text: args.command === 'read unchanged bytes' ? '0d 0a 7d 00' : '9848 encoded bytes; 9846 characters' }]; } }));
    await ctx.plugin(guard);
    await ctx.plugin((await moduleOf('dsh-agent-loop')).default);
    const agent = await ctx.agentLoop.create('recovery-regression', { provider: 'qwen-3080', model: 'bonsai2-heretic' });
    agent.followup(createUserMessage({ content: [{ type: 'text', text: 'Resolve the byte-offset discrepancy.' }], source: { kind: 'user' } }));
    await agent.whenIdle();
    assert.deepEqual(commands, ['read unchanged bytes', 'read unchanged bytes', 'read unchanged bytes', 'measure encoded byte length']);
    assert.equal(requests, 6);
    assert.equal(agent.session.snapshotEvents().findLast(event => event.type === 'turn/end').data.reason.kind, 'completed');
  } finally { await ctx.fiber.dispose(); }
});
