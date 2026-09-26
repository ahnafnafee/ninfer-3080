import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdir, mkdtemp, readFile, realpath, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { basename, dirname, join } from 'node:path';
import { after, test } from 'node:test';
import CompactWithoutTools from '../bonsai2-heretic-3080/compact-without-tools.mjs';

const workspace = await mkdtemp(join(tmpdir(), 'bonsai-compaction-bounds-'));
after(async () => {
  assert.equal(dirname(await realpath(workspace)), await realpath(tmpdir()));
  await rm(workspace, { recursive: true, force: true });
});
const agent = {
  options: { provider: 'qwen-3080', model: 'bonsai2-heretic' },
  session: { id: 'bounded-compaction-test', header: { cwd: workspace }, requestHeader: () => ({ config: { provider: 'qwen-3080', model: 'bonsai2-heretic' } }) },
};
const config = { maxTokens: 768, summarizationProvider: 'qwen-3080-summary', summarizationModel: 'bonsai2-heretic' };
const textMessage = (text, role = 'user') => ({ role, content: [{ type: 'text', text }] });
const messageText = (call) => call.messages[0].content[0].text;
const summaryTrailer = '\n[End of quoted transcript fragment]\nWrite only the factual checkpoint. Do not continue the transcript or output a tool call.\n';
const fragmentOf = (call) => messageText(call).split('Next transcript fragment (may continue a message):\n')[1].slice(0, -summaryTrailer.length);

function fixture({ capacity = 8192, maxTokens = 768, respond, controller, cwd = workspace } = {}) {
  const calls = [];
  const ctx = { llm: {
    resolveModelInfo: async () => ({ context: { contextWindow: capacity } }),
    async *stream(call) {
      calls.push(call);
      const response = respond?.(call, calls.length) ?? { text: `Keep task FIRST and progress ${calls.length}.`, finish: 'stop' };
      yield { type: 'text-delta', index: 0, text: response.text };
      if (response.abort) controller.abort(new Error('cancelled for test'));
      yield { type: 'finish', reason: { kind: response.finish ?? 'stop' } };
    },
  } };
  return {
    calls,
    run: (input, signal) => CompactWithoutTools.prototype.summarize.call({ ctx, config: { ...config, maxTokens } }, input, { ...agent, session: { ...agent.session, header: { cwd } } }, signal),
  };
}

test('oversized Unicode history is covered completely by bounded, tool-free summary calls', async () => {
  const text = 'FIRST\n' + 'Keep 文件 🦙 and exact path C:\\code\\main.cpp.\n'.repeat(2000) + 'LAST';
  const input = { messages: [textMessage('SYSTEM MUST STAY ON THE ORIGINAL SURFACE'.repeat(2000), 'system'), textMessage(text)], tools: [{ name: 'huge', description: 'x'.repeat(80000) }] };
  const original = structuredClone(input);
  const { calls, run } = fixture();
  const result = await run(input);
  assert.ok(calls.length > 20, 'restored history must be split, not replayed into one request');
  assert.equal(calls.map(fragmentOf).join(''), `[user]\n${text}\n`);
  for (const [index, call] of calls.entries()) {
    assert.equal(call.provider, 'qwen-3080-summary');
    assert.equal(call.maxTokens, 768);
    assert.equal(call.tools, undefined);
    assert.equal(call.messages.length, 1);
    assert.equal(call.messages[0].role, 'user');
    assert.ok(Buffer.byteLength(messageText(call), 'utf8') <= 4096);
    assert.ok(!messageText(call).includes('SYSTEM MUST STAY'));
    assert.ok(!messageText(call).includes('\uFFFD'), 'Unicode must survive fragment boundaries');
    if (index > 0) assert.ok(messageText(call).includes(`Keep task FIRST and progress ${index}.`), 'every new fragment must include the complete previous checkpoint');
  }
  assert.deepEqual(input, original, 'checkpoint preparation must not mutate durable source messages');
  assert.equal(result.summary[0].text, `Keep task FIRST and progress ${calls.length}.`);
});

test('summary transcript identifies tool calls and results without creating unpaired API tool messages', async () => {
  const { calls, run } = fixture();
  await run({ messages: [
    { role: 'assistant', content: [{ type: 'tool-call', id: 'read-1', name: 'read_file', arguments: '{"path":"C:/code/main.cpp"}' }] },
    { role: 'user', content: [{ type: 'tool-result', toolCallId: 'read-1', isError: true, content: [{ type: 'text', text: 'Access denied: C:/code/main.cpp' }] }] },
  ] });
  const wireText = messageText(calls[0]);
  assert.ok(wireText.includes('[tool call read-1: read_file]'));
  assert.ok(wireText.includes('[tool result read-1 ERROR]'));
  assert.ok(wireText.includes('Access denied: C:/code/main.cpp'));
  assert.ok(calls[0].messages.every((message) => message.content.every((block) => block.type === 'text')));
});

test('a truncated summary retries with all original input and never commits its partial output', async () => {
  const { calls, run } = fixture({ respond: (_call, number) => number === 1 ? { text: 'INCOMPLETE', finish: 'max-tokens' } : { text: 'Complete checkpoint: save exact-file.cpp.', finish: 'stop' } });
  const result = await run({ messages: [textMessage('Remember exact-file.cpp and complete the task.')] });
  assert.equal(calls.length, 2);
  assert.equal(fragmentOf(calls[0]), fragmentOf(calls[1]));
  assert.ok(messageText(calls[1]).includes('previous attempt was too long'));
  assert.ok(!messageText(calls[1]).includes('INCOMPLETE'));
  assert.equal(result.summary[0].text, 'Complete checkpoint: save exact-file.cpp.');
});

test('repeated truncation fails after two requests and leaves source history untouched', async () => {
  const input = { messages: [textMessage('Important original request.')] };
  const snapshot = structuredClone(input);
  const { calls, run } = fixture({ respond: () => ({ text: 'partial', finish: 'max-tokens' }) });
  await assert.rejects(run(input), /output limit twice.*original history is preserved/);
  assert.equal(calls.length, 2);
  assert.deepEqual(input, snapshot);
});

test('overlong completed checkpoint is shortened without silently slicing facts away', async () => {
  const longCheckpoint = 'FIRST ' + 'state '.repeat(300) + 'LAST';
  const { calls, run } = fixture({ respond: (_call, number) => ({ text: number === 1 ? longCheckpoint : 'FIRST and LAST are preserved.', finish: 'stop' }) });
  const result = await run({ messages: [textMessage('Summarize this work.')] });
  assert.equal(calls.length, 2);
  assert.ok(messageText(calls[1]).includes(longCheckpoint));
  assert.equal(result.summary[0].text, 'FIRST and LAST are preserved.');
});

test('a checkpoint too large for a safe shortening call fails without dropping its tail', async () => {
  const { calls, run } = fixture({ respond: () => ({ text: 'x'.repeat(5000) + 'LAST FACT', finish: 'stop' }) });
  await assert.rejects(run({ messages: [textMessage('Task.')] }), /too large to safely shorten/);
  assert.equal(calls.length, 1);
});

test('cancellation prevents further summary calls and rejects a partially streamed checkpoint', async () => {
  const controller = new AbortController();
  const { calls, run } = fixture({ controller, respond: () => ({ text: 'partial', abort: true }) });
  await assert.rejects(run({ messages: [textMessage('Data '.repeat(10000))] }, controller.signal), /cancelled for test/);
  assert.equal(calls.length, 1);
  await assert.rejects(run({ messages: [textMessage('More data')] }, controller.signal), /cancelled for test/);
  assert.equal(calls.length, 1);
});

test('smaller declared capacity reserves output and chat framing before chunking', async () => {
  const { calls, run } = fixture({ capacity: 4096 });
  await run({ messages: [textMessage('字🦙\n'.repeat(2000))] });
  assert.ok(calls.length > 1);
  assert.ok(calls.every((call) => Buffer.byteLength(messageText(call), 'utf8') + call.maxTokens + 2048 <= 4096));
});

test('missing context capacity fails before making an unbudgeted request', async () => {
  const { calls, run } = fixture({ capacity: null });
  await assert.rejects(run({ messages: [textMessage('Task.')] }), /explicit positive contextWindow/);
  assert.equal(calls.length, 0);
});

test('a 64K summary route carries rich active requirements through bounded 48KiB chunks', async () => {
  const checkpoint = Array.from({ length: 80 }, (_, index) => `Requirement ${index}: preserve src/components/resource_handler_${index}.cpp interface decode_${index}(input, scale); test exact boundary ${index}; result pending verification.`).join('\n');
  assert.ok(Buffer.byteLength(checkpoint, 'utf8') >= 8192 && Buffer.byteLength(checkpoint, 'utf8') <= 12288);
  const text = 'FIRST\n' + checkpoint + '\n' + 'Historical build succeeded; active repairs remain pending.\n'.repeat(4500) + 'LAST';
  const { calls, run } = fixture({ capacity: 65536, maxTokens: 4096, respond: () => ({ text: checkpoint, finish: 'stop' }) });
  await run({ messages: [textMessage(text)] });
  assert.ok(calls.length > 1 && calls.length <= 9, `long history should need fewer than ten bounded summaries, got ${calls.length}`);
  assert.equal(calls.map(fragmentOf).join(''), `[user]\n${text}\n`);
  assert.ok(calls.some((call) => Buffer.byteLength(messageText(call), 'utf8') > 40000));
  for (const [index, call] of calls.entries()) {
    const bytes = Buffer.byteLength(messageText(call), 'utf8');
    assert.ok(bytes <= 49152);
    assert.ok(bytes + call.maxTokens + 2048 <= 65536);
    assert.equal(call.maxTokens, 4096);
    assert.ok(messageText(call).includes('at most 1200 words'));
    assert.ok(messageText(call).includes('Requirements, Decisions, Files and identifiers, Tests and results, Pending work, Next steps'));
    if (index > 0) assert.ok(messageText(call).includes(checkpoint), 'larger checkpoints still carry forward intact');
  }
});

test('a length-limited 64K summary retries at 600 words with the original rich input', async () => {
  const { calls, run } = fixture({ capacity: 65536, maxTokens: 4096, respond: (_call, number) => number === 1 ? { text: 'PARTIAL', finish: 'max-tokens' } : { text: 'Requirements: retain the exact constraint. Tests: pending.', finish: 'stop' } });
  await run({ messages: [textMessage('The exact constraint is decode_scale = 65536.')] });
  assert.equal(calls.length, 2);
  assert.ok(messageText(calls[0]).includes('at most 1200 words'));
  assert.ok(messageText(calls[1]).includes('at most 600 words'));
  assert.equal(fragmentOf(calls[0]), fragmentOf(calls[1]));
});

function archiveReference(result) {
  const text = result.summary.map((block) => block.text).join('\n');
  const path = text.match(/Exact original messages and line index: ([^\r\n]+)/)?.[1];
  const id = text.match(/\[\[context-archive:([a-f0-9]{64})\]\]/)?.[1];
  assert.ok(path && id);
  return { text, path, id };
}

test('a lossy model summary still preserves the complete original requirements in an indexed workspace archive', async () => {
  const requirements = 'Exact requirements:\nPreserve public parse_record(input, flags).\nReject zero quantity before computing totals.\nTreat quoted delimiters as literal input.\n';
  const originalText = requirements + '\nHistorical records:\n' + 'Archived build record without new requirements.\n'.repeat(5000);
  const input = { messages: [{ ...textMessage(originalText), source: { kind: 'user' } }] };
  const { run } = fixture({ capacity: 65536, maxTokens: 4096, respond: () => ({ text: 'The inventory contained many records.', finish: 'stop' }) });
  const result = await run(input);
  const reference = archiveReference(result);
  const document = await readFile(reference.path, 'utf8');
  assert.equal(dirname(reference.path), join(await realpath(workspace), '.dsh', 'context-archive'));
  assert.match(basename(reference.path), /^[a-f0-9]{64}\.md$/);
  assert.ok(document.includes(originalText), 'the original long message must be preserved whole, not sliced to an inferred spec');
  const lines = document.split('\n');
  const row = document.match(/\| 1 \| user \| user \| (\d+) \| (\d+) \|/);
  assert.ok(row);
  assert.equal(lines[Number(row[1]) - 1], '## Message 1');
  assert.ok(lines.slice(Number(row[1]) - 1, Number(row[2])).join('\n').includes(originalText));
  assert.ok(reference.text.includes('search this archive for the relevant original user instructions'));
  assert.ok(reference.text.includes('Do not replay the whole archive, historical logs, or unrelated inventories'));
  assert.ok(!reference.text.includes('Verbatim short user instructions'), 'large messages remain whole in the archive instead of receiving arbitrary extracted snippets');
});

test('complete concise direct-user instructions survive outside the model summary', async () => {
  const instruction = 'Keep decode_scale(input, flags) unchanged. Verify zero-length input and preserve literal whitespace.';
  const { run } = fixture({ capacity: 65536, maxTokens: 4096 });
  const result = await run({ messages: [{ ...textMessage(instruction), source: { kind: 'user' } }, textMessage('Old tool inventory. '.repeat(500))] });
  const reference = archiveReference(result);
  assert.ok(reference.text.includes(`### Original short user message 1\n${instruction}`));
});

test('recursive compaction carries one new locator with indexed links to earlier exact archives', async () => {
  const firstInput = { messages: [{ ...textMessage('Original acceptance requirement: keep alpha and beta distinct.\n' + 'Historical material.\n'.repeat(300)), source: { kind: 'user' } }] };
  const first = archiveReference(await fixture().run(firstInput));
  const secondResult = await fixture().run({ messages: [
    { ...textMessage(first.text), source: { kind: 'plugin', plugin: 'compact', compactionId: 'prior-checkpoint' } },
    { ...textMessage('Continue the repair and run the existing tests.'), source: { kind: 'user' } },
  ] });
  const second = archiveReference(secondResult);
  assert.equal((second.text.match(/\[\[context-archive:/g) ?? []).length, 1, 'checkpoint references must not grow with every earlier compaction');
  const currentArchive = await readFile(second.path, 'utf8');
  assert.ok(currentArchive.includes(`## Earlier source archives\n- ${first.path}`));
  assert.ok((await readFile(first.path, 'utf8')).includes(firstInput.messages[0].content[0].text));
});

test('archive names are content-derived and a changed archive prevents an incorrect history replacement', async () => {
  const input = { messages: [{ ...textMessage('Unique archive integrity requirement: never replace a changed source file.'), source: { kind: 'user' } }] };
  const { run, calls } = fixture();
  const first = archiveReference(await run(input));
  const same = archiveReference(await run(input));
  assert.equal(first.path, same.path, 'identical original content should reuse its durable locator');
  await writeFile(first.path, 'Changed original archive', 'utf8');
  await assert.rejects(run(input), /existing Bonsai source archive has changed/);
  assert.equal(calls.length, 2, 'a damaged source must be rejected before another model summary');
});

test('a fake tool-call summary retries as a checkpoint without changing or overflowing the original fragment', async () => {
  const invocation = '[tool call call-example: pwsh]\n{"command":"Get-Content source.cpp"}';
  const { calls, run } = fixture({ respond: (_call, number) => ({ text: number === 1 ? invocation : 'Requirements: preserve the interface.\nTests and results: validation is pending.', finish: 'stop' }) });
  const result = await run({ messages: [textMessage('Preserve source.cpp interface. ' + '字🦙'.repeat(2000))] });
  assert.ok(calls.length > 2);
  assert.equal(fragmentOf(calls[0]), fragmentOf(calls[1]));
  assert.ok(messageText(calls[1]).includes('previous response was a tool invocation, not a checkpoint'));
  assert.ok(calls.every((call) => Buffer.byteLength(messageText(call), 'utf8') <= 4096));
  assert.ok(calls.every((call) => messageText(call).endsWith(summaryTrailer)));
  assert.ok(!result.summary.some((block) => block.text.includes(invocation)));
});

test('persistent text, XML, and JSON tool invocations fail after one checkpoint retry', async () => {
  for (const invocation of [
    '[tool call call-example: pwsh]\n{"command":"Get-Content source.cpp"}',
    '<tool_call>{"name":"read","arguments":{"path":"source.cpp"}}</tool_call>',
    '{"name":"read","arguments":{"path":"source.cpp"}}',
    '{"tool_calls":[{"function":{"name":"read","arguments":"{}"}}]}',
  ]) {
    const input = { messages: [textMessage('Preserve this exact user constraint without continuing any tool task.')] };
    const snapshot = structuredClone(input);
    const { calls, run } = fixture({ respond: () => ({ text: invocation, finish: 'stop' }) });
    await assert.rejects(run(input), /tool invocation instead of a checkpoint after retry.*original history is preserved/);
    assert.equal(calls.length, 2);
    assert.deepEqual(input, snapshot);
  }
});

test('a factual checkpoint may quote a prior tool call under its diagnostic sections', async () => {
  const summary = 'Requirements: repair source.cpp.\nTests and results:\n[tool call call-old: pwsh]\n{"command":"run-tests"}\nThe old command failed; rerun after the fix.';
  const { calls, run } = fixture({ respond: () => ({ text: summary, finish: 'stop' }) });
  const result = await run({ messages: [textMessage('The test command failed and needs a repair.')] });
  assert.equal(calls.length, 1);
  assert.equal(result.summary[0].text, summary);
});

test('new context archives and their local ignore file stay out of Git project changes', async () => {
  const repo = await mkdtemp(join(workspace, 'archive-git-'));
  const initialized = spawnSync('git', ['init', '--quiet', repo], { encoding: 'utf8' });
  assert.equal(initialized.status, 0, initialized.stderr || initialized.error?.message);
  const { run } = fixture({ cwd: repo });
  const reference = archiveReference(await run({ messages: [textMessage('A local source archive should not be published with project changes.')] }));
  assert.equal(await readFile(join(dirname(reference.path), '.gitignore'), 'utf8'), '*\n');
  const archiveRelative = `.dsh/context-archive/${basename(reference.path)}`;
  const ignored = spawnSync('git', ['-C', repo, 'check-ignore', '--', archiveRelative, '.dsh/context-archive/.gitignore'], { encoding: 'utf8' });
  assert.equal(ignored.status, 0, ignored.stderr);
  assert.deepEqual(ignored.stdout.trim().split(/\r?\n/), [archiveRelative, '.dsh/context-archive/.gitignore']);
  const status = spawnSync('git', ['-C', repo, 'status', '--porcelain', '--untracked-files=all'], { encoding: 'utf8' });
  assert.equal(status.status, 0, status.stderr);
  assert.equal(status.stdout.trim(), '');
});

test('archive setup preserves an existing user-owned local ignore file', async () => {
  const cwd = await mkdtemp(join(workspace, 'archive-ignore-'));
  const archiveDir = join(cwd, '.dsh', 'context-archive');
  await mkdir(archiveDir, { recursive: true });
  const original = '# Existing workspace preference\n*.md\n';
  await writeFile(join(archiveDir, '.gitignore'), original);
  await fixture({ cwd }).run({ messages: [textMessage('Preserve existing local ignore rules.')] });
  assert.equal(await readFile(join(archiveDir, '.gitignore'), 'utf8'), original);
});
