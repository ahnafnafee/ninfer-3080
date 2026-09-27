import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { pathToFileURL } from 'node:url';
const requireDsh = createRequire(join(dirname(process.execPath), 'node_modules/@deepseek-ai/dsh/package.json'));
const { createUserMessage } = await import(pathToFileURL(requireDsh.resolve('@deepseek-ai/dsh-llm')));

const polling = new Set(['job_output', 'job_list', 'list_agents', 'get_goal']);
const canonical = value => JSON.stringify(value, (_key, item) => item && typeof item === 'object' && !Array.isArray(item)
  ? Object.fromEntries(Object.keys(item).sort().map(key => [key, item[key]])) : item);
const digest = value => createHash('sha256').update(canonical(value)).digest('hex');
function callKey(name, args) {
  if (typeof args === 'string') { try { args = JSON.parse(args); } catch {} }
  return digest([name, args]);
}
function observe(state, key, result) {
  const signature = digest([Boolean(result.isError), result.content]);
  const previous = state.calls.get(key);
  const count = previous?.signature === signature ? previous.count + 1 : 1;
  state.calls.delete(key);
  state.calls.set(key, { signature, count });
  if (state.calls.size > 16) state.calls.delete(state.calls.keys().next().value);
  return count;
}

export const name = 'bonsai-loop-guard';
const recoverySummary = 'Loop recovery';
const blockedPrefix = 'Blocked a repeated-tool call:';
const stoppedPrefix = 'Stopped a repeated-tool loop:';
export const isRecoveryNotice = message => message.source?.kind === 'plugin' && message.source.plugin === name && message.source.summary === recoverySummary;
export function recoveryNotice() {
  return createUserMessage({
    source: { kind: 'plugin', plugin: name, form: 'notice', summary: recoverySummary },
    content: [{ type: 'text', text: 'One loop-recovery attempt is available. The duplicate call was not executed. Earlier assistant reasoning is omitted from subsequent model prompts; the original history and all tool evidence remain intact. Reassess the last result and the current user requirement. Test a different falsifiable explanation with a different operation, or state the unresolved blocker. Do not repeat the blocked command, rename it, or merely change its description. Do not claim completion without verification.' }],
  });
}
export function apply(ctx) {
  const states = new WeakMap();
  const fresh = () => ({ calls: new Map(), halted: false, recoveryUsed: false, pendingRecovery: false });
  const stateFor = agent => {
    let state = states.get(agent);
    if (state) return state;
    state = fresh();
    // Resume/compaction must not erase evidence of a loop. Read only the recent
    // durable tail once; a newly claimed human message explicitly resets it.
    const pending = new Map();
    for (const event of (agent.session?.snapshotEvents() ?? []).slice(-256)) {
      if (event.type === 'user/message' && event.data.source?.kind === 'user') { state = fresh(); pending.clear(); }
      if (event.type === 'user/message' && isRecoveryNotice(event.data)) { state.recoveryUsed = true; state.pendingRecovery = false; }
      if (event.type === 'tool/call' && !polling.has(event.data.name)) pending.set(event.data.callId, callKey(event.data.name, event.data.arguments));
      if (event.type === 'tool/result') {
        const id = event.data.message.source.callId;
        const key = pending.get(id);
        if (key) {
          const result = event.data.message.content.find(block => block.type === 'tool-result');
          const text = result?.content.filter(block => block.type === 'text').map(block => block.text).join('\n') ?? '';
          // A denied call did not execute: it must not replace the repeated
          // result's signature and grant another three attempts after resume.
          if (result?.isError && text.startsWith(`Error: ${blockedPrefix}`)) {
            state.recoveryUsed = true;
            state.pendingRecovery = true;
          } else if (result?.isError && text.startsWith(`Error: ${stoppedPrefix}`)) state.halted = true;
          else if (result) observe(state, key, result);
          pending.delete(id);
        }
      }
    }
    states.set(agent, state);
    return state;
  };
  ctx.on('agent/pre-step', async ({ agent, messages }, next) => {
    if (messages.some(message => message.source.kind === 'user')) states.set(agent, fresh());
    const state = stateFor(agent);
    if (state.halted) return { kind: 'reject' };
    const decision = await next();
    if (decision.kind !== 'enter' || !state.pendingRecovery) return decision;
    state.pendingRecovery = false;
    return { ...decision, messages: [...decision.messages, recoveryNotice()] };
  });
  ctx.on('tools/pre-execute', async (exec, next) => {
    if (!exec.agent || exec.parent) return next();
    const state = stateFor(exec.agent);
    if (state.halted || (!polling.has(exec.name) && (state.calls.get(callKey(exec.name, exec.arguments))?.count ?? 0) >= 3)) {
      if (!state.halted && !state.recoveryUsed) {
        state.recoveryUsed = true;
        state.pendingRecovery = true;
        return { kind: 'deny', reason: `${blockedPrefix} ${exec.name} has already returned unchanged results three times. This call was not executed. Use the following recovery step to test a different explanation or report the blocker.` };
      }
      state.halted = true;
      return { kind: 'deny', reason: `${stoppedPrefix} ${exec.name} has already returned unchanged results three times and the recovery attempt did not resolve the loop. This call was not executed. The task is not verified complete.` };
    }
    return next();
  });
  ctx.on('tools/post-execute', async (exec, result, next) => {
    const decision = await next();
    if (!exec.agent || exec.parent || polling.has(exec.name)) return decision;
    const state = stateFor(exec.agent);
    if (state.halted || state.pendingRecovery) return decision;
    // Compare what the model actually receives, including downstream policy.
    const content = decision.kind === 'block' ? decision.feedback : decision.content ?? result.content;
    const count = observe(state, callKey(exec.name, exec.arguments), { content, isError: decision.kind === 'block' || result.isError });
    if (count !== 2) return decision;
    const warning = createUserMessage({
      source: { kind: 'plugin', plugin: name, form: 'notice', summary: 'Repeated call without progress' },
      content: [{ type: 'text', text: `${exec.name} returned the same result twice. Read any stderr even when exit status is zero. Fix the actual error, change approach, or report the blocker; repeating unchanged commands will stop this turn.` }],
    });
    return { ...decision, additionalContexts: [...decision.additionalContexts ?? [], warning] };
  });
}
