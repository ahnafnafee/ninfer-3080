import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { pathToFileURL } from 'node:url';
const requireDsh = createRequire(join(dirname(process.execPath), 'node_modules/@deepseek-ai/dsh/package.json'));
const upstream = await import(pathToFileURL(requireDsh.resolve('@deepseek-ai/dsh-session-stats')));

export function measuredTimings(value, outputTokens) {
  if (!value || !Number.isSafeInteger(value.predicted_n) || value.predicted_n !== outputTokens || value.predicted_n < 0) return undefined;
  if (![value.predicted_ms, value.ttft_ms].every(number => Number.isFinite(number) && number >= 0)) return undefined;
  if (value.predicted_n > 1 && value.predicted_ms === 0) return undefined;
  return { predicted_n: value.predicted_n, predicted_ms: value.predicted_ms, ttft_ms: value.ttft_ms };
}

export const name = 'ninfer-session-stats';
export const inject = ['sessionProjections'];
export function apply(ctx) {
  // Keep DSH's event/count/wire contract. Replace only its timing fold for the
  // local adapter; versioning rebuilds existing caches from the unchanged log.
  upstream.apply({ sessionProjections: { register(base) {
    ctx.sessionProjections.register({
      ...base,
      stateVersion: 2,
      apply(state, event) {
        const next = base.apply(state, event);
        const source = event.data?.message?.source;
        if (event.type !== 'assistant/message' || !['qwen-3080', 'qwen-3080-summary'].includes(source?.provider) || next === state) return next;
        const timing = measuredTimings(source.replayState?.response?.ninferTimings, event.data.usage?.outputTokens);
        // Text/reasoning deltas really stream. Tool deltas from this server are
        // parsed at completion, so their arrival cannot establish TTFT/decode.
        const visibleDelta = event.data.stream.some(record =>
          ((record.type === 'text-chunks' || record.type === 'reasoning-chunks') && record.texts.some(text => text !== '')) ||
          (record.type === 'chunk' && ['text-delta', 'reasoning-delta'].includes(record.chunk.type) && record.chunk.text !== ''));
        if (!timing && visibleDelta) return next;
        return {
          ...next,
          ttftMs: state.ttftMs + (timing?.ttft_ms ?? 0),
          ttftSteps: state.ttftSteps + (timing ? 1 : 0),
          decodeMs: state.decodeMs + (timing?.predicted_ms ?? 0),
          decodeTokens: state.decodeTokens + (timing ? Math.max(0, timing.predicted_n - 1) : 0),
        };
      },
    });
  } } });
}
