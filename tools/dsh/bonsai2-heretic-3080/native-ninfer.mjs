import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { pathToFileURL } from 'node:url';

const requireDsh = createRequire(join(dirname(process.execPath), 'node_modules/@deepseek-ai/dsh/package.json'));
const { LlmAdapter, LlmError, attributionHeaders } = await import(pathToFileURL(requireDsh.resolve('@deepseek-ai/dsh-llm')));

const ROUTES = ['qwen-3080', 'qwen-3080-summary'];
const terminalTypes = new Set(['response.completed', 'response.incomplete', 'response.failed', 'response.cancelled']);

function positiveInteger(value, label) {
  if (!Number.isSafeInteger(value) || value < 1) throw new Error(`${label} must be a positive integer`);
  return value;
}

function contentText(blocks) {
  return blocks.map(block => {
    if (block.type === 'text' || block.type === 'reasoning') return block.text;
    if (block.type === 'file') return `File: ${JSON.stringify(block.attachment)}`;
    throw new LlmError(`The local Bonsai model cannot accept ${block.type} content.`, 'UNSUPPORTED_CONTENT');
  }).join('\n');
}

/** Translate the persisted message blocks without dropping user or tool-result content. */
export function responseInput(messages) {
  const resultIds = new Set(messages.flatMap(message => message.content.filter(block => block.type === 'tool-result').map(block => block.toolCallId)));
  const input = [];
  for (const message of messages) {
    for (const block of message.content) {
      if (block.type === 'tool-result') {
        input.push({ type: 'function_call_output', call_id: block.toolCallId, output: contentText(block.content) || '(no output)' });
      } else if (message.role === 'assistant' && block.type === 'tool-call') {
        // A length-limited answer can contain unfinished or unexecuted calls. They
        // remain visible history, but must never become executable replay items.
        let validArguments = false;
        try { const args = JSON.parse(block.arguments); validArguments = args !== null && typeof args === 'object' && !Array.isArray(args); } catch {}
        if (resultIds.has(block.id) && validArguments) {
          input.push({ type: 'function_call', call_id: block.id, name: block.name, arguments: block.arguments });
        } else {
          input.push({ type: 'message', role: 'assistant', content: `[Unexecuted tool call ${block.name}: ${block.arguments}]` });
        }
      } else if (message.role === 'assistant' && block.type === 'reasoning') {
        if (block.text) input.push({ type: 'reasoning', content: [{ type: 'reasoning_text', text: block.text }] });
      } else if (block.type === 'text' || block.type === 'file') {
        const text = contentText([block]);
        if (text) input.push({ type: 'message', role: message.role, content: text });
      } else {
        throw new LlmError(`Unsupported ${block.type} block in ${message.role} history.`, 'UNSUPPORTED_CONTENT');
      }
    }
  }
  return input;
}

/** The same prompt body is counted and generated; only transport/output fields differ. */
export function responseRequest(options) {
  if (options.stop?.length) throw new LlmError('The local Responses route does not accept custom stop strings.', 'UNSUPPORTED_OPTION');
  return {
    model: options.model,
    input: responseInput(options.messages),
    ...(options.system ? { instructions: options.system } : {}),
    reasoning: { effort: options.provider === 'qwen-3080-summary' || options.purpose === 'session-title' ? 'none' : (options.reasoningEffort === 'off' ? 'none' : options.reasoningEffort ?? 'medium') },
    ...(options.tools?.length ? { tools: options.tools.map(tool => ({ type: 'function', name: tool.name, description: tool.description, parameters: tool.parameters, strict: false })) } : {}),
    truncation: 'disabled',
  };
}

function normalizeFailure(error, status) {
  const detail = error?.error ?? error;
  let message = typeof detail?.message === 'string' ? detail.message : 'NInfer request failed.';
  const code = detail?.code === 'context_length_exceeded' || /context.*exceed|exceed.*context/i.test(message)
    ? 'CONTEXT_WINDOW_EXCEEDED' : (status >= 500 ? 'SERVER' : 'INVALID_REQUEST');
  if (code === 'CONTEXT_WINDOW_EXCEEDED') message += ' Automatic compaction can reduce earlier history; if this persists, shorten the latest message or supply large material as files to read in smaller sections.';
  return new LlmError(message, code, status ? { status } : undefined);
}

export async function* sseEvents(body) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
      buffer = buffer.replace(/\r\n/g, '\n');
      let boundary;
      while ((boundary = buffer.indexOf('\n\n')) >= 0) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const data = frame.split('\n').filter(line => line.startsWith('data:')).map(line => line.slice(5).trimStart()).join('\n');
        if (data && data !== '[DONE]') yield JSON.parse(data);
      }
      if (done) break;
    }
    if (buffer.trim() && !buffer.trim().startsWith(':')) throw new LlmError('NInfer ended an incomplete SSE frame.', 'TRANSPORT');
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

/** Stream text immediately, but expose complete tool JSON only after its item closes. */
export async function* responseChunks(events) {
  const blocks = new Map();
  const pendingTools = [];
  let nextIndex = 0;
  let sawTerminal = false;
  let sawTool = false;
  const keyFor = event => `${event.output_index}:${event.content_index ?? 0}`;
  for await (const event of events) {
    if (event.type === 'response.output_text.delta' || event.type === 'response.reasoning_text.delta') {
      const key = keyFor(event);
      const type = event.type === 'response.output_text.delta' ? 'text' : 'reasoning';
      let block = blocks.get(key);
      if (!block) {
        block = { index: nextIndex++, type, text: '', closed: false };
        blocks.set(key, block);
        yield { type: 'block-start', index: block.index, blockType: type };
      }
      block.text += event.delta;
      yield { type: type === 'text' ? 'text-delta' : 'reasoning-delta', index: block.index, text: event.delta };
    } else if (event.type === 'response.output_text.done' || event.type === 'response.reasoning_text.done') {
      const key = keyFor(event);
      let block = blocks.get(key);
      if (!block) {
        const type = event.type === 'response.output_text.done' ? 'text' : 'reasoning';
        block = { index: nextIndex++, type, text: event.text ?? '', closed: false };
        blocks.set(key, block);
        yield { type: 'block-start', index: block.index, blockType: type };
      }
      if (!block.closed) {
        block.closed = true;
        yield { type: 'block-end', index: block.index, block: { type: block.type, text: event.text ?? block.text } };
      }
    } else if (event.type === 'response.output_item.done' && event.item?.type === 'function_call') {
      pendingTools.push(event.item);
    } else if (terminalTypes.has(event.type)) {
      sawTerminal = true;
      for (const block of blocks.values()) if (!block.closed) {
        block.closed = true;
        yield { type: 'block-end', index: block.index, block: { type: block.type, text: block.text } };
      }
      const response = event.response ?? {};
      for (const item of pendingTools) {
        const index = nextIndex++;
        let args;
        try { args = JSON.parse(item.arguments); } catch {}
        const complete = event.type === 'response.completed' && (item.status === undefined || item.status === 'completed') && args !== null && typeof args === 'object' && !Array.isArray(args);
        if (complete) {
          sawTool = true;
          yield { type: 'block-start', index, blockType: 'tool-call' };
          yield { type: 'tool-call-delta', index, id: item.call_id, name: item.name, argumentsDelta: item.arguments };
          yield { type: 'block-end', index, block: { type: 'tool-call', id: item.call_id, name: item.name, arguments: item.arguments } };
        } else {
          yield { type: 'block-start', index, blockType: 'text' };
          yield { type: 'block-end', index, block: { type: 'text', text: `[Unexecuted tool call ${item.name}: ${item.arguments}]` } };
        }
      }
      if (response.usage) {
        const usage = response.usage;
        const cached = usage.input_tokens_details?.cached_tokens ?? 0;
        yield { type: 'usage', usage: { inputTokens: Math.max(0, usage.input_tokens - cached), outputTokens: usage.output_tokens, cacheReadTokens: cached, totalTokens: usage.total_tokens ?? usage.input_tokens + usage.output_tokens, reasoningTokens: usage.output_tokens_details?.reasoning_tokens ?? 0 } };
      }
      if (event.type === 'response.failed') {
        const error = normalizeFailure(response.error ?? event.error);
        yield { type: 'finish', reason: { kind: 'error', failure: error.failure } };
      } else if (event.type === 'response.cancelled') {
        yield { type: 'finish', reason: { kind: 'aborted', failure: { code: 'ABORTED', message: 'NInfer generation was cancelled.' } } };
      } else {
        yield { type: 'finish', reason: { kind: event.type === 'response.incomplete' ? 'max-tokens' : sawTool ? 'tool-calls' : 'stop' } };
      }
      return;
    }
  }
  if (!sawTerminal) throw new LlmError('NInfer stream ended before its terminal response.', 'TRANSPORT');
}

export class NativeNinferAdapter extends LlmAdapter {
  constructor(config = {}, fetchImpl = fetch) {
    super();
    this.baseURL = (config.baseURL ?? 'http://127.0.0.1:18020/v1').replace(/\/$/, '');
    this.model = config.model ?? 'bonsai2-heretic';
    this.maxTokens = positiveInteger(config.maxTokens ?? 4096, 'maxTokens');
    this.summaryMaxTokens = positiveInteger(config.summaryMaxTokens ?? 768, 'summaryMaxTokens');
    this.minOutputTokens = positiveInteger(config.minOutputTokens ?? 4096, 'minOutputTokens');
    this.reserveTokens = positiveInteger(config.reserveTokens ?? 256, 'reserveTokens');
    this.generationTimeoutMs = positiveInteger(config.generationTimeoutMs ?? 600000, 'generationTimeoutMs');
    this.fetchImpl = fetchImpl;
  }
  providerInfo(provider) { return { id: provider, name: provider.endsWith('-summary') ? 'NInfer Bonsai summary' : 'NInfer Bonsai (RTX 3080)' }; }
  async request(path, body, signal) {
    const timeout = AbortSignal.timeout(path === '/responses' ? this.generationTimeoutMs : 30000);
    const response = await this.fetchImpl(this.baseURL + path, {
      method: body === undefined ? 'GET' : 'POST',
      headers: { ...attributionHeaders(), ...(body === undefined ? {} : { 'content-type': 'application/json' }) },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
      signal: signal === undefined ? timeout : AbortSignal.any([signal, timeout]),
    });
    if (!response.ok) {
      let error;
      try { error = await response.json(); } catch { error = { message: `NInfer returned HTTP ${response.status}.` }; }
      throw normalizeFailure(error, response.status);
    }
    return response;
  }
  async metadata(provider, model, signal) {
    if (!ROUTES.includes(provider) || model !== this.model) throw new LlmError(`NInfer route serves only ${this.model}.`, 'MODEL_NOT_FOUND');
    const result = await (await this.request('/models', undefined, signal)).json();
    const entry = result.data?.find(candidate => candidate.id === model);
    const capacity = entry?.context_window ?? entry?.max_model_len;
    if (!Number.isSafeInteger(capacity) || capacity <= 0) throw new LlmError('NInfer did not advertise the selected model and its context capacity.', 'MODEL_NOT_FOUND');
    const summary = provider === 'qwen-3080-summary';
    return { provider, id: model, name: 'Bonsai 2 Heretic', inputModalities: ['text'], context: { contextWindow: capacity }, defaultMaxTokens: summary ? this.summaryMaxTokens : this.maxTokens, reasoning: { efforts: (summary ? ['off'] : ['medium', 'off']).map(id => ({ id, name: id })), defaultEffort: summary ? 'off' : 'medium' } };
  }
  async listModels(provider) { return [await this.metadata(provider, this.model)]; }
  async resolveModel(provider, model, signal) { return this.metadata(provider, model, signal); }
  async prepareCall(provider, model, signal) {
    const metadata = await this.metadata(provider, model, signal);
    return { model: metadata, stream: options => this.streamWithMetadata(options, metadata) };
  }
  async *stream(options) { yield* this.streamWithMetadata(options, await this.metadata(options.provider, options.model, options.signal)); }
  async *streamWithMetadata(options, metadata) {
    try {
      const body = responseRequest(options);
      const counted = await (await this.request('/responses/input_tokens', body, options.signal)).json();
      const inputTokens = counted.input_tokens;
      if (!Number.isSafeInteger(inputTokens) || inputTokens < 0) throw new LlmError('NInfer returned an invalid prompt token count.', 'INVALID_RESPONSE');
      const requested = Math.min(positiveInteger(options.maxTokens ?? metadata.defaultMaxTokens, 'maxTokens'), metadata.defaultMaxTokens);
      const available = metadata.context.contextWindow - inputTokens - this.reserveTokens;
      const minimum = options.provider === 'qwen-3080-summary' ? requested : Math.min(requested, this.minOutputTokens);
      if (available < minimum) {
        throw new LlmError(`The prepared prompt uses ${inputTokens} tokens; at least ${minimum} output tokens and ${this.reserveTokens} reserve tokens require compaction within the active ${metadata.context.contextWindow}-token NInfer context. If automatic compaction cannot make room, shorten the latest message or supply large material as files to read in smaller sections.`, 'CONTEXT_WINDOW_EXCEEDED');
      }
      const response = await this.request('/responses', { ...body, max_output_tokens: Math.min(requested, available), ...(options.temperature === undefined ? {} : { temperature: options.temperature }), stream: true, store: false }, options.signal);
      if (!response.body) throw new LlmError('NInfer response has no stream body.', 'TRANSPORT');
      yield* responseChunks(sseEvents(response.body));
    } catch (error) {
      if (options.signal?.aborted) {
        yield { type: 'finish', reason: { kind: 'aborted', failure: { code: 'ABORTED', message: 'NInfer request cancelled.' } } };
      } else {
        const failure = error instanceof LlmError ? error.failure : { code: 'TRANSPORT', message: error.message ?? 'NInfer request failed.' };
        yield { type: 'finish', reason: { kind: 'error', failure } };
      }
    }
  }
}

export const name = 'native-ninfer';
export const inject = ['llm'];
export function apply(ctx, config = {}) {
  ctx.llm.registerAdapter(ROUTES, new NativeNinferAdapter(config));
}
