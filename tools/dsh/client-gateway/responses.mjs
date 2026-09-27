import { randomUUID } from 'node:crypto';

export class RequestError extends Error {
  constructor(message, code = 'invalid_request_error', status = 400) { super(message); this.code = code; this.status = status; }
}
const source = { kind: 'plugin', plugin: 'client-gateway' };
const textContent = content => {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) throw new RequestError('Expected text content');
  return content.map(block => {
    if (['input_text', 'output_text', 'text', 'summary_text', 'reasoning_text'].includes(block.type)) return block.text;
    throw new RequestError(`Unsupported content: ${block.type}. This gateway advertises text input only.`);
  }).join('\n');
};
export function translateRequest(body, route, signal) {
  if (body.previous_response_id) throw new RequestError('Send complete conversation history; previous_response_id is not supported.');
  const toolMap = new Map();
  const tools = [];
  const addTool = (tool, namespace) => {
    if (tool.type === 'namespace') { for (const child of tool.tools) addTool(child, tool.name); return; }
    if (!['function', 'custom'].includes(tool.type)) throw new RequestError(`Unsupported tool type: ${tool.type}`);
    const name = namespace ? `${namespace}__${tool.name}` : tool.name;
    if (toolMap.has(name)) throw new RequestError(`Duplicate tool: ${name}`);
    toolMap.set(name, { ...tool, namespace });
    tools.push({ name, description: tool.description ?? '', parameters: tool.type === 'custom'
      ? { type: 'object', properties: { input: { type: 'string', description: 'The complete raw tool input, including any patch or code.' } }, required: ['input'], additionalProperties: false }
      : tool.parameters ?? { type: 'object', properties: {} } });
  };
  for (const tool of body.tools ?? []) addTool(tool);
  const messages = [];
  const system = body.instructions ? [body.instructions] : [];
  const add = (role, block) => {
    const last = messages.at(-1);
    if (last?.role === role) last.content.push(block);
    else messages.push({ role, source, content: [block] });
  };
  for (const item of typeof body.input === 'string' ? [{ role: 'user', content: body.input }] : body.input ?? []) {
    if (item.type === 'function_call' || item.type === 'custom_tool_call') {
      add('assistant', { type: 'tool-call', id: item.call_id,
        name: item.namespace ? `${item.namespace}__${item.name}` : item.name,
        arguments: item.type === 'custom_tool_call' ? JSON.stringify({ input: item.input }) : item.arguments });
    } else if (item.type === 'function_call_output' || item.type === 'custom_tool_call_output') {
      add('user', { type: 'tool-result', toolCallId: item.call_id, content: [{ type: 'text', text: textContent(item.output) }] });
    } else if (item.type === 'reasoning') {
      const text = textContent(item.summary?.length ? item.summary : item.content ?? []);
      if (text) add('assistant', { type: 'reasoning', text });
    } else if (item.type === 'message' || (!item.type && item.role)) {
      const text = textContent(item.content);
      if (['system', 'developer'].includes(item.role)) system.push(text);
      else if (['user', 'assistant'].includes(item.role)) add(item.role, { type: 'text', text });
      else throw new RequestError(`Unsupported role: ${item.role}`);
    } else throw new RequestError(`Unsupported input item: ${item.type}`);
  }
  const efforts = route.info.reasoning?.efforts.map(level => level.id) ?? ['off'];
  const requestedEffort = body.reasoning?.effort === 'none' ? 'off' : body.reasoning?.effort;
  const effort = requestedEffort ?? route.info.reasoning?.defaultEffort ?? 'off';
  if (!efforts.includes(effort)) throw new RequestError(`${route.id} supports thinking: ${efforts.join(', ')}; selected ${effort}.`);
  return { toolMap, options: {
    provider: route.provider, model: route.model, messages, system: system.join('\n\n'),
    tools, reasoningEffort: effort, signal,
    maxTokens: Math.min(body.max_output_tokens ?? route.info.defaultMaxTokens ?? 8192, route.info.defaultMaxTokens ?? 32768),
    ...(body.temperature === undefined ? {} : { temperature: body.temperature }),
  } };
}

/** Adapt DSH's stream contract to the Responses subset used by the CLI. */
export async function* responseEvents(chunks, model, toolMap) {
  const response = { id: `resp_${randomUUID().replaceAll('-', '')}`, object: 'response', created_at: Math.floor(Date.now() / 1000),
    model, status: 'in_progress', output: [], error: null, incomplete_details: null, usage: null };
  let sequence = 0;
  const event = (type, fields) => ({ type, sequence_number: sequence++, ...fields });
  yield event('response.created', { response: { ...response, output: [] } });
  yield event('response.in_progress', { response: { ...response, output: [] } });
  const blocks = new Map();
  const pendingCalls = [];
  let finish;
  for await (const chunk of chunks) {
    if (chunk.type === 'block-start' && chunk.blockType !== 'tool-call') {
      const reasoning = chunk.blockType === 'reasoning';
      const item = reasoning
        ? { id: `rs_${randomUUID().replaceAll('-', '')}`, type: 'reasoning', summary: [] }
        : { id: `msg_${randomUUID().replaceAll('-', '')}`, type: 'message', role: 'assistant', content: [], status: 'in_progress' };
      const block = { item, index: response.output.length, reasoning, text: '' };
      response.output.push(item); blocks.set(chunk.index, block);
      yield event('response.output_item.added', { output_index: block.index, item: structuredClone(item) });
      const part = reasoning ? { type: 'summary_text', text: '' } : { type: 'output_text', text: '', annotations: [] };
      yield event(reasoning ? 'response.reasoning_summary_part.added' : 'response.content_part.added',
        { item_id: item.id, output_index: block.index, ...(reasoning ? { summary_index: 0 } : { content_index: 0 }), part });
    } else if (chunk.type === 'text-delta' || chunk.type === 'reasoning-delta') {
      const block = blocks.get(chunk.index);
      if (!block) throw new Error('Adapter delta has no starting block');
      block.text += chunk.text;
      yield event(block.reasoning ? 'response.reasoning_summary_text.delta' : 'response.output_text.delta',
        { item_id: block.item.id, output_index: block.index, ...(block.reasoning ? { summary_index: 0 } : { content_index: 0 }), delta: chunk.text });
    } else if (chunk.type === 'block-end') {
      if (chunk.block.type === 'tool-call') {
        pendingCalls.push(chunk.block);
      } else {
        const block = blocks.get(chunk.index);
        if (!block) throw new Error('Adapter end has no starting block');
        // Some adapters supply text only on block-end (including error context).
        const text = chunk.block.text;
        if (!block.text && text) yield event(block.reasoning ? 'response.reasoning_summary_text.delta' : 'response.output_text.delta',
          { item_id: block.item.id, output_index: block.index, ...(block.reasoning ? { summary_index: 0 } : { content_index: 0 }), delta: text });
        const part = block.reasoning ? { type: 'summary_text', text } : { type: 'output_text', text, annotations: [] };
        if (block.reasoning) block.item.summary = [part];
        else { block.item.content = [part]; block.item.status = 'completed'; }
        yield event(block.reasoning ? 'response.reasoning_summary_text.done' : 'response.output_text.done',
          { item_id: block.item.id, output_index: block.index, ...(block.reasoning ? { summary_index: 0 } : { content_index: 0 }), text });
        yield event(block.reasoning ? 'response.reasoning_summary_part.done' : 'response.content_part.done',
          { item_id: block.item.id, output_index: block.index, ...(block.reasoning ? { summary_index: 0 } : { content_index: 0 }), part });
        yield event('response.output_item.done', { output_index: block.index, item: block.item });
      }
    } else if (chunk.type === 'usage') {
      const usage = chunk.usage;
      const input = (usage.inputTokens ?? 0) + (usage.cacheReadTokens ?? 0);
      response.usage = { input_tokens: input, output_tokens: usage.outputTokens ?? 0,
        total_tokens: input + (usage.outputTokens ?? 0), input_tokens_details: { cached_tokens: usage.cacheReadTokens ?? 0 },
        output_tokens_details: { reasoning_tokens: usage.reasoningTokens ?? 0 } };
    } else if (chunk.type === 'finish') { finish = chunk.reason; break; }
  }
  if (!finish) throw new Error('Adapter ended without a finish event');
  // A complete-looking tool block can precede a length/error finish. Publish
  // calls only after a successful terminal result, so clients cannot execute
  // a tool from an interrupted generation.
  if (['stop', 'tool-calls'].includes(finish.kind)) for (const call of pendingCalls) {
    const tool = toolMap.get(call.name);
    if (!tool) throw new Error(`Model called undeclared tool: ${call.name}`);
    const args = JSON.parse(call.arguments);
    if (!args || typeof args !== 'object' || Array.isArray(args)) throw new Error('Tool arguments must be an object');
    const custom = tool.type === 'custom';
    if (custom && typeof args.input !== 'string') throw new Error('Custom tool did not return a string input');
    const item = { type: custom ? 'custom_tool_call' : 'function_call', id: `fc_${randomUUID().replaceAll('-', '')}`,
      call_id: call.id, name: tool.name, ...(tool.namespace ? { namespace: tool.namespace } : {}), status: 'completed',
      ...(custom ? { input: args.input } : { arguments: call.arguments }) };
    const index = response.output.length;
    response.output.push(item);
    yield event('response.output_item.added', { output_index: index, item: { ...item, status: 'in_progress', ...(custom ? { input: '' } : { arguments: '' }) } });
    yield event(custom ? 'response.custom_tool_call_input.delta' : 'response.function_call_arguments.delta',
      { output_index: index, item_id: item.id, delta: custom ? args.input : call.arguments });
    yield event(custom ? 'response.custom_tool_call_input.done' : 'response.function_call_arguments.done',
      { output_index: index, item_id: item.id, ...(custom ? { input: args.input } : { arguments: call.arguments }) });
    yield event('response.output_item.done', { output_index: index, item });
  }
  if (finish.kind === 'error' || finish.kind === 'aborted') {
    response.status = 'failed';
    response.error = { code: finish.failure.code === 'CONTEXT_WINDOW_EXCEEDED' ? 'context_length_exceeded' : finish.failure.code, message: finish.failure.message };
    yield event('response.failed', { response });
  } else if (finish.kind === 'max-tokens') {
    response.status = 'incomplete'; response.incomplete_details = { reason: 'max_output_tokens' };
    yield event('response.incomplete', { response });
  } else {
    response.status = 'completed';
    yield event('response.completed', { response });
  }
}
