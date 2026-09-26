// Keep the coding tools in the durable request header without carrying the
// deployment's entire MCP catalog into every local model request.
export const name = 'bonsai-compact-prompt';
export const inject = ['systemPrompt'];

export const descriptions = Object.freeze({
  pwsh: 'Execute PowerShell in the workspace. Inspect failed commands. Use background jobs for long commands; request wider access only after a sandbox denial and approval.',
  bash: 'Execute a shell command in the workspace. Inspect failures; use background jobs for long commands.',
  read: 'Read a UTF-8 file with line numbers. Use offset and limit for small relevant ranges.',
  write: 'Write a complete UTF-8 file. Read existing files first and use the same file_path spelling; preserve unrelated content.',
  edit: 'Replace an exact string in a text file. Read first, use the same file_path spelling, and supply the exact original text.',
  glob: 'Find workspace paths matching a glob.',
  grep: 'Search file contents with a ripgrep regular expression. Use read for surrounding lines.',
  job_output: 'Collect output from a background job.',
  job_list: 'List the session background jobs.',
  job_kill: 'Stop a background job.',
  skill_search: 'Find skills by keyword before a matching task. Returns names and short descriptions; use skill_load for instructions.',
  skill_load: 'Load instructions for one exact skill name returned by skill_search.',
  web_search: 'Search the web for current facts and sources.',
  web_fetch: 'Fetch a web page for inspection.',
  ask_user_question: 'Ask the user a necessary clarification or approval question.',
  exit_plan_mode: 'Present the implementation plan for approval and exit plan mode only when approved.',
  todo_write: 'Update the task checklist and its completion state.',
});

// Preserve the preset's native planning/delegation capabilities at 64K.
const nativeTools = new Set(['get_goal', 'create_goal', 'update_goal', 'send_message',
  'interrupt_agent', 'list_agents', 'subagent', 'subagent_fork', 'workflow', 'ralph']);

export function projectAssembly(assembly) {
  return {
    ...assembly,
    tools: assembly.tools.filter(tool => Object.hasOwn(descriptions, tool.name) || nativeTools.has(tool.name)).map(tool => ({
      ...tool,
      description: descriptions[tool.name] ?? tool.description,
      // Keep parameter documentation, required fields and validation intact.
      parameters: structuredClone(tool.parameters),
    })),
    sections: assembly.sections.filter(section => !section.name.startsWith('tool:') || ['jobs', 'goal', 'subagent', 'workflow', 'ralph'].includes(section.name.slice(5)) || Object.hasOwn(descriptions, section.name.slice(5))),
  };
}

export function apply(ctx) {
  ctx.on('system-prompt/assemble', async (_assembly, _context, next) => projectAssembly(await next()));
  // A restored session can still carry the old maximum-thinking/output choice.
  ctx.on('agent/request', async (_request, next) => {
    const config = await next();
    if (config.provider !== 'qwen-3080' || config.model !== 'bonsai2-heretic') return config;
    return { ...config, reasoningEffort: config.reasoningEffort === 'off' ? 'off' : 'medium', maxTokens: 8192 };
  });
}
