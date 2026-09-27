import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { homedir } from 'node:os';
import { readFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import { parseEnv } from 'node:util';

// Use the same installed adapters and credential references as DSH. Secrets
// remain in its credential store, never in the generated client catalog.
const requireDsh = createRequire(join(dirname(process.execPath), 'node_modules/@deepseek-ai/dsh/package.json'));
const moduleOf = name => import(pathToFileURL(requireDsh.resolve('@deepseek-ai/' + name)));
const yaml = requireDsh('js-yaml');
async function optionalText(path) {
  try { return await readFile(path, 'utf8'); }
  catch (error) { if (error.code === 'ENOENT') return ''; throw error; }
}

export async function loadRuntime(dshHome = process.env.DSH_HOME ?? join(homedir(), '.dsh')) {
  const settings = yaml.load(await readFile(join(dshHome, 'settings.yaml'), 'utf8'));
  const patch = yaml.load(await optionalText(join(dshHome, 'cordis.patch.yml'))) ?? [];
  const rows = patch.flatMap(row => [row, ...(row.insert ?? [])]);
  const { parseCredentialsDocument } = await moduleOf('dsh-credentials-local');
  const credentialPath = join(dshHome, '.credentials.yaml');
  const credentials = async () => {
    const text = await optionalText(credentialPath);
    return text ? parseCredentialsDocument(text, credentialPath) : { refs: new Map(), records: new Map() };
  };
  const environment = { ...parseEnv(await optionalText(join(dshHome, '.env'))), ...process.env };
  const adapters = new Map();
  // Mount only the adapter registration seams. No DSH agent, tools, UI, or
  // session storage is started, and no credential record is modified.
  const context = {
    get(name) {
      if (name === 'launchEnvironment') return { get: key => environment[key] ? { value: environment[key] } : undefined };
      if (name === 'credentials') return {
        async resolve(key) { const value = (await credentials()).refs.get(key) ?? environment[key]; return value ? { value } : undefined; },
        async readRecord(key) { return (await credentials()).records.get(key); },
        async listRecords() { return [...(await credentials()).records].map(([key, record]) => ({ key, kind: record.kind })); },
      };
    },
    inject() {},
    logger: { warn() {}, error() {} },
    llm: {
      registerAdapter(providers, adapter) { for (const provider of providers) adapters.set(provider, adapter); return { replace() {} }; },
      registerConfigurableProviders() { return { replace() {} }; },
      registerModelDiscovery() {},
    },
  };
  const pi = await moduleOf('dsh-llm-pi-ai');
  pi.apply(context, settings['llm-pi-ai'] ?? { providers: {} });
  const deepseek = await moduleOf('dsh-llm-deepseek');
  const deepseekRow = rows.find(row => row.id === 'llm-deepseek');
  if (!deepseekRow?.disabled) deepseek.apply(context, settings['llm-deepseek'] ?? deepseekRow?.config ?? {});
  const native = rows.find(row => row.id === 'native-ninfer' && !row.disabled);
  if (native) {
    const { NativeNinferAdapter } = await import('../bonsai2-heretic-3080/native-ninfer.mjs');
    const adapter = new NativeNinferAdapter(native.config ?? {});
    for (const provider of ['qwen-3080', 'qwen-3080-summary']) adapters.set(provider, adapter);
  }
  const routes = new Map();
  const failures = [];
  for (const [provider, adapter] of adapters) {
    try {
      // Native listing reads the loaded server; other adapters use DSH's exact
      // configured catalog. A model cannot silently fall back to another route.
      const models = provider.startsWith('qwen-3080') && !provider.includes('legacy')
        ? [await adapter.resolveModel(provider, native.config?.model ?? 'bonsai2-heretic', AbortSignal.timeout(10000))]
        : await adapter.listModels(provider);
      for (const model of models) {
        try {
          const info = await adapter.resolveModel(provider, model.id, AbortSignal.timeout(10000));
          const id = `${provider}/${model.id}`;
          routes.set(id, { id, provider, model: model.id, info, adapter });
        } catch (error) { failures.push({ provider, model: model.id, message: error.message }); }
      }
    } catch (error) { failures.push({ provider, message: error.message }); }
  }
  return { routes, failures, defaultModel: `${settings['agent-default-model']?.provider}/${settings['agent-default-model']?.model}` };
}

export function clientCatalog(runtime) {
  return { models: [...runtime.routes.values()].map((route, index) => {
    const info = route.info;
    const levels = (info.reasoning?.efforts ?? [{ id: 'off' }]).map(level => level.id === 'off' ? 'none' : level.id);
    const selected = info.reasoning?.defaultEffort === 'off' ? 'none' : info.reasoning?.defaultEffort ?? 'none';
    return {
      slug: route.id, display_name: `${info.name ?? route.model} [${route.provider}]`,
      description: `DSH ${route.provider}; ${info.context.contextWindow.toLocaleString('en-US')} token context`,
      base_instructions: 'You are a coding assistant working in the user workspace. Follow the user request and applicable AGENTS.md instructions. Inspect relevant source before editing, use the available tools, and verify the final change with appropriate checks. Treat command errors as evidence to correct your approach. Do not repeat unchanged failed commands. Preserve independent expected test results and report material limitations honestly. Keep tool output bounded and retain exact paths and requirements when summarizing.',
      default_reasoning_level: selected,
      supported_reasoning_levels: levels.map(effort => ({ effort, description: effort === 'none' ? 'Thinking off' : `${effort} thinking` })),
      shell_type: 'unified_exec', visibility: 'list', supported_in_api: true,
      priority: route.id === runtime.defaultModel ? 0 : index + 1,
      additional_speed_tiers: [], service_tiers: [], upgrade: null,
      include_skills_usage_instructions: false, include_plugin_usage_instructions: false, include_apps_usage_instructions: false,
      default_reasoning_summary: 'none', support_verbosity: false,
      apply_patch_tool_type: 'freeform', web_search_tool_type: 'text',
      truncation_policy: { mode: 'tokens', limit: 6000 },
      supports_image_detail_original: false, context_window: info.context.contextWindow,
      max_context_window: info.context.contextWindow, effective_context_window_percent: 85,
      experimental_supported_tools: [], input_modalities: ['text'],
      supports_search_tool: false, supports_experimental_context: false,
      use_responses_lite: false, supports_reasoning_effort_updates: true,
      node_repl_auto_review_required: false, node_repl_disabled: true,
    };
  }) };
}
