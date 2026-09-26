import assert from 'node:assert/strict';
import { test } from 'node:test';
import { projectAssembly, apply } from '../bonsai2-heretic-3080/compact-prompt.mjs';

test('Bonsai preserves instruction authority and tool validation while removing the large catalog', () => {
  const parameters = { type: 'object', required: ['command'], additionalProperties: false,
    properties: { command: { type: 'string' }, sandbox_permissions: { enum: ['require_approval'], description: 'Only after sandbox denial and user approval.' } } };
  const assembly = { tools: [{ name: 'pwsh', description: 'Long background documentation. '.repeat(200), parameters },
    { name: 'mcp-large-catalog', description: 'External catalog. '.repeat(20000), parameters: {} }],
    sections: [{ name: 'plan:policy', text: 'Wait for plan approval.' }, { name: 'user:policy', text: 'Read AGENTS.md.' },
      { name: 'tool:pwsh', text: 'Inspect failed commands.' }, { name: 'tool:mcp-large-catalog', text: 'Catalog usage.' }],
    contexts: [{ name: 'sandbox', text: 'Sandbox and approval requirements.' }], variables: { cwd: 'C:/test' } };
  const projected = projectAssembly(assembly);
  assert.deepEqual(projected.tools.map(tool => tool.name), ['pwsh']);
  assert.deepEqual(projected.tools[0].parameters, parameters);
  assert.notEqual(projected.tools[0].parameters, parameters);
  assert.deepEqual(projected.sections.map(section => section.name), ['plan:policy', 'user:policy', 'tool:pwsh']);
  assert.deepEqual(projected.contexts, assembly.contexts);
  assert.equal(assembly.tools.length, 2, 'projection must not mutate a different preset sharing the catalog');
  assert.ok(JSON.stringify(projected).length < 1500);
});

test('restored maximum-thinking and output settings are migrated only for Bonsai', async () => {
  const listeners = new Map();
  apply({ on: (name, fn) => listeners.set(name, fn) });
  const request = listeners.get('agent/request');
  const old = { provider: 'qwen-3080', model: 'bonsai2-heretic', reasoningEffort: 'high', maxTokens: 32768 };
  assert.deepEqual(await request({}, async () => old), { ...old, reasoningEffort: 'medium', maxTokens: 8192 });
  assert.equal((await request({}, async () => ({ ...old, reasoningEffort: 'off' }))).reasoningEffort, 'off');
  const other = { ...old, provider: 'fast-qwen' };
  assert.equal(await request({}, async () => other), other);
});
