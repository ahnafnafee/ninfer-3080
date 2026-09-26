# Bonsai DSH preset

This directory is the repository copy of the RTX 3080 Bonsai coding preset and its local plugins.
Make future changes here, run the tests, and copy the preset files to the DSH installation.
The server profile remains in [`../windows/profiles.json`](../windows/profiles.json).

The six files in [`bonsai2-heretic-3080/`](bonsai2-heretic-3080/) contain the preset composition,
native NInfer adapter, prompt projection, recoverable compaction, and on-demand skill discovery.
The four JavaScript modules match the deployed implementation. The configuration derives the
optional user skill directory from `USERPROFILE` instead of embedding a particular user's path.
Credentials, complete host settings, conversation histories, context archives, and obsolete backups
are not part of this bundle.

## Install or update

This integration was verified on Windows with DSH **0.1.5-rc.3** and Node **24.19.0**. Use the Node
installation containing the global `@deepseek-ai/dsh` package: the plugins resolve its dependencies
from `node_modules/@deepseek-ai/dsh` beside `node.exe`. No installed DSH package is modified.

From this repository's root, copy the preset and prepare a host configuration fragment:

```powershell
$dshDirectory = if ($env:DSH_HOME) { $env:DSH_HOME } else { Join-Path $env:USERPROFILE '.dsh' }
$presetDirectory = Join-Path $dshDirectory '.agent-presets/bonsai2-heretic-3080'
if (Test-Path -LiteralPath $presetDirectory) {
    $backupDirectory = Join-Path $dshDirectory ('backups/bonsai2-heretic-3080-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
    New-Item -ItemType Directory -Path (Split-Path -Parent $backupDirectory) -Force | Out-Null
    Copy-Item -LiteralPath $presetDirectory -Destination $backupDirectory -Recurse
}
New-Item -ItemType Directory -Path $presetDirectory -Force | Out-Null
Get-ChildItem -LiteralPath './tools/dsh/bonsai2-heretic-3080' -File | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $presetDirectory -Force
}
$nativePluginUrl = ([uri](Join-Path (Resolve-Path -LiteralPath $presetDirectory).Path 'native-ninfer.mjs')).AbsoluteUri
$hostFragment = (Get-Content './tools/dsh/host.cordis.patch.example.yml' -Raw).Replace('{{NATIVE_PLUGIN_URL}}', $nativePluginUrl.Replace("'", "''"))
$hostFragment | Set-Content -LiteralPath (Join-Path $dshDirectory 'bonsai-host-fragment.yml') -Encoding utf8
```

Merge the generated `bonsai-host-fragment.yml` operation into the DSH home's `cordis.patch.yml`.
If it already contains `native-ninfer`, update that row instead of adding another. The example
endpoint is `http://127.0.0.1:18020/v1`, matching the tuned local deployment; the repository switcher
uses port 8080 without an override, so set `baseURL` to the actual server port.

Merge [`settings.example.yaml`](settings.example.yaml) into the DSH home's `settings.yaml`,
preserving unrelated fields. Remove duplicate `qwen-3080` and `qwen-3080-summary` entries from
`llm-pi-ai.providers` if present: the native adapter owns these two provider names. Retain other
providers and credentials. These are configuration fragments, not replacements for the whole files.
An existing installation with the same adapter registration and defaults only needs the file copy.
Restart DSH to load the updated modules.

Start NInfer with `./tools/windows/ninfer-switch.ps1 reason64`. That profile supplies the 65,536-token
minimum, FP32 recurrent state, MTP, 2,048 generated thinking tokens per step and 8,192 maximum output
tokens. The client discovers actual server capacity; its preset does not allocate GPU context.

## Behavior and qualification

The adapter counts complete prompts through NInfer before generation, reserves answer space, and
requests compaction when necessary. The compactor keeps exact original messages in workspace-local
archives with indexed references, while rejecting incomplete or tool-call-only summaries. The
preset preserves normal coding tools and their validation, bounds reads/search output, and discovers
skills on demand. Archive files have a local Git ignore rule.

The default model still makes reasoning and verification mistakes. The real long-history coding
diagnostic completed without context errors or output cutoffs, but required independent feedback to
correct its initial implementation. See the [precision and reliability report](../../docs/rtx-3080-precision.md#dsh-context-handling-and-coding-diagnostics)
for the measured results and limitations.

## Offline regression tests

Run with the Node installation described above. These tests use mock HTTP servers and temporary
workspaces; they do not read personal DSH configuration or require a loaded model.

```powershell
$env:BONSAI_LIVE_TOKEN_COUNT = '0'
node --test ./tools/dsh/tests/bonsai-local.test.mjs `
  ./tools/dsh/tests/native-ninfer.test.mjs `
  ./tools/dsh/tests/bonsai-prompt.test.mjs `
  ./tools/dsh/tests/bonsai-compaction-bounds.test.mjs `
  ./tools/dsh/tests/bonsai-compaction-recovery.test.mjs
```

All 44 checks pass on the verified installation. The optional `BONSAI_LIVE_TOKEN_COUNT=1` path
contacts the local server for token counts; it is not needed for the offline suite.
