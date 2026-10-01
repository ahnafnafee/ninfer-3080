# Bonsai DSH preset

This directory is the repository copy of the RTX 3080 Bonsai coding preset and its local plugins.
Make future changes here, run the tests, and copy the preset files to the DSH installation.
The server profile remains in [`../windows/profiles.json`](../windows/profiles.json).

To select these and the other configured DSH models in Codex CLI, use the
[client gateway and `/model` setup](client-gateway/README.md).

The eight files in [`bonsai2-heretic-3080/`](bonsai2-heretic-3080/) contain the preset composition,
native NInfer adapter, timing projection, loop guard, prompt projection, recoverable compaction,
and on-demand skill discovery. The six JavaScript modules match the deployed implementation. The configuration derives the
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
$statsPluginUrl = ([uri](Join-Path (Resolve-Path -LiteralPath $presetDirectory).Path 'ninfer-session-stats.mjs')).AbsoluteUri
$hostFragment = (Get-Content './tools/dsh/host.cordis.patch.example.yml' -Raw).Replace('{{NATIVE_PLUGIN_URL}}', $nativePluginUrl.Replace("'", "''")).Replace('{{STATS_PLUGIN_URL}}', $statsPluginUrl.Replace("'", "''"))
$hostFragment | Set-Content -LiteralPath (Join-Path $dshDirectory 'bonsai-host-fragment.yml') -Encoding utf8
```

Merge the generated `bonsai-host-fragment.yml` operation into the DSH home's `cordis.patch.yml`.
If it already contains `native-ninfer`, update that row instead of adding another. Also apply the
`session-stats` disable operation and insert `ninfer-session-stats` once. DSH treats `name` in an
ordinary patch as a match condition, so changing the stock row's name does not replace its plugin.
The example
endpoint is `http://127.0.0.1:18020/v1`, matching the tuned local deployment; the repository switcher
uses port 8080 without an override, so set `baseURL` to the actual server port.

Merge [`settings.example.yaml`](settings.example.yaml) into the DSH home's `settings.yaml`,
preserving unrelated fields. Remove duplicate `qwen-3080` and `qwen-3080-summary` entries from
`llm-pi-ai.providers` if present: the native adapter owns these two provider names. Retain other
providers and credentials. These are configuration fragments, not replacements for the whole files.
After the adapter and stats registrations are installed, later module updates only need the file
copy. Restart DSH to load the updated modules; use `dsh web --port 3081 --no-open` to retain the
existing browser tab without opening another.

Start NInfer with `./tools/windows/llm-switch.ps1 reason64`. That profile supplies the 65,536-token
minimum, RK4 KV, FP32 recurrent state, low thinking capped at 2,048 tokens per step, temperature 0.2,
and 8,192 maximum output tokens. It unloads the MTP draft head to make room for the more accurate
cache. The client discovers actual server capacity; its preset does not allocate GPU context.
Low thinking remains enabled; medium and off are explicit alternatives. An existing session's
selection overrides the new-session default, so select low for a session previously set to off.

## Behavior and qualification

The adapter counts complete prompts through NInfer before generation, reserves answer space, and
requests compaction when necessary. The compactor keeps exact original messages in workspace-local
archives with indexed references, while rejecting incomplete or tool-call-only summaries. The
preset preserves normal coding tools and their validation, bounds reads/search output, and discovers
skills on demand. Archive files have a local Git ignore rule.

Checkpoint instructions are supplied as system instructions, separately from the quoted history.
When a summary reaches its output limit on a large fragment, the compactor halves that fragment
and continues from the last complete checkpoint. All unconsumed source is retained. If large
fragments overflow twice, or a checkpoint still cannot be shortened, a large history uses bounded
whole source excerpts plus its complete indexed archive. This fallback is labeled explicitly and
never accepts a failed model summary. Small spans still fail without replacing their history when
they cannot be compacted usefully; the base engine always checks that a replacement is smaller.

When old reasoning leaves insufficient answer space, preflight first omits reasoning from earlier
assistant steps and recounts the replacement prompt. It preserves the newest assistant step's
reasoning and every user instruction, visible answer, code block, tool call, and tool result.
The stored conversation is unchanged; `omittedReasoningBlocks` records the prompt adjustment in
the response metadata. If that still cannot fit, the existing compaction path handles the error.
This avoids rejecting a final answer merely because obsolete reasoning filled the window.

The loop guard warns after two identical call/result pairs and refuses another execution after
three unchanged results for the same arguments. It allows one recovery attempt per human turn,
with earlier reasoning omitted from subsequent prompts while all instructions, visible answers,
and tool evidence remain intact. A different operation can proceed; another blocked duplicate
ends the turn. Recovery consumption and denied-call evidence survive resume and compaction,
so an ignored warning cannot produce an unlimited stream of denied calls. The last 16 distinct call signatures
are tracked separately per agent; alternating calls cannot immediately reset detection. Background
job polling and status queries are exempt. New human input resets the guard, while compaction does
not; resumed agents reconstruct evidence from the last 256 durable events. This bounds runaway
execution, but does not establish that a model has solved the task or repaired its error.

NInfer parses tool calls at completion, so their SSE arrival time cannot measure generation speed.
The adapter records the server's measured `timings` in response metadata, and the host projection
uses `max(predicted_n - 1, 0) / (predicted_ms / 1000)` for decode throughput. Invalid or absent
timing metadata on an old tool-only record contributes neither decode tokens nor decode duration;
ordinary token-usage and wall-time totals are preserved. Old records with real text/reasoning
deltas retain their observable timing. Projection version 2 invalidates old cached calculations
when the session is loaded; cold sidebar hints can remain stale until that authoritative snapshot
arrives. Conversation events are never rewritten. Build the server change as well as installing
the plugins to obtain measured timings for new tool-only responses.

The preset tells the agent to run checks on the final revision, inspect errors, and preserve
independent expected results. Its live qualification checks the resulting code independently of
the agent's final message. Incorrect first drafts still occur; passing these fixtures establishes
the tested repair workflows, not universal model accuracy. See the
[precision and reliability report](../../docs/rtx-3080-precision.md#end-to-end-coding-qualification)
for the measured results and limitations.

## Offline regression tests

Run with the Node installation described above. These tests use mock HTTP servers and temporary
workspaces; they do not read personal DSH configuration or require a loaded model.

```powershell
$env:BONSAI_LIVE_TOKEN_COUNT = '0'
node --test ./tools/dsh/tests/*.test.mjs
```

The suite includes the real DSH tool executor and agent loop, server-timing folds, patch composition,
exact context limits, and compaction recovery. The optional `BONSAI_LIVE_TOKEN_COUNT=1` path
contacts the local server for token counts; it is not needed for the offline suite.

## Live accuracy qualification

Run these explicitly with the GPU server available. No personal DSH configuration is read. The
coding evaluator mounts DSH's real agent loop, adapter, prompt projection and loop guard, while
confining file tools to in-memory fixture files. Its `pwsh` tool dispatches only the independent
verifier; model output cannot execute an actual shell command or edit the oracle.

```powershell
node tools/dsh/eval/coding-agent.mjs short
$env:EVAL_CASE = 'money'
$env:EVAL_LONG_ROWS = '2300'
node tools/dsh/eval/coding-agent.mjs long
Remove-Item Env:EVAL_CASE, Env:EVAL_LONG_ROWS
node tools/dsh/eval/context-recovery.mjs
$env:EVAL_EFFORT = 'low'
$env:EVAL_TEMPERATURE = '0.2'
node tools/dsh/eval/tool-arguments.mjs tool-correction
Remove-Item Env:EVAL_EFFORT, Env:EVAL_TEMPERATURE
```

The three coding tasks cover decimal money, duration parsing and interval merging, with 315
independent cases. Success requires passing code, an observed verification of its final revision,
and a normally completed turn. The long variant inserts about 51K background tokens before the
final instruction. `EVAL_REPEATS`, `EVAL_EFFORT`, and `EVAL_TEMPERATURE` select repetitions or
controlled alternatives. Reports and fixture-only traces go under `profiles/accuracy-3080/`.
Without an effort override, `tool-arguments.mjs` compares none, medium and low; this is a diagnostic
matrix, so consult each `pass` result. Its `--reference` mode targets a separately started Prism
chat server through `NINFER_BASE_URL`; stop NInfer before loading another full GPU model.
