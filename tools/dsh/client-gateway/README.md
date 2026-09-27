# Use DSH models in Codex CLI

Launch `codex -p dsh`, then enter `/model` to choose a configured DSH model and
its reasoning effort. The picker can apply a choice to the current session
with **s**, or save it as the profile default with **Enter**. Plain `codex`
keeps the existing base configuration and provider.

This integration was verified with Codex CLI 0.157.0, DSH 0.1.5-rc.3 and Node
24.19.0 on Windows. The CLI uses one provider connection per session. Its
[custom provider transport](https://learn.chatgpt.com/docs/config-file/config-reference)
accepts Responses, while the configured DSH services also use Chat Completions
and Anthropic Messages. This localhost gateway reuses the installed DSH adapters
and exposes the text/tool portion of Responses needed by the CLI. It does not
run another agent loop or change DSH sessions.

## Install and refresh

Use the Node installation that contains the global DSH package, as described in
the [preset instructions](../README.md). From this repository:

```powershell
node tools/dsh/client-gateway/server.mjs --install
codex -p dsh
```

The installer reads DSH's `settings.yaml`, provider rows in `cordis.patch.yml`,
and its installed model definitions. The native NInfer route reads the loaded
server's actual model/context metadata. Keep that server available during
installation. The generated catalog preserves provider/model identity, context
capacity and supported reasoning levels; summary and legacy routes remain
distinct. Model IDs include the provider prefix to prevent ambiguous routing.

Installation creates `$CODEX_HOME/dsh.config.toml` (normally under `~/.codex`),
and a catalog and local authentication token under
`$CODEX_HOME/dsh-client-gateway`. These paths are shared by the desktop app and
ordinary terminals. AppData is not used because Windows can redirect writes from
the packaged desktop app into a private directory, causing terminal startup to
fail with `Error loading configuration: ... (os error 3)`.
Existing generated files are backed up before
replacement. The base CLI configuration is not edited. The starting model and
default subagent model follow DSH's default at installation time; `/model`
changes the main session selection. Explicit subagent overrides must also use
the provider-prefixed catalog IDs.

The profile's credential helper starts the gateway automatically when needed,
hidden and bound to `127.0.0.1:18021`. The helper receives absolute state and DSH
home paths so launching from a different Windows environment does not change
which credential store it uses. Backend keys remain in DSH's existing store or
environment and are read when a request needs them. They are not copied into
the CLI profile, catalog, repository or logs. The separate local gateway token
is consumed privately by the CLI; do not invoke `--auth-token` in a terminal.

After changing DSH's model configuration, rerun `--install` and reopen the CLI.
This refreshes the catalog and reloads a running gateway's routes; an incomplete
reload preserves the previous routes. Updating gateway JavaScript itself needs
a restart of that gateway process. Its PID is recorded in the local
`gateway.log`; verify its command line before stopping it. The next CLI model
request starts the updated gateway. No DSH or NInfer restart is required.

Use an exact catalog ID to start directly on Bonsai:

```powershell
codex -p dsh -m qwen-3080/bonsai2-heretic -c model_reasoning_effort=low
```

## Behavior and limits

Text, visible reasoning, function calls, custom `apply_patch` input, namespaced
tools and tool results are translated without executing tools in the gateway.
Codex continues to own execution, permissions, hooks and conversation history.
The gateway preserves usage and failure status, propagates cancellation, and
publishes tool calls only after successful generation. Context errors and output
limits remain errors/incomplete responses, rather than being reported as success.

This catalog advertises text input only, including for model names containing
“Vision.” Image/file uploads and provider-hosted search tools are not implemented;
unsupported input is rejected explicitly. CLI web search is disabled in this
profile. This does not prevent separately configured client-side MCP tools.
Stored response IDs and remote Responses compaction are not supported; the
custom-provider CLI sends full history and performs its own local compaction.
The DSH Bonsai loop guard and compactor are agent plugins, so they are not part
of a Codex session; the native NInfer adapter still provides exact prompt counts
and answer-space reservation for its route.

The catalog describes configured routes, not live availability. It does not
load a remote model, switch a NInfer server profile, or substitute another model
after an error. An unloaded legacy Bonsai route needs its matching server; remote
services and their existing credentials must remain available.

## Verification

```powershell
node --test tools/dsh/client-gateway/gateway.test.mjs
```

Eight offline tests cover catalog metadata, instruction/tool replay, explicit
unsupported-input failures, streaming usage, custom and namespaced tool calls,
failed-generation handling, HTTP authentication, disconnect cancellation and
atomic catalog reload behavior. They use local mock adapters and make no model
requests.

On September 27, 2026, the installed CLI loaded all 30 configured DSH routes.
The interactive `/model` picker was opened, and a session switched from Fast
Qwen to Z.ai GLM-4.7. Live text requests succeeded through Fast Qwen, Z.ai,
the loaded Unsloth Qwen route and native NInfer Bonsai. A CLI coding probe using
Fast Qwen created a fixture through `apply_patch`, read it back with one
PowerShell command and reported the independently observed marker. This final
probe used an isolated directory and the user's existing execution policy,
without unrelated hooks/plugins. A normal-profile probe also applied its patch,
but the repository's existing post-tool hook failed afterward.

H200 Qwen's configured proxy returned a backend connection error during its
probe. The route remains selectable; its backend must recover before generation
can work. Other configured models were catalog-validated, not all generated
against. This qualification establishes routing and the exercised coding/tool
workflow, not model quality or full Responses API compatibility.

The Windows path correction was checked against the physical filesystem outside
the packaged app: the original AppData catalog path was missing, while the
replacement catalog and token under `~/.codex` were readable. A fresh CLI request
from `~/Postman` using the repaired profile completed successfully; all eight
offline gateway tests also passed.
