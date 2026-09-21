# The `franken/v0.11` line

`franken/v0.11` is the ashalliants `ninfer-3090` v0.11.0 release (317ddea7) plus the behaviours
this fork needs in production, re-implemented on the v0.11 tree rather than cherry-picked from
the earlier v0.10-based line. Each commit stands on its own and carries its rationale; this note
is the map.

## What the line carries over v0.11.0

| area | behaviour | where it lives in the v0.11 tree |
|---|---|---|
| context cache | a private capture that cannot be placed reclaims the oldest eligible private resident (publication order), so every conversation regains reuse from its second turn | `runtime/engine/context_cache/resource_manager.h`, `shared_capture_planner.h`, the capture gate in `models/qwen3_5/program/transactions/capture.cpp` |
| context cache | an infeasible shared capture takes the same reclamation route, so the shared reuse frontier keeps advancing under pressure | `resource_manager.h` |
| context cache | replacement scenarios are offered only when no catalog slot is vacant, so an exact repeat does not displace the leading-instruction prefix the next question needs | `resource_manager.h` |
| context cache | the demand window behind the committed mask holds 64 requests, beyond the replay distance of a workload that replays a few dozen prefixes twice | `context_portfolio_value.h`, `materialization_planner.h`, `resource_manager.h`, `shared_capture_planner.h` |
| engine | store exhaustion during placement (`ContextCacheExhausted`, a `std::bad_alloc` naming the store) fails only the request with HTTP 429 and is counted, instead of failing the worker | the reservation sites in `models/qwen3_5/program/{program_impl,graphs,storage/context,transactions/materialization}.cpp`, the catches in `progress_materialization_transaction`, `runtime/engine/engine_core.h` |
| engine | a Paged KV exhaustion names its page numbers; three consecutive exhaustions without a successful admission mark the Engine unavailable for the healthcheck | `core/paged_kv_cache.cpp`, `engine_core.h` |
| serve | a named or single-tool forced `tool_choice` is executed by opening the call in the generation prompt (v0.11 rejects it with `tool_choice_not_supported`) | `models/qwen3_5/frontend/chat_template.cpp` (opener after the generation prompt), `tool_call_parser.{h,cpp}` (seeded decoder), `serve/translate.cpp` (reasoning rule), the three request parsers, `serve/request_log.*` (schema v22) |

Deliberately not carried: the LRU catalog policy (`--context-cache-policy`), the host-state byte
budget, the context-trace diagnostics and the prefix-cache scenario battery of the previous line.
Measured in production, the branching policy did not help and sometimes hurt; the fixes above are
what actually keeps prefills from being triggered.

## Verification

- `ninfer_resource_manager_test` after every context-cache commit (it builds on a Mac without
  CUDA: Homebrew clang, `-include exception`, the Xcode SDK sysroot).
- `ninfer_tool_call_parser_test`, `ninfer_qwen3_5_frontend_test` (fixture tokenizer), the OpenAI,
  Responses and Anthropic schema tests cover the forced tool call.
- The full build and `ctest` on an RTX 3090 (sm_86) before the branch is published.

## Deployment

Deployment files are not part of this public line. The production checkout adds them on a
private branch on top of `franken/v0.11`.
