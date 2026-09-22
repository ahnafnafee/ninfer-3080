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
| context cache | a private-only capture with no committed demand and no shared credit goes straight to that reclamation: the capture search could not produce a plan for it, yet spent its whole 4,096-target budget on the engine thread first (80 ms to 0.5 s before the first token of every new conversation once the private catalog filled) | `resource_manager.h` |
| engine | store exhaustion during placement (`ContextCacheExhausted`, a `std::bad_alloc` naming the store) fails only the request with HTTP 429 and is counted, instead of failing the worker | the reservation sites in `models/qwen3_5/program/{program_impl,graphs,storage/context,transactions/materialization}.cpp`, the catches in `progress_materialization_transaction`, `runtime/engine/engine_core.h` |
| engine | a Paged KV exhaustion names its page numbers; three consecutive exhaustions without a successful admission mark the Engine unavailable for the healthcheck | `core/paged_kv_cache.cpp`, `engine_core.h` |
| serve | a named or single-tool forced `tool_choice` is executed by opening the call in the generation prompt (v0.11 rejects it with `tool_choice_not_supported`) | `models/qwen3_5/frontend/chat_template.cpp` (opener after the generation prompt), `tool_call_parser.{h,cpp}` (seeded decoder), `serve/translate.cpp` (reasoning rule), the three request parsers, `serve/request_log.*` (schema v22) |

The line also carries ternary checkpoints, ported from the earlier v0.10-based ternary work:

| area | behaviour | where it lives in the v0.11 tree |
|---|---|---|
| format | `t2_g128_fp16`: ternary codes as 2-bit two's complement with one binary16 scale per 128 columns, row-split only | `core/weight.h`, `core/weight_view.cpp`, `artifact/formats.cpp`, `tools/artifact/` |
| ops | T2 linear routes (small-T tensor-core kernel, narrow MMA tiles for prefill), composed T2 attention/GDN input projections, `linear_add`/`linear_swiglu`, the T2 full head in `linear_topk` | `ops/linear/t2/`, `ops/wrapper/`, `ops/linear_topk/t2.cu` |
| ops | `hadamard_transform` and `silu_mul_hadamard` | `ops/kernel/hadamard_transform.cuh`, `ops/wrapper/hadamard_transform.cpp` |
| model | Hadamard-rotated Uses (`hadamard_signs` auxiliary): the execution layer rotates the inputs of rotated projections and heads; the residual stream stays primal | `models/qwen3_5/execution/rotation.h`, `load/prepare.cpp`, `model.cpp`, the attention/GDN/FFN/head sites |
| convert | `bonsai2_27b_ternary` builds a Qwen3.8-27B artifact whose text tower comes from PrismML's PQ2_0 GGUF | `tools/convert/ternary.py`, `tools/convert/sources/gguf.py` |

Deliberately not carried: the LRU catalog policy (`--context-cache-policy`), the host-state byte
budget, the context-trace diagnostics and the prefix-cache scenario battery of the previous line.
Measured in production, the branching policy did not help and sometimes hurt; the fixes above are
what actually keeps prefills from being triggered.

## Verification

- `ninfer_resource_manager_test` after every context-cache commit (it builds on a Mac without
  CUDA: Homebrew clang, `-include exception`, the Xcode SDK sysroot).
- `ninfer_tool_call_parser_test`, `ninfer_qwen3_5_frontend_test` (fixture tokenizer), the OpenAI,
  Responses and Anthropic schema tests cover the forced tool call.
- `ninfer_hadamard_transform_test`, `ninfer_linear_t2_a16_test` and the T2 case of
  `ninfer_linear_topk_test` check the ternary ops against FP64 oracles; `tests/convert/test_ternary.py`
  covers the GGUF mapping.
- The full build and `ctest` on an RTX 3090 (sm_86) before the branch is published.

## Deployment

Deployment files are not part of this public line. The production checkout adds them on a
private branch on top of `franken/v0.11`.
