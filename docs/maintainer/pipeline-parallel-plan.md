# Multi-GPU: layer pipeline stages

`--devices A,B,...` splits the model's layers into one pipeline stage per GPU. Each stage owns its
layers whole: weights, the KV cache of its attention layers, the recurrent state of its GDN layers,
and the scratch it runs in. The point is memory: a model that does not fit one card, or a context
that does not, spreads across several with every card's memory usable for KV.

This replaces the earlier expert-offload split, which moved only each layer's MLP to the second card
and kept everything stateful on the first. That design capped KV room at whatever the first card had
left; whole-layer stages have no such asymmetry, and they cross a device boundary once per stage
rather than twice per layer.

## What it is and is not

**It is a memory feature.** The stages run in sequence: while one computes, the others wait.
Decode is weight-bandwidth-bound and each stage reads only its own weights, so a single stream is
about as fast as one GPU, minus the boundary hops. Splitting the batch into micro-batches would make
each stage read its weights once per micro-batch and win nothing.

**It is not tensor parallelism.** No layer is divided across devices, so no collective runs inside a
layer.

**Multi-GPU is a Linux feature.** Repeating one device id (`--devices 0,0`) exercises the whole
stage path on a single card and is accepted everywhere, which is how the path is tested without a
second GPU. Distinct ids are refused on Windows.

## How it works

- **`StagePlan`** (`core/stage_plan.h`) says which contiguous layers each stage owns. `--stage-layers
  30,34` sets the counts; without it the layers are split evenly. `solve_stage_plan` is an exact
  memory-balancing solver (maximum page groups every stage can hold, then least-full stage) that the
  engine does not call yet; it is verified against a brute-force partition oracle.
- **Placement.** `bind_text` places every layer's weights on its stage's device
  (`Bindings::place`, recorded for rank 0 as well so an object shared between a rank-0 layer and a
  later stage's layer is refused as a conflicting placement). The embedding, final norm and head stay
  on rank 0.
- **The head stays on rank 0.** The last stage sends the residual back; rank 0 then runs the final
  norm, head and sampling exactly as on one GPU. That costs one extra ~20 us hop per forward pass and
  keeps the round buffers, prefill buffers and every other head-side structure where they are.
- **Per-rank state.** `DeviceKVPagePool` is one page-group allocator over planes bound to several
  ranks' backings, so admission, prefix reuse and the context-cache policy never see how many devices
  hold the cache. Block tables are one host shadow with a device copy per attention-bearing rank
  (`KVExecutionTablePool`); GDN state is one `LinearAttentionStatePool` shard per stage
  (`StateImageDevicePool`); the host image of a state keeps the whole model's layers at their global
  offsets. Data movement takes `RankStreams`, so each plane's copy goes on its own rank's stream, and a
  single stream given to a multi-rank object fails instead of using the wrong device.
- **Capacity.** Each further device has its own affine reservation curve
  (`RankCapacityCurve`); the resolved page-group count is the smallest any device allows, and a
  failure names the device that could not hold the plan. Ranks sharing one physical device split its
  free memory.
- **`StageLink`** (`core/stage_link.h`) moves the residual between stages, and the small control
  tensors (positions, KV rows, slot indices, valid columns) the layers read from rank 0, through a
  ring of pinned-host slots. The copies are ordinary graph nodes. Decode captures the whole pass into
  one multi-device graph, so `StageLink` tracks, per event, which capture made its last record: a wait
  inside a capture may only target an event recorded in that capture, and an eager wait may not target
  one whose last record was in a capture (`cudaErrorInvalidValue`, even after the graph has
  launched).
- **`TextContext::run_staged`** packs the control tensors on rank 0, sends each stage its block up
  front, runs stage 0, and for each later stage rebinds the context's device, stream and workspace
  (`ScopedDeviceRank`, `ScopedArenaRank`), receives the residual and control block, runs the stage's
  layers and passes the residual on.

## What is not covered yet

Refused with a clear message when the model spans more than one device: speculative decoding (MTP,
DFlash), vision, and the context cache (`--no-prefix-reuse`). The context cache's transactions copy
state on rank 0's streams; speculative decoding's replay records and draft state live on rank 0
and are read by kernels that would run on later stages.

## Verification

- `ninfer_stage_link_test`: byte integrity across sizes and ring reuse, a two-hop chain, payload
  halves through separate graphs, the slot-reuse dependency read off the graph, and eager use of a
  slot before and after a capture. Three ranks share one card, with the staged protocol forced.
- `ninfer_kv_cache_ranks_test`: the multi-rank pool and block-table copies against the single-rank
  pool as the oracle, in both plane orders.
- `ninfer_stage_plan_test`: the solver against a brute-force partition enumeration.
- `ninfer_kv_capacity_test`: the per-device curve, including a device without KV and errors that
  name the device.
- End to end, greedy output with `--devices 0,0` must equal `--device 0` byte for byte, since the
  layers run the same kernels on the same data.

Ranks sharing a device cannot show a wrong-device pointer: a stale pointer into "another rank's"
memory still works there. That class of bug needs a real second card.
