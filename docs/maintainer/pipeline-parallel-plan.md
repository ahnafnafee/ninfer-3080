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
  30,34` sets the counts. Without it `default_stage_layers` (`models/qwen3_5/load.cpp`) sizes each
  layer from the artifact and calls `solve_stage_plan`, an exact memory-balancing solver (maximum
  page groups every stage can hold, then least-full stage), verified against a brute-force
  partition oracle. It works from the bytes free on each device and rough constants for what a
  stage needs beyond weights (context, workspace, graphs; more on rank 0 for the head), and
  ignores per-slot GDN state, so it is a good default rather than exact; if it finds nothing fits
  it deals the layers evenly and lets the Program's planning report the device that runs out.
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

- **Vision** and **DFlash/DFlash2** are refused when the model spans more than one device. DFlash reads
  layer outputs from several depths (its feature taps) into rank 0 buffers, which from a later stage
  are another device's memory; they need to cross the stage boundaries first.
- **Prefill does not overlap stages.** A prefill chunk runs through the stages in turn and the
  engine synchronizes after each chunk, so at any moment one stage is busy. Overlapping stages needs
  micro-chunks inside a chunk (later stages start on micro-chunk 0 while stage 0 runs micro-chunk 1);
  that changes the chunk shapes the kernels see, and its benefit can only be measured on real cards.
- **MTP works** across stages: the replay records and their fold are per state shard, on the shard's
  own device, and the draft layer and head stay on rank 0. **The context cache works**: its
  transactions copy each rank's planes and state shards on that rank's streams.

## Verification

- `ninfer_stage_link_test`: byte integrity across sizes and ring reuse, a two-hop chain, payload
  halves through separate graphs, the slot-reuse dependency read off the graph, and eager use of a
  slot before and after a capture. Three ranks share one card, with the staged protocol forced.
- `ninfer_kv_cache_ranks_test`: the multi-rank pool and block-table copies against the single-rank
  pool as the oracle, in both plane orders.
- `ninfer_stage_plan_test`: the solver against a brute-force partition enumeration.
- `ninfer_kv_capacity_test`: the per-device curve, including a device without KV and errors that
  name the device.
- `ninfer_qwen3_5_stages_real_test` (set `NINFER_TEST_ARTIFACT`): greedy output with `--devices 0,0`
  and `0,0,0` equals `--device 0` byte for byte, since the layers run the same kernels on the same
  data. Rows: graphs and eager, forced pinned-host transport, an uneven split, MTP, and a
  continuation that reuses the context cache. The 27B on the RTX 3090 passes every row.
- `ninfer_qwen3_5_loading_real_test`: the default split covers the model, gives the head stage
  fewer layers, and gives a device with twice the memory more.

Ranks sharing a device cannot show a wrong-device pointer: a stale pointer into "another rank's"
memory still works there. That class of bug needs a real second card.

## Tensor parallelism: status

Not built. What was established, so the next attempt starts from evidence rather than from the plan:

- **Transport** (`tools/tp_probe.cu`, rented boxes, 2026-09-21): neither a 2x RTX 3090 box (one slot
  PCIe 3.0 x4) nor a 2x RTX A4000 box had peer access or NVLink. An all-reduce of one decode column
  (20 KB, FP32) costs 17-33 us whether done by a host-mapped kernel, a staged pinned-host copy or
  NCCL; at 640 KB it is 190-440 us and at 20 MB 5-13 ms. The transports are within about 20% of each
  other at every size, so a first TP should reuse `StageLink`-style staged copies (they already
  capture into graphs) and a local reduce, not write spin-wait kernels.
- **Where it pays.** Projected against this pipeline on two A4000s: 1.6-1.7x at one decode column,
  about 1.5x at four, roughly break-even at sixteen, a loss at thirty-two (MTP3 with eight lanes).
  TP prefill is bound by the link (~840 ms of all-reduce per 1,024-token chunk), so prefill speed
  belongs to the micro-chunk overlap above, not TP. Link width is per machine and decides
  everything; run the probe on the target box first.
- **Smallest useful first step: MLP-only TP on the dense 27B** (about 64% of the weight bytes;
  projected ~1.35x decode). Each rank keeps attention and GDN whole and replicated (weights and KV),
  and takes half the MLP: `gate_up` split by rows (gate rows then up rows, so the SwiGLU pairing
  holds) and `down` split along K. A rank's down projection can reuse `linear_add`: rank 0 adds its
  partial into the real residual, the other rank into zeros, and one BF16 exchange plus an add gives
  every rank the same new residual (one extra rounding against the single-device path, so the oracle
  is a tolerance on logits and perplexity, not byte equality).
- **What it needs before any speed can be claimed:** loader-time slicing of row-split quantized
  parents (contiguous row ranges for `gate_up`, per-plane column ranges for `down`, group-aligned);
  the `linear_swiglu` Q4 kernel at 17,408 rows and `linear_add` Q5 at K=8,704, each registered for
  those exact shapes and qualified against the FP64 oracle and tuned on the target card (routes
  inherited from the full-width shapes are hypotheses); per-rank replicas of the KV pool and state
  pool with fan-out in the transactions; and the exchange itself. Expect weeks, and the decisive
  measurement is the probe on the machine that will run it.
