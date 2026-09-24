# NInfer-3090 consolidated

One RTX 3090 line of [NInfer](https://github.com/Neroued/ninfer), consolidated from the forks that
carry it and extended with this repository's own work. The base is the `master` of
[ashalliants/ninfer-3090](https://github.com/ashalliants/ninfer-3090) (v0.11.0 and the multi-GPU
pipeline stages), which continues [Don-Chad/ninfer-3090](https://github.com/Don-Chad/ninfer-3090).
On top of it come patches from [TertiumOrganum1/ninfer-3090](https://github.com/TertiumOrganum1/ninfer-3090),
ideas backported from [UDPSendToFailed/ninfer-4090](https://github.com/UDPSendToFailed/ninfer-4090),
pull requests to [Neroued/ninfer](https://github.com/Neroued/ninfer), and work from
[IMGillusion/ninfer-disk-kv](https://github.com/IMGillusion/ninfer-disk-kv),
[MirkoCovizzi/ninfer-rtx5090-mobile](https://github.com/MirkoCovizzi/ninfer-rtx5090-mobile),
[Wallawalla47/ninfer-custom](https://github.com/Wallawalla47/ninfer-custom) and
[tmark00/ninfer](https://github.com/tmark00/ninfer).
Everything the engine does beyond the list below (building, packages, serving APIs, supported models,
flags, measurements) is described in the original READMEs of
**[NInfer-3090](https://github.com/ashalliants/ninfer-3090#readme)** and
**[NInfer-4090](https://github.com/UDPSendToFailed/ninfer-4090#readme)**. This page covers only what
this line adds and how to run it. New features that change numbers or serving behaviour are
opt-in; without their flag or build option the engine behaves as its base. The exceptions are bug
fixes and TertiumOrganum1's ternary prefill tile, which is on by default
(`NINFER_T2_A8_TILE=off` restores the kernel it replaces).

## What this line adds

- **Ternary Bonsai 2 27B.** PrismML's [ternary Qwen3.8-27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
  runs from `t2_g128_fp16` weights: 2.125 bits per weight, imported from the PQ2_0 GGUF without
  rounding, with the checkpoint's Hadamard rotations fused into the norms and gates that produce
  each projection input. The token table and the heads stay ternary, and the converter recipe
  `bonsai2_27b_ternary` adds ProCreations' Bonsai-trained MTP head and DFlash2 adapter and an exact
  proposal head.
- **Integer activations for ternary projections.** Decode, speculative verification and prompts up
  to 192 tokens use a small-T kernel over s8 activations. Longer prompts use the int8-activation
  GEMM, which pads a ragged prompt to its cheapest tile. The output, draft and proposal heads take
  the same route.
- **RTX 3090 tuning.** The rotating producers run four warps per 1024-point transform. The small-T
  attention over `rk8v4` launches its splits in whole waves of 82 SMs. The GDN record stages its
  window in shared memory. The DFlash2 adapter of a ternary target runs in Q4.
- **DFlash2 with Vision in overlay.** An image encode can borrow the drafter's memory, so DFlash2,
  Vision and the model's whole 262,144-token window fit on one 24 GB card.
- **Serving fixes.**
  - A forced `tool_choice` opens the named call in the generation prompt, and the template's
    default thinking yields to it.
  - A context-cache store that cannot place a request fails only that request (HTTP 429) instead of
    the engine.
  - A Paged KV exhaustion names its page numbers, and three in a row mark the engine unhealthy.
  - Several context-cache fixes keep long agent sessions from re-prefilling: private reclamation,
    the demand window, and the capture search for a zero-value candidate.
- **Build.** Tests build against CUDA 13's `cudaGraphGetEdges`.

Taken from [TertiumOrganum1's fork](https://github.com/TertiumOrganum1/ninfer-3090):

- **`rk4v4-e8` KV cache.** Keys are rotated as in `rk8v4` and snapped per octet to the E8 lattice
  in int4, and values keep `rk8v4`'s int4 plane. That is 280 bytes per token and KV head against
  408. On Ternary Bonsai 2 the whole 262,144-token window takes 2.0 GiB less, two lanes get a
  whole window each (524,288 tokens, where `rk8v4` fits 519,744) with 5.7 GiB to spare, the three
  codes planted at 131K and 250K are still found, and quick-corpus perplexity moves from 5.631 to
  5.650.
- **A 128x64 int8 tile for the ternary prefill route.** Activations are quantised per token and
  128-column group, so the int32 sum runs over a whole weight group. On Ternary Bonsai 2 prefill
  runs 33% faster at 8K, 21% at 32K and 15% at 64K than with the kernel it replaces
  (`NINFER_T2_A8_TILE=off`), and quick-corpus perplexity stays at 5.631.
- **Tool calls.** A malformed tool-call region is recovered as far as it reads, instead of leaking
  its markup into the answer.
- **Shared captures.** A shared-prefix capture whose replacement releases less than was assessed is
  abandoned. Before, the engine failed for good and answered 503 until a restart.
- **Build.** `sm_120a` (RTX 50-series) builds on the `mma.sync` compatibility path.

Taken from [NInfer-4090](https://github.com/UDPSendToFailed/ninfer-4090), re-implemented here:

- **Whole-program CUDA build.** The core and ops archives no longer build relocatable device code,
  so ptxas keeps shuffles from a computed lane inline and pipelines loads across loops: a fifth
  fewer kernels need a stack frame, and the server binary grows by a quarter.
- **Shared-memory scale reads.** The INT8-family attention kernels read their query, key and value
  scales from shared memory instead of shuffling them from a computed lane. With the whole-program
  build, the `rk8v4` attention of a verify step takes 9 to 21% less time, so on Ternary Bonsai 2
  an MTP step at 64K of context is 6.6% shorter and prefill 2 to 7% faster from 8K up, with the
  same answers.
- **`TCP_NODELAY`** on the server socket, so a streamed token leaves as soon as it is written.
- **Sigmoid, SiLU and softplus on the SFU, opt-in.** A build with `-DNINFER_SFU_SIGMOID_SILU=ON`
  evaluates sigmoid and SiLU with `ex2.approx` and a correctly rounded reciprocal instead of `expf`
  and a divide: on Ternary Bonsai 2 prefill measured 2% faster from 8K up, quick-corpus perplexity
  moves from 5.6306 to 5.6309, and MTP decode is unchanged. `-DNINFER_SFU_SOFTPLUS=ON` evaluates
  the GDN decay gate's softplus the same way, switching to a log1p series where e^x is below 1/16
  so the slow decays of long-memory heads keep their precision (quick-corpus perplexity 5.6302).
- **Keys past 262,144.** The small-T attention kernels read each page's physical index from the
  block table once a split spans more than the 64 page IDs it stages, and the visible-key limit
  rises to 1,048,576.
- **Four times the native window, with YaRN.** `--max-context` accepts up to 1,048,576 tokens on
  the 262,144-token models. Past the native window positions run plain RoPE, or YaRN with
  `--rope-yarn`, at Qwen's documented factor (`--max-context` / 262,144) and computed as Hugging
  Face and vLLM do. On Ternary Bonsai 2 with `rk2v4-e8`, a needle test (three codes at 33, 66 and
  90% of a prose document) finds all three at 500,000 tokens without the flag and two of three
  with it at 131,072, 500,000 and 1,000,000 tokens, so YaRN stays off unless plain RoPE stops
  answering.
- **`rk2v4-e8` KV cache.** Each 8-dimension block of a rotated, G64-scaled key is stored in two
  bytes: the nearest of E8's 240 roots, and a byte holding a 4-bit log-radius and a signed
  residual axis. That is 216 bytes per token and KV head, against 280 for `rk4v4-e8` and 408 for
  `rk8v4`, and the one format that holds 1,048,576 tokens beside Ternary Bonsai 2 on a 24 GB card
  (2.8 GiB spare without speculation, 1.3 GiB with MTP). Two lanes over the 262,144-token window
  leave 7.7 GiB spare. The price is quality: quick-corpus perplexity rises from 5.631 to 5.820
  (`rk4v4-e8`: 5.651), and DFlash2 accepts fewer drafts (51.8% against 54.4%), so decode is 4%
  slower. The three planted codes are all found at 131,072 and 250,000 tokens, and at 500,000 in
  a 1,048,576-token window.
- **D3D12-resident arenas on Windows.** A build with `-DNINFER_D3D12_RESIDENCY=ON` offers
  `--wddm-evictable-budget`: the device arenas come from a shared D3D12 heap made resident at the
  highest priority and imported into CUDA, and the KV cache is sized as if WDDM will evict other
  allocations. Untested here, since this line has no Windows machine; the code only passes a
  MinGW syntax check.

Further 4090 ideas: a server default reasoning effort, MTP draft windows up to 15, `/metrics`,
`/slots` and `/props`, a WebUI compiled in from `NINFER_WEBUI_DIR`, output limits bounded only by
the context, the block sampler's candidates in shared memory, an opt-in bf16 residual add
(`-DNINFER_BF16_RESIDUAL_ADD=ON`), vector stores in the chunked GDN prefill, and bounded split
compilation with ptxas reports as build options.

From other forks:

- **A disk tier under the Host tier** ([IMGillusion](https://github.com/IMGillusion/ninfer-disk-kv)).
  `--disk-kv-path DIR` writes an evicted conversation's KV pages and state images to CRC-checked,
  LRU files keyed by the prompt digest, which survive restarts; with `--disk-kv-restore` a new
  request whose prompt starts with a stored prefix is seeded from disk and prefills only the rest.
  On Ternary Bonsai 2 with MTP, a 17,444-token prompt evicted by two others comes back from disk
  in 1.2 s instead of 9.6 s, and in 1.0 s after a restart, with the same answer; DFlash2 and no
  speculation restore the same way. On Windows, a build with `-DNINFER_DIRECTSTORAGE=ON` reads
  those restores through DirectStorage (`--disk-kv-directstorage`; untested, as the D3D12
  option).
- **Adaptive MTP** ([Mirko Covizzi](https://github.com/MirkoCovizzi/ninfer-rtx5090-mobile)).
  `--adaptive-mtp` lets each round verify 3..K of the K drafts, the width that measured draft
  survival and measured round cost favour, with CUDA Graphs for each width. On Ternary Bonsai 2
  on an RTX 3090 with K=5 it verified five drafts in 59% of rounds, four in 23% and three in 18%,
  and did not beat the card's fixed K=3: 200 against 204 tok/s on short prompts, 150 against 163
  at 8K. The graphs for every width cost memory too, so Huihui with a 198,400-token cache no
  longer fits a 24 GB card with it. A near-tied token can come out differently at another width,
  as it does between two fixed windows.
- **A fast INT8 prompt-attention kernel** ([Wallawalla47](https://github.com/Wallawalla47/ninfer-custom)).
  `--fast-prefill-kernel` prefills an `int8` KV cache with FP16 PV accumulation per 64-key tile
  and rounds the prefill chunk to whole attention waves (+15% at 64K and +25% at 128K on an
  RTX 5090, for 0.08% perplexity). On an RTX 3090 with Huihui it is 5% slower at 16K and 3%
  faster at 64K, and quick-corpus perplexity at 32K moves from 4.1253 to 4.1280, so it stays off
  unless the context is long.
- **Agent-harness tool calls.** `<function name=...>`, `<invoke name=...>`, `<function_calls>` and
  `<param name=...>` are read as tool calls (upstream PR #300 by pkochubey, via Wallawalla47), next
  to the Qwen form, and go through the same recovery pass.
- **Structured output** through xgrammar, speculative decoding included, opt-in with
  `--structured-output` (upstream PR #294 by Andrey Shvartsman).
- **First-token log probabilities.** With `--first-token-logprobs`, a Chat Completions request may ask
  for `top_logprobs` and gets the first generated token's log probability with its alternatives
  (IMGillusion).
- **Rolling retention.** `--context-cache-policy rolling` lets one long conversation keep rolling
  its cached frontier forward (IMGillusion).
- **NVFP4 expert banks on Blackwell** (upstream PRs #286-#290 by Mykhailo Dementii). The published
  Qwen3.6-35B-A3B NVFP4 checkpoint converts with `--recipe qwen3_6_35b_a3b_nvfp4` and runs on a
  native build (`-DCMAKE_CUDA_ARCHITECTURES=120a -DNINFER_SM120_NATIVE=ON`): on an RTX 5090 the
  20.6 GB text artifact prefilled 27,663 tok/s at 4K and decoded 397 tok/s. Its prefill
  quantizes activations to four bits for W4A4, which only Blackwell has, so builds on the
  compatibility path (sm_8x, and sm_120a without `NINFER_SM120_NATIVE`) refuse the banks.
- **Engine and serving fixes**: out-of-memory recovery of the worker, `--kv-headroom-mib`,
  `--cuda-graph-allowance-mib`, `--thinking-budget-message` (Wallawalla47, Gideon Zenz); the WebUI's
  MCP traffic relayed behind `--webui-mcp-proxy`, E8 root codes decoded from tables and an SM-count
  RMSNorm cutoff ([tmark00](https://github.com/tmark00/ninfer)); openable server URLs and CORS
  preflight echoes (pelebel, natpate); and the upstream pull requests listed in the map, among them
  GGUF as a conversion source (giveen), a Q6 recipe (bingchengcc), sparse-MoE, NVFP4 and
  attention-epilogue tuning (Mykhailo Dementii, Duncan Betts, MOVIBALE), quoted-marker and
  duplicate-parameter tool-call fixes (Fedor Suchkov, adubkov) and Copilot tool shapes (Damian
  Sromek).

The [maintainer map](docs/maintainer/consolidated-line.md) lists each change with the files it
touches and the tests that cover it.

## Ternary Bonsai 2 27B on one RTX 3090

One request at a time, at the card's full 350 W, with a 198,400-token `rk8v4` window and Vision
loaded in overlay. Decode is each suite's generated tokens over its decode time, and MTP drafts
through the proposal head:

| profile | decode, short chat | decode, GSM8K answers | tokens per step | VRAM in use |
|---|---:|---:|---:|---:|
| no speculation | 91.4 tok/s | 90.5 tok/s | 1 | 12.6 GiB |
| MTP, three drafts | 199.1 tok/s | 212.2 tok/s | 3.1 | 13.5 GiB |
| DFlash2, seven drafts | 237.2 tok/s | 295.1 tok/s | 4.7 | 14.5 GiB |

- **Prefill:** 1,730 tok/s on a 1,000-token prompt, 1,794 at 8K, 1,505 at 32K and 1,234 at 64K.
- **MTP decode at depth:** 176 tok/s at 8K of context, 137 at 32K and 116 at 64K.
- **Long context:** three codes planted in a document are all found at 32K, 131K and 250K tokens.
- **Quality:** a fixed 1,179-item slice scores 84.4% under MTP and 84.7% under DFlash2. GSM8K alone
  scores 95.5 and 96.5, against 94.5 for llama.cpp on the same GGUF. Perplexity is 5.630.

The [model card](https://huggingface.co/WaveCut/Ternary-Bonsai-2-27B-NInfer-v3) has the device
memory of every profile and the concurrency numbers.

## Running

Download an artifact from the table below and point `ninfer-serve` at it. The server speaks the
OpenAI and Anthropic APIs on `127.0.0.1:8080` by default.

Bonsai 2 27B, the fastest single stream (DFlash2 with seven drafts, Vision in overlay):

```bash
ninfer-serve Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b \
  --max-context 198400 --kv-capacity 198400 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec dflash2 --draft-tokens 7 \
  --vision --vision-residency overlay --vision-max-merged 12288
```

Bonsai 2 27B with MTP drafting through the proposal head (0.16 GiB more, 4% faster than MTP alone):

```bash
ninfer-serve Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b \
  --max-context 198400 --kv-capacity 198400 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --vision --vision-residency overlay --vision-max-merged 12288
```

Bonsai 2 27B over the whole 262,144-token window, with the engine sizing the cache. Two lanes
(`--max-concurrency 2`) share 519,744 tokens:

```bash
ninfer-serve Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b \
  --max-context 262144 --kv-capacity auto --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec dflash2 --draft-tokens 7 \
  --vision --vision-residency overlay --vision-max-merged 12288
```

Bonsai 2 27B with adaptive MTP over up to five drafts, and a 64 GiB disk tier that keeps evicted
conversations for later requests and across restarts:

```bash
ninfer-serve Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b \
  --max-context 198400 --kv-capacity 198400 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec mtp --draft-tokens 5 --lm-head-draft --adaptive-mtp \
  --disk-kv-path /var/cache/ninfer --disk-kv-gib 64 --disk-kv-restore
```

Qwen3.8-27B (the abliterated Huihui artifact), MTP with the proposal head and Vision:

```bash
ninfer-serve Huihui-Qwen3.8-27B-abliterated-ninfer-v3.ninfer --model-id qwen3.8-27b \
  --max-context 198400 --kv-capacity 198400 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --vision --vision-residency overlay --vision-max-merged 12288
```

## Artifacts

| model | artifact | notes |
|---|---|---|
| Ternary Bonsai 2 27B | [WaveCut/Ternary-Bonsai-2-27B-NInfer-v3](https://huggingface.co/WaveCut/Ternary-Bonsai-2-27B-NInfer-v3) | 8.87 GiB. Ternary text tower, token table and head, Vision, Bonsai-trained MTP head and DFlash2 adapter, and an exact proposal head. Runs only on this line. |
| Qwen3.8-27B, abliterated | [WaveCut/Huihui-Qwen3.8-27B-abliterated-NInfer-v3](https://huggingface.co/WaveCut/Huihui-Qwen3.8-27B-abliterated-NInfer-v3) | 19.03 GiB, official `qwen3_8_27b` recipe with MTP, DFlash2 and a proposal head |

The official NInfer artifacts listed in the original READMEs load here too.
[Weight conversion](docs/weight-conversion.md#ternary-bonsai-2-27b) shows how the Bonsai artifact
is built.

## Building

Linux with CUDA 13.1 for the RTX 3090 (`sm_86`):

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --target ninfer-serve
```

The opt-in build options are listed in the
[Linux build guide](docs/rtx-3090-linux.md#build-options). Windows builds, release packages, tests
and benchmarks work as in the [NInfer-3090 README](https://github.com/ashalliants/ninfer-3090#readme).

## License

Apache-2.0, as upstream. The Bonsai artifact's weights come from PrismML, ProCreations and Qwen,
all Apache-2.0; its card lists the notices.
