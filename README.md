# NInfer-3080

[![Host checks](https://github.com/ahnafnafee/ninfer-3080/actions/workflows/host-checks.yml/badge.svg)](https://github.com/ahnafnafee/ninfer-3080/actions/workflows/host-checks.yml)
![GPU: RTX 3080 10 GB](https://img.shields.io/badge/GPU-RTX%203080%2010%20GB-76b900)
![Platform: Windows 11](https://img.shields.io/badge/platform-Windows%2011-0078d4)
![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)

**A 27B model at 139 tok/s on a single 10 GB RTX 3080, under native Windows.**

NInfer-3080 is [NInfer](https://github.com/Neroued/ninfer), by way of the consolidated
[NInfer-3090](https://github.com/iamwavecut/ninfer-3090) line, tuned and packaged for one card: a
10 GB GeForce RTX 3080 that also drives a Windows desktop. It keeps the upstream history in full. For
other GPUs and the complete engine reference, see the
[NInfer-3090 README](https://github.com/iamwavecut/ninfer-3090#readme).

| | |
|---|---|
| **Fastest** | 139 tok/s on short answers, 94 tok/s with 61K tokens in context (MTP, 64K window) |
| **Longest** | a 96K window that still decodes at 53 tok/s when nearly full |
| **Prompts** | 1,400 to 1,900 tok/s prefill |
| **Runs** | natively on Windows with MSVC and CUDA 12.8 or 13.x; no WSL |

## Contents

- [Benchmarks](#benchmarks)
- [Quick start](#quick-start)
- [Profiles](#profiles)
- [ninfer-switch](#ninfer-switch)
- [Connecting a client](#connecting-a-client)
- [Building](#building)
- [VRAM on a desktop card](#vram-on-a-desktop-card)
- [What this fork changes](#what-this-fork-changes)
- [Credits and license](#credits-and-license)

## Benchmarks

Every number in this README was measured on one machine:

| | |
|---|---|
| GPU | GeForce RTX 3080 10 GB (68 SMs, 320 W), also driving a 1080p monitor |
| CPU / OS | Core i9-14900K, Windows 11 |
| Toolchain | CUDA 13.1, driver 596.49, MSVC 14.44 |
| Model | [Ternary Bonsai 2 27B Heretic](https://huggingface.co/emiltsoi/Ternary-Bonsai-2-27B-Uncensored-Heretic-NInfer) |
| Sampling | one request at a time, thinking on, `temperature 1.0`, `top_p 0.95`, `top_k 20` |

### Decode and prefill

Tokens per second. "Nearly full" is one request whose prompt fills about 92% of the window.

| Profile | Speculation | KV cache | Window | Short answers | Nearly full | Prefill |
|---|---|---|---:|---:|---:|---:|
| `mtp` | MTP, 3 drafts | `rk2v4-e8` | 65,536 | **139** reasoning, **98** code | **94** at 61,602 | 1,629 |
| `96k` | none | `rk2v4-e8` | 98,304 | **70** | **53** at 91,837 | 1,424 |
| `mtp48` | MTP, 3 drafts | `rk4v4` | 49,152 | 137 reasoning, 94 code | 106 at 28,973 | 1,903 |
| (for comparison) | none | `rk4v4` | 65,536 | 72 | 65 at 28,973 | 1,939 |

### Where the limits are

- **With MTP, VRAM is the limit.** A 76K window is refused at startup (about 120 MB short), so 64K is
  the largest MTP window.
- **Without speculation, speed is the limit.** A 104K window starts, but decode drops to 49.7 tok/s
  at 99,391 tokens, so 96K is the largest window that stays above 50 tok/s when full.
- **The desktop moves the ceiling.** The monitor's share of the card varied from 0.4 to 1.3 GB during
  these runs. At the high end the 64K MTP window does not fit, and `ninfer-switch` starts it at 56K.

### Perplexity (KV cache quality)

The two KV caches the profiles use, scored by `ninfer-perplexity` on the repository's fixed corpus
`ninfer-ppl-1m-v1` in `--quick` mode: one stream from each of four domains, a 4,096-token context and a
2,048-token stride, 261,223 scored tokens per run. Lower is better.

| Domain | Scored tokens | `rk4v4` | `rk2v4-e8` | Change |
|---|---:|---:|---:|---:|
| English reference | 65,328 | 8.034 | 8.308 | +3.4% |
| English long-form | 65,484 | 9.378 | 9.608 | +2.5% |
| Chinese reference | 65,513 | 8.266 | 8.604 | +4.1% |
| NInfer C++/CUDA code | 64,898 | 1.910 | 1.982 | +3.7% |
| **Overall** | **261,223** | **5.887** | **6.088** | **+3.4%** |

| KV cache | Mean NLL | Tokens per GiB of KV | Scoring speed | Run time |
|---|---:|---:|---:|---:|
| `rk4v4` | 1.7728 | 59,900 | 1,108 tok/s | 4.2 min |
| `rk2v4-e8` | 1.8063 | 77,700 | 1,088 tok/s | 4.4 min |

`rk2v4-e8` buys 30% more context for a 3.4% rise in perplexity, spread evenly across domains. Use
`mtp48` (`rk4v4`) when that trade is not worth it. Every window starts from empty state, so this
measures quality within a 4K context; it does not test recall across a long one.

To reproduce, with `--kv-dtype` set to the cache under test:

```powershell
.\build-ninja\apps\ninfer-perplexity.exe models\Ternary-Bonsai-2-27B-Heretic-ninfer.ninfer `
  --corpus eval\corpora\perplexity-1m\manifest.json --quick --kv-dtype rk2v4-e8 --gdn-state-fp16
```

### Other engines on the same card

The same ternary weights in PrismML's llama.cpp fork, and a conventional 4-bit build of Qwen3.8-27B:

| Engine | Model | Window | Short answers | At 28,973 tokens |
|---|---|---:|---:|---:|
| NInfer-3080 (`mtp48`) | Bonsai 2 Heretic | 49,152 | 137 | 106 |
| llama.cpp, PrismML fork | Bonsai 2 Abliterated PQ2_0 | 49,152 | 58 | 44 |
| llama.cpp, PrismML fork | Bonsai 2 Heretic PTQ1_0 | 131,072 | 49 | 36 |
| llama.cpp | Qwen3.8-27B IQ4_XS, feed-forward weights in system RAM | n/a | 5 | n/a |

MTP in llama.cpp made the ternary models slower on this card (58 down to 43-51 tok/s at temperature
1.0); in NInfer it nearly doubles reasoning speed.

## Quick start

1. **Build** (details under [Building](#building)):

   ```powershell
   $env:VCPKG_ROOT = 'C:\src\vcpkg'
   .\scripts\build.ps1
   ```

2. **Get the model** into `models\`, or point `NINFER_MODELS` at its folder:

   ```powershell
   hf download emiltsoi/Ternary-Bonsai-2-27B-Uncensored-Heretic-NInfer --local-dir models
   ```

3. **Serve it.** Add `tools\windows` to `PATH`, then:

   ```text
   ninfer-switch mtp
   ```

The server listens on `http://127.0.0.1:8080/v1` and speaks both the OpenAI and the Anthropic API.

## Profiles

| Profile | Model id | Use it for |
|---|---|---|
| `mtp` | `bonsai2-heretic` | everyday use: fastest, up to a 64K window |
| `96k` | `bonsai2-heretic-96k` | long documents: the largest window that stays above 50 tok/s |
| `mtp48` | `bonsai2-heretic-48k` | higher KV precision (`rk4v4`) at 48K |

Profiles live in [`tools/windows/profiles.json`](tools/windows/profiles.json). The official,
non-abliterated model is
[WaveCut/Ternary-Bonsai-2-27B-NInfer-v3](https://huggingface.co/WaveCut/Ternary-Bonsai-2-27B-NInfer-v3);
point a profile at it to serve it instead.

## ninfer-switch

Start, switch, stop and inspect the server from any terminal. The server runs hidden and outlives the
window that started it.

| Command | Does |
|---|---|
| `ninfer-switch <profile>` | start a profile, stopping whatever runs now |
| `ninfer-switch status` | profile, uptime, port owner, served model, GPU memory, KV capacity, last rate |
| `ninfer-switch logs [-f]` | show or follow the server log |
| `ninfer-switch kill` | stop the server |
| `ninfer-switch start` | start the last profile again |
| `ninfer-switch list` | list the profiles |

- **Short on VRAM?** When NInfer refuses a window, the tool retries 8K smaller (down to 16K) and says
  which window it got.
- **Your own profiles.** A second file named by `NINFER_PROFILES` adds or replaces profiles and can
  set `"port"` and `"models"`, so servers of other engines fit behind the same command.
- **Safe on a shared port.** It only stops server binaries it can start; anything else on the port is
  reported and left alone.

## Connecting a client

Point any OpenAI-compatible client at the endpoint, with the model id of the running profile. The
server needs no key; for clients that insist on one, any placeholder works.

Thinking is on by default. Choose the level per request with `reasoning_effort`:

| `reasoning_effort` | Effect |
|---|---|
| `none` | thinking off |
| `medium` | shorter reasoning |
| `xhigh` | full reasoning, the default |

`low` is accepted, but this checkpoint does not support it and reasons about as long as `xhigh`. For
a client with an off / low / medium / high ladder, send `none` / `medium` / `medium` / `xhigh`; that
mapping also works with llama.cpp, whose Qwen template rejects `high`.

## Building

**Requirements**

- Visual Studio 2022 with "Desktop development with C++" (MSVC 14.4x)
- CUDA 12.8 or newer
- CMake and Ninja
- a [vcpkg](https://github.com/microsoft/vcpkg) checkout

**Steps**

```powershell
git clone https://github.com/microsoft/vcpkg C:\src\vcpkg
C:\src\vcpkg\bootstrap-vcpkg.bat -disableMetrics
$env:VCPKG_ROOT = 'C:\src\vcpkg'
.\scripts\build.ps1
```

The first configure builds FFmpeg and libcurl through vcpkg (9.4 minutes on the machine above, cached
after that). The server lands in `build-ninja\apps\ninfer-serve.exe`. After moving a checkout, delete
`build-ninja\CMakeCache.txt` before rebuilding.

## VRAM on a desktop card

NInfer reserves all of its memory at startup and never grows, but Windows keeps its own allocations on
the same card. When a server on this card was left with about 100 MB free, WDDM moved allocations into
system RAM and decode fell by 30 to 90% with no error at all. Two things help:

- plug the monitor into the motherboard's integrated graphics, which returns up to 1.3 GB to the model
- in the NVIDIA Control Panel, set **CUDA - Sysmem Fallback Policy** to **Prefer No Sysmem Fallback**,
  so an overcommit fails loudly instead

## What this fork changes

| Change | Why |
|---|---|
| `std::countr_zero` in place of `__builtin_ctz` | MSVC has no `__builtin_ctz`, so the Windows build stopped in the NVFP4 divisor epilogue |
| `scripts/build.ps1` finds VS 2022, CUDA 13.x and vcpkg | it never looked in `Program Files` for VS 2022 IDE editions, knew only CUDA 12.8 and 12.9, and passed no vcpkg toolchain |
| Built-in RTX 3080 device profile | the card starts on measured routes instead of calibrating for 76 s on first launch |
| `--kv-headroom-mib` listed in `--help` | `--kv-capacity auto` holds back 1 GiB by default, which a 10 GB card cannot spare |
| `tools/windows/ninfer-switch` | start, switch, stop and inspect profiles on Windows |
| CI on checkout v7, setup-node v7 and Node 24 | the Linux script guard passes again, and nothing runs on the deprecated Node 20 runtime |

## Credits and license

Apache-2.0, as upstream. NInfer is by [Neroued](https://github.com/Neroued/ninfer); the RTX 3090 line
and its contributors are credited in the
[NInfer-3090 README](https://github.com/iamwavecut/ninfer-3090#readme). Ternary Bonsai 2 is by
[PrismML](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf), the Heretic decensoring by
OS-Software, and the NInfer artifact by emiltsoi.
