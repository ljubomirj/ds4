# l26f — Ling-2.6-Flash Narrow-Metal Inference Engine

`l26f` is a small native Metal inference engine for **Ling-2.6-Flash**
(Inclusion AI), a 104B-parameter 256-expert MoE model with a hybrid
GLA/MLA architecture. It does for Ling-2.6-Flash on Apple Silicon what
[ds4](https://github.com/antirez/ds4) does for DeepSeek-V4-Flash: a
deliberately narrow, single-model, Metal-native inference path with no
generic GGUF runner, no wrapper, and no framework.

The engine is a fork/branch of `ds4.c`, adapted for Ling-2.6-Flash's
different architecture: 28 GLA (Gated Linear Attention) recurrent layers
+ 4 MLA (Multi-head Latent Attention) layers, a 256-expert MoE FFN with
8 active experts per token, and a custom IQ4_NL-quantized GGUF format
using the `bailing_hybrid` (BailingMoeV2_5) architecture.

This project would not exist without **ds4** and **llama.cpp/GGML** — see
the acknowledgements section below.

## Why Ling-2.6-Flash?

Ling-2.6-Flash is a 256-expert MoE model with 7.4B active parameters per
token. It uses a hybrid recurrent architecture (GLA + MLA) that avoids the
KV cache scaling problems of traditional transformers while maintaining
high quality. Key properties:

1. **Fast inference**: Only 7.4B active parameters out of 104B total, with
   efficient Metal GLA kernels. Achieves ~30-40 tok/s generation on M3 Max
   with IQ4_NL quantization.
2. **Hybrid architecture**: GLA layers use a compressed recurrent state
   (128-dim per layer) instead of a full KV cache. MLA layers use
   low-rank KV compression (512-dim latent space). This means no growing
   KV cache for most layers — only 4 MLA layers have a traditional KV
   cache, and even that is heavily compressed.
3. **MoE quality at dense-model speed**: 256 experts with 8 active per
   token gives the capacity of a 104B model at the compute cost of a 7.4B
   one.
4. **No "thinking" mode needed**: Unlike DeepSeek models, Ling-2.6-Flash
   produces direct answers without a separate reasoning token stream.
   This makes it simpler to integrate as a coding agent backend.
5. **Strong on coding and reasoning**: Trained on a diverse corpus with
   high-quality code and reasoning data, it performs competitively with
   much larger models on benchmarks.

## What makes this project different from ds4

Ling-2.6-Flash is **not** a DeepSeek architecture. The engine was adapted
from ds4 by replacing:

| Component | ds4 (DeepSeek-V4-Flash) | l26f (Ling-2.6-Flash) |
|---|---|---|
| Attention | Multi-head Latent Attention (MLA) all layers | 28 GLA layers + 4 MLA layers |
| KV state | Compressed KV per layer | Recurrent state per GLA layer; compressed KV per MLA layer |
| MoE | 1 shared expert + routed experts | 256 routed experts in 8 groups, no shared expert |
| Expert selection | Top-K routing | Grouped Top-1 routing (8 groups × 1 expert = 8 active) |
| Quantization | Mixed IQ2_XXS/Q2_K | IQ4_NL (4-bit) via MLX DWQ |
| Context | 1M token window (disk KV cache) | 128K token native; recurrent GLA needs no KV beyond state |
| Hardware target | M3 Ultra 512GB / M3 Max 128GB | M3 Max 64GB+ (58 GiB model fits in RAM) |

## Project Status

**Current**: All 32 layers run end-to-end on Metal. GLA kernel correct,
verified against known-good outputs. MoE routing works with correct expert
group selection and per-type byte strides. MLA runs on CPU (GPU path
pending). Tokenizer loads and decodes. End-to-end text generation works
at ~6-7 tok/s.

**Under development**: GPU-accelerated MLA decode, sampling (temperature,
top-k, top-p), prefill optimization, benchmarking, server API.

See `docs/` for detailed status journal.

## Model Weights and GGUF

The GGUF file used by this engine (also available on HuggingFace):
[`Ling-2.6-flash-IQ4_NL-quality-bailing_hybrid-20260508-LJ.gguf`](https://huggingface.co/ljupco/Ling-2.6-flash-GGUF) (58 GiB).

### How the GGUF was created

The GGUF was **converted from the F16 model**, not from the 4-bit AWQ
quantization. The conversion chain:

1. The original model: [inclusionAI/Ling-2.6-flash](https://huggingface.co/inclusionAI/Ling-2.6-flash)
   on HuggingFace (F16 weights, `bailing_hybrid` architecture).
2. Converted to MLX 4-bit DWQ by `mlx-community`:
   [mlx-community/Ling-2.6-flash-mlx-4bit-DWQ](https://huggingface.co/mlx-community/Ling-2.6-flash-mlx-4bit-DWQ)
3. A working [llama.cpp fork](https://github.com/ljubomirj/llama.cpp/tree/LJ-Ling-2.6-flash-r2)
   (branch `LJ-Ling-2.6-flash-r2`) already supports Ling-2.6-flash as a
   general GGUF runner, confirming the GGUF format works for this model.
   The fork's [HF model card](https://huggingface.co/inclusionAI/Ling-2.6-flash/discussions/8#69fcf2e6e446ab24d7810f64)
   has build instructions, CLI/server examples, and benchmarks.
4. The l26f GGUF was generated via the llama.cpp-based conversion path
   from the F16 weights, producing a model-specific GGUF for this engine.

This engine only works with this specific GGUF layout. It is not a general
GGUF loader.

### The MLX 4-bit checkpoint and the `bailing_hybrid` architecture

The 4-bit AWQ model was initially used to run inference via
[omlx](https://github.com/jundot/omlx), which is an optimized MLX inference
server for Apple Silicon. However, the `bailing_hybrid` architecture does
**not** exist in upstream `mlx-lm` — the open PR
[ml-explore/mlx-lm#1227](https://github.com/ml-explore/mlx-lm/pull/1227)
(adds the `bailing_hybrid` model architecture) is still pending review.

The reference implementation of the model architecture is
[`bailing_hybrid.py`](https://github.com/ml-explore/mlx-lm/pull/1227)
from PR #1227. A copy was committed to the llama.cpp fork at
[`docs/bailing_hybrid.py`](https://github.com/ljubomirj/llama.cpp/blob/LJ-Ling-2.6-flash-r2/docs/bailing_hybrid.py)
and was the primary reference for porting the architecture to both
llama.cpp and this l26f Metal engine.

The initial working setup used a fork of `mlx-lm` with PR #1227, checked
out by Ivan Fioravanti at `~/llama.cpp/pr-1227-ivan/`. One gotcha:
running `uv pip install -e .` in the omlx repo pulls `mlx-lm` from the
git commit pinned in omlx's `pyproject.toml`, which overwrites your local
editable install of the PR #1227 fork. You must re-install the fork after
any omlx dependency refresh.

### GLA slope fix

The upstream model had an off-by-one bug in the GLA decay slope:
`self.layer_idx - 1` was used instead of `self.layer_idx` in the
layer-dependent decay scaling (see [upstream fix](https://huggingface.co/inclusionAI/Ling-2.6-flash/commit/7c60792051a885a3f14a75576f01f7f5cb6a08ff)).
This caused incorrect decay rates, with the most severe effect on layer 0
(which got a negative slope). The llama.cpp `-r2` fork and l26f use the
correct formula: `layer_factor = 1.0 - il / (n_layer - 1) + 1e-5`.

### MTP (Multi-Token Prediction)

Ling-2.6-flash has only 1 MTP head (`nextn_predict_layers=1`), limiting
speculative decoding to 1 draft token per verification pass. With the
MTP head on CPU, the extra draft overhead exceeds any trunk pass savings.
The MTP path is implemented in the `-r2` fork but disabled by default.
Models with multiple MTP heads (e.g. DeepSeek-V3 with 3 heads) would
benefit more. l26f does not implement MTP.

## llama.cpp Benchmarks (Reference, M2 Max)

These are from the `LJ-Ling-2.6-flash-r2` llama.cpp fork running the IQ4_NL
GGUF on a MacBook Pro M2 Max (96 GB). The fork is a **full llama.cpp with
`bailing_hybrid` support**, not a narrow engine — useful as a correctness
and performance baseline.

```
|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|   512 |    128 |    1 |    640 |    1.169 |   437.96 |    2.739 |    46.73 |    3.908 |   163.75 |
|  1024 |    128 |    1 |   1152 |    2.855 |   358.72 |    3.534 |    36.22 |    6.389 |   180.32 |
|  2048 |    128 |    1 |   2176 |    6.073 |   337.25 |    3.535 |    36.20 |    9.608 |   226.48 |
|  4096 |    128 |    1 |   4224 |   12.564 |   326.00 |    3.753 |    34.10 |   16.318 |   258.86 |
|  8192 |    128 |    1 |   8320 |   26.474 |   309.43 |    3.938 |    32.50 |   30.412 |   273.57 |
| 16384 |    128 |    1 |  16512 |   57.800 |   283.46 |    4.252 |    30.10 |   62.052 |   266.10 |
| 32768 |    128 |    1 |  32896 |  131.884 |   248.46 |    4.631 |    27.64 |  136.515 |   240.97 |
```

- Prefill: **248–438 tok/s** (decreasing with context)
- Generation: **28–47 tok/s** (decreasing with context)
- Flash attention enabled, 99 offloaded layers, max context 36K

## oMLX Benchmarks (Reference, M3 Max)

These benchmarks were obtained with the 4-bit MLX DWQ model running under
omlx with the `bailing_hybrid` architecture from PR #1227. These are
**reference numbers** — not from the l26f Metal engine — showing the
throughput available at each context depth. l26f aims to match or exceed
these figures with native Metal kernels.

**Model**: Ling-2.6-flash-mlx-4bit-DWQ  
**Server**: [omlx](https://github.com/jundot/omlx) (MLX-based, optimized for Apple Silicon)

### Single Request Results

| Test | TTFT (ms) | TPOT (ms) | pp TPS | tg TPS | E2E (s) | Throughput | Peak Mem |
|---|---:|---:|---:|---:|---:|---:|---:|
| pp1024/tg128 | 3,513 | 27.3 | 291.5 t/s | 36.9 t/s | 6.98 | 165.0 t/s | 55.33 GB |
| pp4096/tg128 | 16,396 | 25.7 | 249.8 t/s | 39.2 t/s | 19.66 | 214.9 t/s | 55.99 GB |
| pp8192/tg128 | 32,563 | 27.1 | 251.6 t/s | 37.2 t/s | 36.00 | 231.1 t/s | 56.57 GB |
| pp16384/tg128 | 66,355 | 27.7 | 246.9 t/s | 36.3 t/s | 69.88 | 236.3 t/s | 57.75 GB |
| pp32768/tg128 | 136,723 | 29.0 | 239.7 t/s | 34.8 t/s | 140.40 | 234.3 t/s | 60.10 GB |
| pp65536/tg128 | 296,868 | 32.7 | 220.8 t/s | 30.8 t/s | 301.02 | 218.1 t/s | 64.80 GB |

### Continuous Batching (pp1024/tg128)

| Batch | tg TPS | Speedup | pp TPS | pp TPS/req | TTFT (ms) | E2E (s) |
|---|---:|---:|---:|---:|---:|---:|
| 1× | 36.9 t/s | 1.00× | 291.5 t/s | 291.5 t/s | 3,513 | 6.98 |
| 2× | 46.1 t/s | 1.25× | 212.6 t/s | 106.3 t/s | 9,536 | 15.19 |
| 4× | 51.4 t/s | 1.39× | 213.1 t/s | 53.3 t/s | 18,898 | 29.18 |

### Current l26f Metal Speed

Current l26f Metal generation throughput is **~6-7 tok/s** on MacBook
Pro M3 Max with the fused MoE + GLA path. The IQ4_NL matvec kernel is
the current bottleneck; further optimization of the MoE dispatch and
MLA GPU implementation are expected to improve this toward the ~37 tok/s
single-request level demonstrated by omlx.

## Build

```sh
cd worktrees/l26f
make clean && make test_l26f_multilayer
```

No external libraries. Metal framework required (macOS only, Apple Silicon).

## Acknowledgements

This project is a fork/branch of **ds4** by antirez. The original ds4 engine
for DeepSeek-V4-Flash provided the architecture, kernel patterns, Metal graph
executor design, and testing methodology that made l26f possible. We are
thankful for that foundation.

This project also exists thanks to **llama.cpp and GGML** (Georgi Gerganov
and contributors), whose quantization formats, GGUF ecosystem, Metal kernels,
and hard-won engineering knowledge are the bedrock this entire local inference
ecosystem stands on.

The Ling-2.6-Flash model was created by **Inclusion AI**. The MLX 4-bit DWQ
quantization was provided by the **mlx-community**. The `bailing_hybrid`
architecture reference implementation comes from
[mlx-lm PR #1227](https://github.com/ml-explore/mlx-lm/pull/1227).

Some source-level pieces from ds4 are retained under the MIT license.
The l26f-specific additions (GLA kernel, grouped MoE routing, IQ4_NL
matvec for bailing_hybrid, tokenizer integration) are also MIT-licensed.

## AI-Assisted Development

This project was developed with strong assistance from AI coding agents,
including:

- **OpenCode + GLM-5.1** — architecture design, kernel porting, debugging
- **OpenCode + Kimi-K2.6** — debugging, code review
- **OpenCode/Pi + DeepSeek-V4-Pro** — debugging, code review

Humans led the ideas, testing, and debugging. The agents accelerated the
implementation of Metal kernels, MoE dispatch, tokenizer integration, and
test harnesses.

