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

## Model Weights

The GGUF file is at:
`models/Ling-2.6-flash-IQ4_NL-quality-bailing_hybrid-20260508-LJ.gguf` (58 GiB).

It was quantized to 4-bit using the MLX DWQ (Data-aware Weight Quantization)
method from the `mlx-community/Ling-2.6-flash-mlx-4bit-DWQ` repository.

This engine only works with this specific GGUF layout. It is not a general
GGUF loader.

## Build

```sh
cd worktrees/l26f
make clean && make test_l26f_multilayer
```

No external libraries. Metal framework required (macOS only, Apple Silicon).

## Speed

Current generation throughput is approximately **6-7 tok/s** on MacBook Pro
M3 Max with the fused MoE + GLA path. The fused IQ4_NL matvec kernel is
the current bottleneck; further optimization of the MoE dispatch and
MLA GPU implementation are expected to improve this.

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
quantization was provided by the **mlx-community**.

Some source-level pieces from ds4 are retained under the MIT license.
The l26f-specific additions (GLA kernel, grouped MoE routing, IQ4_NL
matvec for bailing_hybrid, tokenizer integration) are also MIT-licensed.
