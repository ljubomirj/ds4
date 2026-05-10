# On-Disk Resources for l26f Development

## llama.cpp Worktrees (at ~/llama.cpp/worktrees/)

| Path | Branch | Purpose |
|------|--------|---------|
| `LJ-Ling-2.6-flash-r2/` | `LJ-Ling-2.6-flash-r2` | **Primary reference** — latest Metal kernels, MLA graph construction |
| `LJ-Ling-2.6-flash/` | `LJ-Ling-2.6-flash` | Stable, working GLA state write-back fix |
| `LJ-Ling-2.6-flash-mtp/` | `LJ-Ling-2.6-flash-mtp` | MTP speculative decoding merged |
| `LJ-Ling-2.6-flash-old/` | older | Snapshot from earlier work |
| `bailing-hybrid/` | early | Original bailing hybrid support |
| `ssweens-ling-2.6/` | ssweens | ssweens' Ling support PR |
| `upstream-master/` | upstream | Clean upstream for diffing |
| `mtp-clean/` | mtp-clean | am17an MTP PR |

## Key Files in LJ-Ling-2.6-flash-r2 (for MLA Metal)

### Metal Kernels
- `ggml/src/ggml-metal/ggml-metal.metal` — all Metal kernels in one file
  - `kernel_rope_neox` — RoPE (Neox style, used by Ling)
  - `kernel_flash_attn_ext` — tiled flash attention (half8x8 simdgroup)
  - `kernel_flash_attn_ext_vec` — vector flash attention (half4x4, small batches)
  - `kernel_mul_mm` — matrix-matrix multiply (handles 3D via r2/r3 broadcast)
  - `kernel_mul_mv_ext` — matrix-vector extended (small batches)
  - `dequantize_iq4_nl` / `dequantize_iq4_nl_t4` — IQ4_NL dequant helpers

### Metal Dispatch
- `ggml/src/ggml-metal/ggml-metal-ops.cpp` — kernel dispatch, threadgroup sizing
- `ggml/src/ggml-metal/ggml-metal-impl.h` — args structs for all kernels
- `ggml/src/ggml-metal/ggml-metal-device.cpp` — pipeline selection/compilation
- `ggml/src/ggml-metal/ggml-metal-device.m` — device-level pipeline tables

### MLA Graph Construction
- `src/models/bailing-hybrid.cpp` — MLA + GLA hybrid model
  - Lines 237-345: MLA attention (Q proj, KV compress, RoPE, absorption, flash attn, V-decompress)
  - Line 293: absorption = `ggml_mul_mat(wk_b, q_nope)` — 3D weight, r2/r3=1
  - Line 302: flash attn = `build_attn(Q, K, V, wv_b)` — MQA, dk=576, dv=512
  - Line 318: V-decompress = `ggml_mul_mat(wv_b, attn_out)` — 3D weight

- `src/llama-graph.cpp` — `build_attn()` with MLA V-decompress path
- `src/llama-hparams.h` — MLA hparams (`n_embd_head_k_mla_impl`, `n_embd_head_v_mla_impl`)

## Other llama.cpp Contributions (at ~/llama.cpp/contrib/)

| Path | Purpose |
|------|---------|
| `ds4/` | DeepSeek-V4 Flash narrow Metal engine by antirez (l26f's inspiration) |
| `kernel-anvil/` | Metal kernel testing infrastructure |
| `llama-benchy/` | Benchmarking tools |
| `asi-speedup/` | ASI speedup experiments |
| `l26f/` | **Our engine** — Ling-2.6-flash narrow Metal inference |

## Apple / MLX Ecosystem (at ~/LJ-asi-mlx/)

| Path | Purpose |
|------|---------|
| `mlx/` | Apple MLX framework source |
| `mlx-lm/` | MLX-based language model inference |
| `vllm-metal/` | vLLM Metal backend |
| `vllm-mlx/` | vLLM MLX backend |
| `MTPLX/` | MTP speculative decoding on MLX |
| `ds4/` | ds4 reference (MLX version?) |
| `metal-int4-sdpa/` | Apple Metal int4 SDPA reference |
| `Rapid-MLX/` | Rapid inference on MLX |
| `autoresearch-macos/` | Auto research macOS Metal kernels |
| `autoresearch-mlx/` | Auto research MLX |
| `omlx/` | Open MLX |
| `mlx-vlm/` | MLX vision-language models |
| `mlx-audio/` | MLX audio models |
| `turboquant-llama3.170B/` | Turbo quantization for large models |
| `mlx-openbench/` | MLX benchmarking |
| `mlx-embeddings/` | MLX embedding models |
| `unsloth-mlx/` | Unsloth MLX fine-tuning |
| `mlx-esm-2/` | MLX ESM-2 protein models |
| `mlx-data/` | MLX data loading |
| `ZMLX/` | ZMLX framework |
| `nanoGPT_mlx/` | nanoGPT on MLX |
| `mlx-pretrain/` | MLX pretraining |
| `mlx-lm-lora/` | MLX LoRA fine-tuning |
| `jangq/` | JangQ inference |

### Symlinks (external)
- `transformers-to-mlx` → `/Users/ljubomir/LJ-asi-mlx/transformers-to-mlx`
- `dflash-mlx` → `/Users/ljubomir/LJ-DLM-diffusion/dflash-mlx`
- `ddtree-mlx` → `/Users/ljubomir/LJ-DLM-diffusion/ddtree-mlx`
- `dflash-metal` → `/Users/ljubomir/LJ-DLM-diffusion/dflash-metal`
- `transformer-vm` → `/Users/ljubomir/LJ-ML-comp/transformer-vm`

## Key Architectural Insight

llama.cpp's Ling/Bailing support has **no dedicated MLA Metal kernels**. MLA is
built from composition of existing generic kernels:

| MLA Operation | llama.cpp Op | Metal Kernel | Notes |
|---|---|---|---|
| Q LoRA (wq_a, wq_b) | `MUL_MAT` | `kernel_mul_mm` / `kernel_mul_mv_ext` | 2D weight |
| KV compress (wkv_a_mqa) | `MUL_MAT` | same | 2D, MQA |
| RoPE (q_pe, k_pe) | `ROPE` | `kernel_rope_neox` | Standard |
| K-absorption (wk_b @ q_nope) | `MUL_MAT` | `kernel_mul_mm` | **3D** [P,C,H], r2/r3=1 |
| Flash Attention | `FLASH_ATTN_EXT` | `kernel_flash_attn_ext` or `_vec` | MQA: ne02/ne_12_2 ratio |
| V-decompress (wv_b @ attn_out) | `MUL_MAT` | `kernel_mul_mm` | **3D** [C,P,H], r2/r3=1 |

The 3D weight tensors use standard mul_mm with broadcast ratios r2=1, r3=1.
The flash attention kernel handles MQA by mapping Q head → KV head via `ne02/ne_12_2`.
For MLA: dk=kv_lora_rank+n_rot=576, dv=kv_lora_rank=512.
