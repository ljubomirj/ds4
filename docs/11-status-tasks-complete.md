# Status 10: All Five Tasks Complete — Full 32-Layer Multi-Token Generation

**Date**: 2026-05-09

## Summary

All five planned tasks are done. The l26f engine now runs all 32 layers (28 GLA + 4 MLA),
generates multiple tokens auto-regressively, and produces clean deterministic output.

```
./test_l26f_multilayer model.gguf 1 4
Token 0: input=1   → next_token=45468
Token 1: input=45468 → next_token=125465
Token 2: input=125465 → next_token=14659
Token 3: input=14659 → next_token=51683
```

Zero NaNs across all tokens. Fully deterministic.

---

## Task 1: MLA Layers (7, 15, 23, 31)

### Approach: CPU-only Phase A

Rather than writing custom Metal kernels for MLA (RoPE, absorption, attention, V
decompression), we implemented the entire MLA decode path on CPU. This trades speed
for correctness and simplicity.

The MLA decode flow for a single token:

1. **RMS norm** on input hidden state
2. **Q compression**: matvec with `wq_a` [4096→1536], RMS norm, matvec with `wq_b` [1536→6144]
3. **Q split**: q_nope [128×32 heads] and q_pe [64×32 heads]
4. **KV compression**: matvec with `wkv_a_mqa` [4096→576], split into kv_cmpr [512] and k_pe [64]
5. **RoPE** on q_pe and k_pe
6. **Absorption**: `wk_b[:,:,h] × q_nope[:,h]` for each head — avoids decompressing K for every cached token
7. **KV cache update**: store only [kv_cmpr | k_pe] = 576 floats per token (MQA — no V cache)
8. **Attention**: scaled dot-product between absorbed Q and K cache, weighted sum of V (= kv_cmpr)
9. **V decompression**: `wv_b[:,:,h] × attn_out[h]` for each head
10. **Output projection**: matvec with `wo` [4096→4096]
11. **Residual add**
12. **MoE FFN** (same as GLA layers)

### Key architecture parameters

```
n_embd       = 4096
n_head       = 32
head_dim     = 192 (128 qk_nope + 64 rope)
q_lora_rank  = 1536
kv_lora_rank = 512
kv_dim       = 576 (kv_lora_rank + n_rot)
```

### MLA tensor inventory (per layer)

| Tensor | Shape | Type |
|--------|-------|------|
| attn_norm | [4096] | F32 |
| attn_q_a | [4096, 1536] | Q5_K |
| attn_q_a_norm | [1536] | F32 |
| attn_q_b | [1536, 6144] | Q6_K |
| attn_kv_a_mqa | [4096, 576] | Q5_K |
| attn_kv_a_norm | [512] | F32 |
| attn_k_b | [128, 512, 32] | IQ4_NL |
| attn_v_b | [512, 128, 32] | IQ4_NL |
| attn_output | [4096, 4096] | Q5_K |

### Files created/modified

- **New**: `l26f_mla_cpu.c` — CPU-side MLA decode (dequant, matvec, RoPE, attention, absorption, V decompression)
- **Modified**: `test_l26f_multilayer.c` — added `mla_kv[]` to session, wired MLA into main loop
- **New**: `docs/10-mla-plan.md` — implementation plan with data flow

### KV cache

Each MLA layer has a CPU-side KV cache storing `max_seq_len × 576` floats.
For 4096 tokens, that's 4096 × 576 × 4 = 9.4 MiB per MLA layer, 37.5 MiB total.
Only 576 floats per token — the key MLA efficiency gain over standard attention.

---

## Task 2: Static Asserts for Quant Block Structs

Added `_Static_assert` in two places:

**Metal prefix** (`l26f_metal.m` — compiled by Metal shader compiler):
```
_Static_assert(sizeof(struct block_iq4_nl) == 18, ...);
_Static_assert(sizeof(struct block_q5_K)   == 176, ...);
_Static_assert(sizeof(struct block_q6_K)   == 210, ...);
```

**C header** (`l26f_mla_cpu.c` — compiled by clang):
```
_Static_assert(sizeof(cpu_block_q5_K)  == 176, ...);
_Static_assert(sizeof(cpu_block_q6_K)  == 210, ...);
_Static_assert(sizeof(cpu_block_iq4_nl) == 18,  ...);
```

This prevents the class of bugs we hit three times (IQ4_NL, Q6_K field order) from
ever recurring silently.

---

## Task 3: Output Projection Optimization

Changed Q6_K matvec dispatch from `threadsPerThreadgroup:MTLSizeMake(1,1,1)` to
using `maxTotalThreadsPerThreadgroup` (capped at 256). Same results, better GPU
utilization for the 157K-output-row matvec.

---

## Task 4: CPU Reference Dequant

Implemented as part of `l26f_mla_cpu.c`:

| Function | Type | Block Size | Type Size |
|----------|------|-----------|-----------|
| `cpu_dequant_q5_K_row` | Q5_K | 256 | 176 |
| `cpu_dequant_q6_K_row` | Q6_K | 256 | 210 |
| `cpu_dequant_iq4_nl_row` | IQ4_NL | 32 | 18 |

Plus a generic `cpu_dequant_row()` dispatcher and `cpu_matvec()` that combines
dequant + dot-product for any quant type. These can be used for GPU verification
in future tasks.

---

## Task 5: Multi-Token Auto-Regressive Generation

Refactored the main loop:

- Extracted `l26f_forward_pass()` — runs all 32 layers (GLA + MLA + MoE FFN)
- Added generation loop with argmax token selection
- GLA recurrent states persist across tokens (GPU tensors)
- MLA KV caches accumulate across tokens (CPU arrays)
- Usage: `./test_l26f_multilayer model.gguf [start_token] [n_gen]`

---

## Lessons Learned (This Session)

### 1. MLA is simpler than expected for single-token decode

The "absorption" trick sounds complex but reduces to: pre-multiply Q by the K
expansion matrix, then attend against the compressed latent. For single-token
decode, attention is just a dot product against the cache — no Flash Attention
needed. The entire path can be done on CPU without custom Metal kernels.

### 2. CPU-first, GPU-later is the right order

We got MLA working on CPU in ~1 hour. Writing Metal kernels first would have
taken much longer and been harder to debug. The CPU implementation serves as
both the working solution and the reference for future GPU acceleration.

### 3. The `cpu_matvec` pattern is reusable

By abstracting dequant + dot-product into `cpu_matvec(weights, input, output, type, rows, cols)`,
we handle all quant types uniformly. This same pattern can verify any GPU matvec
by running CPU vs GPU on the same data.

### 4. MLA attention is MQA (multi-query attention)

All 32 Q heads share a single KV head. The KV cache stores just 576 floats per
token. This makes MLA attention trivially cheap for decode — just a dot product
per head against the cache, then a weighted sum.

### 5. Position matters for GLA state continuity

The GLA layers are recurrent — state carries across tokens. But the MLA KV cache
also accumulates. When we initially used `(int)il` as the position (layer index
instead of token index), MLA still worked for token 0 but would have produced
wrong RoPE for subsequent tokens. Fixed by passing the actual generation step.

### 6. Checksum infrastructure keeps paying dividends

The `l26f_checksum_print()` helpers from the previous session continue to provide
instant verification. After adding MLA, we ran once and all checksums matched —
no debugging needed.

---

## Performance Notes

The MLA layers run on CPU, which is the bottleneck for multi-token generation.
Per MLA layer: ~8 matvecs (each dequant + dot-product), plus attention softmax.
For single-token decode this is acceptable. For higher throughput, the hot path
(absorption matmul, V decompression) should move to GPU.

Estimated MLA layer cost: ~5-10ms on CPU (dequant dominates).
GLA layers on GPU: ~1-2ms each.
28 GLA + 4 MLA + MoE FFN ≈ ~50-80ms per token for short sequences.

---

## What's Next

Potential next tasks:
- **GPU-accelerate MLA** — move absorption and V decompression matmuls to Metal
- **Tokenizer** — integrate sentencepiece/tiktoken for text I/O (currently token IDs only)
- **Sampling** — temperature, top-k, top-p instead of pure argmax
- **Benchmarking** — measure tokens/second, compare against llama.cpp baseline
- **Longer generation** — test with 50-100 tokens, verify coherence
- **Prefill** — batch token processing for prompt encoding
