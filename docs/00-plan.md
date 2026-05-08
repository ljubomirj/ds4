# l26f: Ling-2.6-flash Narrow-Metal Inference Engine

A narrow-cut inference engine for Ling-2.6-flash (104B MoE, 7.4B active, hybrid MLA+GLA),
modeled after [ds4](https://github.com/antirez/ds4) (DeepSeek-V4 Flash narrow engine).

## Motivation

llama.cpp can run Ling-2.6-flash but:
- **KV cache / prompt caching is broken** for hybrid models (see [01-caching-analysis.md](docs/01-caching-analysis.md))
- **Too general** — hundreds of architectures, conditional paths everywhere
- **No per-layer precision control** in GGUF quantization (see [02-quantization-analysis.md](docs/02-quantization-analysis.md))

ds4 proves the narrow-cut approach works: hardcode model dimensions, use a sliding-window
ring buffer for KV cache, self-contained checkpoint format, zero conditional branches.

## Architecture

### Source

Narrow-cut from `LJ-Ling-2.6-flash-r2` branch of llama.cpp (which has full bailing-hybrid
support including GLA Metal kernels, MLA KV cache, and MTP speculative decoding).

### Components to Extract

From llama.cpp:

| Component | Files | Purpose |
|---|---|---|
| GLA Metal kernel | `ggml-metal-ops.cpp`, `ggml-metal.metal` | Gated Linear Attention on GPU |
| Non-fused GLA | `ggml-gated-linear-attn` op | Pre-computed slopes, keep_intermediates for state snapshots |
| MLA attention | Graph builder patterns | Q-absorbed attention, KV compressed cache |
| MoE FFN | `SwitchGLU` pattern | Expert routing, group selection, shared experts |
| GGUF loader | `llama-model-loader` | Tensor loading, mmap (or direct Metal alloc) |
| Tokenizer | BPE tokenizer | Ling-2.6 uses BPE with custom pre-tokenizer |

### Key Differences from llama.cpp

| Aspect | llama.cpp | l26f |
|---|---|---|
| KV cache | Cell-based full prefix | Sliding window ring + compression |
| Prompt cache | General-purpose serialization | Self-contained snapshot (ds4-style) |
| V for MLA | View of K cache | Direct dot-product (Q pre-expanded) |
| GLA state | `llama_memory_hybrid` composition | Explicit GLA state blob in checkpoint |
| Model dispatch | Per-architecture branches | Hardcoded Ling-2.6 dimensions |
| Quantization | Generic GGUF all types | Single quantization mix (IQ4_NL + selective upgrades) |
| Layers | Conditional per-layer paths | Two hardcoded paths: MLA (4 layers) + GLA (28 layers) |

### Ling-2.6-flash Model Specs

| Parameter | Value |
|---|---|
| Total params | 107.49 B (104B active) |
| Active params | ~7.4B (8/256 experts) |
| Hidden size | 4096 |
| FFN size | 9216 |
| Expert FFN | 1024 |
| Shared expert FFN | 1024 |
| Experts | 256 (8 active, 8 groups, 4 group active) |
| Layers | 33 (32 transformer + 1 MTP) |
| MLA layers | 7, 15, 23, 31 (global attention, kv_lora_rank=512, q_lora_rank=1536) |
| GLA layers | All others (gated linear attention, 32 heads, head_dim=128) |
| Vocabulary | 157184 tokens (BPE) |
| Max context | 131072 |
| RoPE | freq_base=6M, dim=64 (partial_rotary_factor=0.5 for GLA) |
| MTP heads | 1 (layer 32, nextn_predict_layers=1) |

### Memory Budget (M2 Max, 96 GB)

| Component | Size |
|---|---|
| Model weights (IQ4_NL + upgrades) | ~58 GB |
| KV cache (ring buffer, MLA layers) | ~50 MB |
| GLA recurrent state | ~57 MB |
| Compute buffers | ~200 MB |
| Checkpoint buffer | ~200 MB |
| **Total** | **~59 GB** |
| **Free** | **~37 GB** |

---

## Implementation Plan

### Phase 1: Weight Loading — DONE
- [x] GGUF loader with metadata parsing (l26f_gguf.c)
- [x] Tensor lookup by name (l26f_model_find_tensor)
- [x] All 540 tensors verified: F32=228, Q5_K=106, Q6_K=6, IQ4_NL=199, Q8_0=1
- [x] MLA/GLA layer detection from head_count_kv metadata

### Phase 2: Metal Layer — DONE
- [x] GLA kernel ported from llama.cpp (metal/l26f_gla.metal) + dispatch
- [x] IQ4_NL/Q5_K/Q6_K block structs + dequant helpers (metal/l26f_dense.metal)
- [x] IQ4_NL, Q5_K, Q6_K single-token matvec kernels + dispatch
- [x] Generic quantized matvec router (l26f_metal_matvec_quant)
- [x] 11 reusable ds4 kernels: norms, activations, bin ops, embeddings, softmax, RoPE
- [ ] MLA kernel (DeepSeek2-style compressed attention)
- [ ] MoE expert matmul with IQ4_NL weights

### Phase 3: Inference Driver — IN PROGRESS
- [x] Tensor verification tool
- [ ] Metal initialization + model map
- [ ] GLA layer end-to-end (single layer, single token)
- [ ] Full 32-layer transformer loop
- [ ] MLA layer forward pass
- [ ] MoE FFN forward pass
- [ ] BPE tokenizer
- [ ] CLI / server

#### Recurrent State Cutoff Strategy

GLA's S state is fixed-size (128×128×32 floats = ~2 MB per layer in F32, ~56 MB
for 28 layers). It doesn't grow with context. But over extremely long sequences
(100K+ tokens), the accumulated state can degrade — old information gets "sticky"
due to gating numerical drift.

**Decision**: Snapshot S state at regular checkpoint intervals. When restoring:
1. Load nearest snapshot S state
2. Replay only tokens since that snapshot to reconstruct current S
3. This bounds both restore time AND the effective recurrent horizon

All model weights are ≤ 8-bit quantized. The S state snapshots (+KV ring) are
tiny compared to the 58 GB model. Checkpoint files stay practical even with
frequent snapshots.

### Phase 6: Server
- [ ] HTTP server (libmicrohttpd or minimal select/poll)
- [ ] Chat completions endpoint
- [ ] Health endpoint

### Phase 7: MTP (Deferred)
- [ ] MTP head forward pass (layer 32)
- [ ] Speculative decoding loop
- [ ] MTP checkpoint state

---

## Questions to Resolve

1. **GLA state in sliding window**: ds4 uses ratio-4 and ratio-128 compression for
   MLA attention. Does GLA need similar compression? The recurrent S state naturally
   compresses history — maybe no extra compression needed.

2. **Precision of compute**: ds4 uses float16 for compute, float32 for KV cache.
   For Ling-2.6 with MoE on IQ4_NL, what compute precision? Metal supports fp16
   and int8 matmul natively on M2 Max (simdgroup matrix mul).

3. **MoE routing**: Can we pre-compute expert selection on CPU and only ship
   selected expert weights to Metal? This would dramatically reduce bandwidth.

4. **Flash attention for MLA**: Metal has flash attention support in llama.cpp
   — can we use it with ds4-style compressed MLA?

5. **Multi-sequence**: ds4 supports only one sequence. Do we need multi-sequence
   for the server? Each sequence needs separate KV ring + GLA state.

---

## Reference Files

| File | Source |
|---|---|
| `docs/bailing_hybrid.py` | Original MLX implementation from [mlx-lm#1227](https://github.com/ml-explore/mlx-lm/pull/1227) |
| `LL-Ling-2.6-flash-r2/src/models/bailing-hybrid.cpp` | llama.cpp graph builder (reference) |
| `LL-Ling-2.6-flash-r2/ggml/src/ggml-metal/ggml-metal-ops.cpp` | GLA Metal kernel |
| `../ds4/ds4.c` | ds4 reference (caching, Metal graph, server) |

## Key Insights from ds4

1. **Hardcode everything** — no generic dispatch, no conditional branches
2. **KV as ring buffer** — bounded, deterministic save/restore
3. **Complete checkpoint** — tokens + logits + all state in one file
4. **Direct Metal tensors** — no GGML intermediate layer
5. **Pre-computed weight copies** — materialize IQ4_NL → float16 on GPU once, reuse
