# Caching Analysis: ds4 vs llama.cpp for Hybrid Models

## Why ds4 Works and llama.cpp Doesn't

### The Problem

llama.cpp's KV cache for hybrid models (MLA + GLA) is unreliable. For a model like
Ling-2.6-flash (28 GLA layers + 4 MLA layers), prompt caching frequently produces
corrupted generation after cache restore. This makes agentic work (long conversations,
chain-of-thought, multi-turn prompts) impractical.

### DeepSeek-V4 (ds4's model) Is Also Hybrid — and Works

DeepSeek-V4 Flash uses MLA (Multi-Latent Attention) — like Ling-2.6's global layers.
ds4's KV cache is **reliable** because of fundamental architectural differences:

---

## The Four Root Causes in llama.cpp

### 1. Sliding Window vs Full Prefix

| | ds4 | llama.cpp |
|---|---|---|
| Raw KV storage | Ring buffer: 128 rows per layer | Cell array: all prefix rows |
| 64K context raw cache | ~256KB/layer | ~35MB/layer (576 dims × 64K cells × 2 bytes) |
| Save/restore | Bounded, deterministic | Unbounded, grows with context |

ds4 only saves the last 128 raw KV rows (sliding window) + all compressed rows +
compressor frontier state. The older raw rows are compressed into sparse index caches.
llama.cpp stores every cell for the full prefix — any serialization bug affects
gigabytes of data.

### 2. V-as-K-View Coupling (MLA-specific)

llama.cpp stores only K in the KV cache for MLA layers (no separate V). At attention
time, V is created as a **view into K**:

```cpp
// llama-graph.cpp:2322
ggml_tensor * v = ggml_view_4d(ctx0, k, v_cur->ne[0],
    k->ne[1], k->ne[2], k->ne[3], ...);
```

The view reads the first `kv_lora_rank` rows of K as V. If cache restore changes
K's strides or shape, V silently reads garbage.

ds4 avoids this entirely — both Q and KV stay in compressed (absorbed) form. Q is
expanded through wq_b during Q computation, KV stays at 512 dims. Attention is a
direct dot-product with no derived views.

### 3. Multi-Memory Composition (Ling-2.6-specific)

Ling-2.6 uses two different memory systems:
- **KV cache** for MLA layers (7, 15, 23, 31)
- **Recurrent state** for GLA layers (all other 28 layers)

These live in `llama_memory_hybrid` which composes `llama_kv_cache` +
`llama_memory_recurrent`. Each has its own save/restore path. Any synchronization
bug between the two produces corrupted generation.

ds4's DeepSeek-V4 uses only MLA layers. No recurrent state. One memory type
to save/restore.

### 4. General-Purpose vs Single-Model

llama.cpp supports hundreds of architectures with conditional branches everywhere:
`is_mla?`, `has_v?`, `v_trans?`, `fa?`, `n_stream?`, `attn_rot?`, `swa?`.

ds4 has **zero conditional branches** in the KV cache code. Every dimension,
layer count, and compression ratio is a hardcoded constant. The checkpoint
format is model-specific with a fixed magic/version header.

---

## What We Need for l26f

Take ds4's approach for Ling-2.6:

1. **Bounded KV cache**: Sliding window ring for raw MLA KV, compressed caches for long-range
2. **No V-as-K views**: Compute absorbed Q and do direct dot-product attention
3. **Separate GLA state**: Treat GLA recurrent state as a separate serializable blob
4. **Complete self-describing checkpoint**: Tokens + logits + all cache rows + GLA state in one file
5. **Model-hardcoded dimensions**: No generic architecture dispatch

### ds4 Checkpoint Format (Reference)

```
HEADER:  magic(6) + version + ctx_size + prefill_cap + raw_cap + raw_window +
         comp_cap + checkpoint_len + n_layer + n_head + head_dim +
         indexer_head_dim + vocab + raw_live
TOKENS:    raw checkpoint tokens (full prefix)
LOGITS:    current logit buffer (vocab_size × 4 bytes)
PER-LAYER: n_comp, n_index_comp, raw_window rows, compressed rows,
           attn_state_kv, attn_state_score, index_comp_kv,
           index_state_kv, index_state_score
```

### l26f Checkpoint Format (Proposed)

```
HEADER:  magic + version + ctx_size + n_layer + n_mla_layers + n_gla_layers +
         kv_lora_rank + n_rot + raw_cap + comp_cap + vocab
TOKENS:    raw checkpoint tokens
LOGITS:    current logit buffer
MLA KV:   per-layer ring buffer positions + raw KV rows + compressed caches (if any)
GLA:      per-layer recurrent state (the S state tensor)
```
