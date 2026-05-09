# l26f — Project Status

**Updated**: 2026-05-09 (morning)
**Branch**: `l26f` (ljubomirj/l26f fork of antirez/ds4)
**Model**: Ling-2.6-flash IQ4_NL quality GGUF (58 GB)

---

## The Short Version

We are building a narrow Metal inference engine for Ling-2.6-flash on top of
antirez's ds4 framework. The model is a 104B-parameter MoE (7.4B active) with
hybrid attention: 28 GLA layers + 4 MLA layers + 1 MTP head.

**What works**: One full transformer layer (layer 0) runs end-to-end on GPU with
zero NaNs — GLA attention + dense FFN. All quantized matvec kernels (Q5_K, IQ4_NL)
work. The Metal runtime handles the 58 GB model via zero-copy MTLBuffers.

**Current blocker**: Layers 1-31 use MoE FFN (256 experts, 8 active). The expert
routing and per-expert matvec are not yet implemented. Without MoE, those layers
produce garbage. Four MLA attention layers (7, 15, 23, 31) also need attention.

---

## History: How We Got Here

### Phase 1: Infrastructure (commits through `5764bf5`)

Ported ds4's Metal runtime, GGUF loader, and kernel infrastructure. Added our
own GGUF parser (`l26f_gguf.c`) for the bailing_hybrid architecture. Wrote Metal
kernels: GLA attention, IQ4_NL/Q5_K/Q6_K dequant+matvec, all in `metal/` directory.

Key decisions:
- Build on ds4's Metal runtime (command buffers, tensor alloc, model map views)
- Write our own GGUF loader (ds4's is hardcoded for DeepSeek)
- Port GLA kernel from llama.cpp's `ggml-metal.metal`
- Write quantized matvec kernels for IQ4_NL, Q5_K, Q6_K

### Phase 2: Getting Layer 0 Working (commits `141b72a` through `ad631b6`)

Four bugs blocked layer 0. Each produced NaN but for different reasons:

**Bug 1**: GGUF `tensor_data_pos` computed before iterating tensor entries.
All offsets wrong. Fix: compute after loop, aligned.

**Bug 2**: Weight offsets had `tensor_data_pos` subtracted. Metal views cover the
full file from byte 0; `abs_offset` is already a file offset. Fix: pass directly.

**Bug 3**: Token embedding is Q8_0, not F32. Raw byte reads produced 10^38 values.
Fix: CPU-side block dequant.

**Bug 4**: `block_iq4_nl` Metal struct was 34 bytes (`uint16_t qs[16]`) instead of
18 bytes (`uint8_t qs[16]`). Every IQ4_NL weight read corrupted. Fix: correct struct
+ rewrite kernel from llama.cpp reference.

After all four fixes, layer 0 runs the full pipeline:
```
embed(Q8_0) → rms_norm → qkv(Q5_K) → gate(Q5_K) → GLA → out_proj(Q5_K) → residual
→ ffn_norm → gate(IQ4_NL) → up(IQ4_NL) → SwiGLU → down(Q5_K) → residual
= valid output, 0 NaNs ✓
```

### Phase 3: Multi-Layer Attempt (uncommitted `test_l26f_multilayer.c`)

Wrote a driver that loops over all 32 layers. It runs GLA attention for each GLA
layer and dense FFN for layer 0 only. MoE FFN (layers 1-31) and MLA attention
(layers 7, 15, 23, 31) are stubbed out. This is the current state.

---

## What Works Now

| Component | Status | Proof |
|-----------|--------|-------|
| GGUF loader (540 tensors) | ✓ | Offsets verified against Python gguf lib |
| Metal runtime (58 GiB zero-copy) | ✓ | 2 overlapping MTLBuffer views |
| Q5_K matvec kernel | ✓ | Layer 0 QKV/gate/output all Q5_K, no NaN |
| IQ4_NL matvec kernel | ✓ | Layer 0 FFN gate/up are IQ4_NL, no NaN |
| Q8_0 CPU dequant | ✓ | Token embedding loads correctly |
| GLA attention kernel | ✓ | S=128, H=32, recurrent state updates |
| RMS norm (ds4) | ✓ | Pre-attention and pre-FFN |
| SwiGLU (ds4) | ✓ | FFN gate+up activation |
| Residual add (ds4) | ✓ | After attention and after FFN |
| Layer 0 full pipeline | ✓ | End-to-end, zero NaNs |
| Multi-layer GLA loop | Partial | Runs but MoE FFN skipped for layers 1-31 |

---

## Current Blocker: MoE FFN

### What MoE Does

Each MoE layer has 256 experts (small FFNs). For each token:
1. Router: hidden [4096] × `ffn_gate_inp` [4096, 256] → scores [256]
2. Softmax + top-k: select 8 experts (grouped: 8 groups, top-4 groups, 1 per group)
3. For each selected expert: gate matvec + up matvec + SwiGLU + down matvec
4. Weighted sum of expert outputs
5. Plus shared expert (always active): gate_shexp + up_shexp + SwiGLU + down_shexp

### MoE Tensor Layout (per layer, e.g. layer 1)

| Tensor | Type | Shape | Size | Purpose |
|--------|------|-------|------|---------|
| `ffn_gate_inp.weight` | F32 | [4096, 256] | 4 MB | Router projection |
| `exp_probs_b.bias` | F32 | [256] | 1 KB | Expert selection bias |
| `ffn_gate_exps.weight` | IQ4_NL | [4096, 1024, 256] | 604 MB | Expert gate projections |
| `ffn_up_exps.weight` | IQ4_NL | [4096, 1024, 256] | 604 MB | Expert up projections |
| `ffn_down_exps.weight` | Q5_K | [1024, 4096, 256] | 738 MB | Expert down projections |
| `ffn_gate_shexp.weight` | IQ4_NL | [4096, 1024] | 2.4 MB | Shared expert gate |
| `ffn_up_shexp.weight` | IQ4_NL | [4096, 1024] | 2.4 MB | Shared expert up |
| `ffn_down_shexp.weight` | Q5_K | [1024, 4096] | 2.9 MB | Shared expert down |

The expert weights are stored as a single merged tensor with 256 experts stacked
along the third dimension. Each expert has in_dim=4096, mid_dim=1024.

### What Needs Building

1. **Router**: F32 matvec [4096]×[4096,256] → softmax → group top-k → 8 expert
   indices + 8 weights. Can use ds4's `ds4_metal_matmul_f32_tensor` for the matvec.
   Need softmax + top-k kernels (ds4 has `kernel_softmax_f32` and argsort).

2. **Per-expert matvec**: For each of 8 selected experts, extract the expert's
   slice from the merged tensor and do gate/up/down matvec. The expert slice is
   at offset `expert_idx * expert_row_bytes` within the merged tensor.

3. **Weighted accumulation**: Sum expert outputs × routing weights.

4. **Shared expert**: Same as dense FFN but with shared expert weights.

### Approach

For single-token decode (our current use case), we can do the routing and expert
selection on CPU (it's just a matvec + softmax + top-8 of 256 — microseconds).
Then dispatch 8 expert matvecs on GPU. The shared expert is a regular FFN.

ds4 has `ds4_metal_routed_moe_one_tensor` but it's Q8_0-only and tightly coupled
to ds4's expert layout. We'll write our own MoE dispatch that reuses our existing
`l26f_metal_matvec_quant` function with expert-specific offsets.

---

## Roadmap

### P1: MoE FFN (current)

- CPU-side router (matvec + softmax + group top-k)
- Per-expert matvec dispatch (8 experts, IQ4_NL gate/up, Q5_K down)
- Shared expert FFN
- Wire into multi-layer driver

### P2: MLA Attention (layers 7, 15, 23, 31)

4 global attention layers use DeepSeek2-style MLA (compressed KV, q-lora).
ds4 has the full MLA pipeline — can likely reuse directly since MLA weights are
F32/Q5_K which we already handle.

### P3: Full Transformer Loop

All 32 layers working: 28 GLA + 4 MLA, each with MoE FFN. Should produce coherent
logits after output projection.

### P4: Tokenizer + Sampling + CLI

BPE tokenizer (157184 vocab), argmax/temperature/top-k/top-p sampling, interactive
CLI. The tokenizer can be ported from llama.cpp's `llama_vocab` or loaded from
the GGUF metadata.

### P5: Prefill + Prompt Processing

Currently single-token decode only. Need batched matvec for prompt prefill
(multiple tokens at once). The IQ4_NL batch matvec template instantiations
already exist (`kernel_mul_mv_ext_iq4_nl_f32_r1_{2,3,4,5}`).

---

## GGUF Files

| File | Size | Q5_K | IQ4_NL | Notes |
|------|------|------|--------|-------|
| `Ling-2.6-flash-IQ4_NL-bailing_hybrid-20260505-LJ.gguf` | 57 GB | 7 | 304 | Original quant |
| `Ling-2.6-flash-IQ4_NL-quality-bailing_hybrid-20260508-LJ.gguf` | 58 GB | 106 | 199 | Re-quantized, higher quality |

Both on NVME at `~/NVME_4TB_SSD_GRAUGEAR_Users_ljubomir/llama.cpp/bailing-hybrid/`.
Using the quality GGUF (attn QKV is Q5_K, which we verified first).

---

## Bugs Fixed (chronological)

1. **GGUF tensor_data_pos too early** (`141b72a`): computed before iterating tensor
   entries. Fix: compute after loop, aligned to 32 bytes.

2. **Weight offset subtracted tensor_data_pos** (`b2c4d08`): `abs_offset` is already
   a file offset. Metal views start at byte 0. Fix: pass `abs_offset` directly.

3. **Token embedding as F32** (uncommitted): it's Q8_0. Fix: CPU block dequant.

4. **block_iq4_nl struct 34 bytes not 18** (`ad631b6`): `uint16_t qs[16]` instead of
   `uint8_t qs[16]`. Also rewrote the IQ4_NL kernel from llama.cpp reference.

---

## Lessons Learned

1. **Copy block structs verbatim from llama.cpp**. Never write from memory. Verify
   sizeof matches GGUF type_size (18 for IQ4_NL, 176 for Q5_K, 210 for Q6_K).

2. **Pass `abs_offset` directly to Metal**. It's already a file offset. Never
   subtract `tensor_data_pos`.

3. **Follow ds4's calling conventions exactly**. If ds4 passes X, we pass X.

4. **NaN = wrong data**. Before debugging kernel math, verify offsets and data.

5. **Don't break what works**. When fixing l26f_dense.metal, I accidentally
   replaced the working Q5_K/Q6_K kernels. Only modify what needs fixing.

6. **One variable at a time**. When adding new code AND a new GGUF, debug with
   the old known-good GGUF first to isolate the variable.

---

## Key Files

| File | Role |
|------|------|
| `l26f_gguf.c` | GGUF model loader (540 tensors, metadata) |
| `l26f.h` | Data structures, quant type table |
| `l26f_metal.m` | Metal runtime glue (model map, kernel dispatch) |
| `l26f_metal.h` | Public Metal C API |
| `metal/l26f_dense.metal` | IQ4_NL/Q5_K/Q6_K dequant + matvec kernels |
| `metal/l26f_gla.metal` | GLA attention kernel |
| `test_l26f.c` | Single-layer test (layer 0, works) |
| `test_l26f_multilayer.c` | Multi-layer driver (MoE FFN not yet wired) |
| `docs/` | This documentation |

## Build & Run

```bash
cd ~/llama.cpp/contrib/l26f
make
./test_l26f <model.gguf>              # Layer 0 end-to-end (works)
./test_l26f_multilayer <model.gguf> 1 # Multi-layer (MoE FFN skipped)
```
