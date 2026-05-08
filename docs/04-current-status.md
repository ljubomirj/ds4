# l26f — Current Status

**Date**: 2026-05-08 (evening)
**Branch**: `l26f` (ljubomirj/l26f fork of antirez/ds4)
**Model**: Two GGUFs available — see "GGUF Files" section below

---

## What Works (Verified End-to-End)

### Single GLA Layer (test_l26f.c)

Layer 0's **attention path** is fully working:

```
Input (4096 floats)
  → RMS Norm (F32 weight, ds4 kernel)
  → QKV Matvec (Q5_K weight [4096×12288], our kernel)
  → Gate Matvec (Q5_K weight [4096×4096], our kernel)
  → GLA Attention (our Metal kernel, S=128, H=32)
  → Output Matvec (Q5_K weight [4096×4096], our kernel)
  → Residual Add (ds4 kernel)
  → Output: sum=2026.0, no NaNs ✓
```

This is the only fully verified pipeline so far. It proves:
- GGUF loader → correct offsets ✓
- Metal model map (58 GiB, zero-copy MTLBuffers) ✓
- Q5_K matvec kernel ✓
- GLA kernel ✓
- RMS norm, residual add ✓

### Infrastructure That Works

- GGUF model loader (540 tensors, all offsets verified against Python gguf lib)
- Metal shader compilation (20 .metal files → single library)
- Metal runtime (init, tensor alloc/view/free, command buffers)
- Q8_0 CPU-side dequant (used for token embedding)

---

## Current Blocker: ~~IQ4_NL Block Struct Size Mismatch~~ FIXED

Fixed in commit `ad631b6`. The `block_iq4_nl` Metal struct had `uint16_t qs[16]`
(32 bytes) instead of `uint8_t qs[16]` (16 bytes). Total was 34 bytes instead of
18. The IQ4_NL kernel was also rewritten, ported from llama.cpp's reference.

Layer 0 full pipeline now works end-to-end with zero NaNs:
```
embed → rms_norm → qkv(Q5_K) → gate(Q5_K) → GLA → out_proj(Q5_K) → residual
→ ffn_rms_norm → ffn_gate(IQ4_NL) → ffn_up(IQ4_NL) → SwiGLU → ffn_down(Q5_K) → residual
= sum=-258.3, 0 NaNs ✓
```

### Next Blocker: Multi-layer loop needs MoE FFN

The multi-layer driver (`test_l26f_multilayer.c`) runs layer 0 with dense FFN
but layers 1-31 all use MoE FFN. The MoE routing + expert matvec is not yet
implemented. Currently those layers just pass through residual (no FFN).

### The Problem

Layer 0's **FFN gate** and **FFN up** projections use IQ4_NL (type 20) weights.
The IQ4_NL matvec kernel (`kernel_mul_mv_iq4_nl_f32`) produces 90% NaN for these
tensors, while the Q5_K matvec kernel works correctly for all Q5_K tensors.

The root cause (discovered but not yet fixed): **the `block_iq4_nl` Metal struct
does not match the actual GGUF block layout.**

### Block Layout Comparison

**GGUF IQ4_NL block** (18 bytes per block, 32 elements per block):
```
Offset  Size  Field
0       2     f16 scale (d)
2       16    uint8_t qs[16] — 4-bit packed quant values (2 per byte)
Total: 18 bytes
```

**Our Metal struct** (l26f_metal.m:1185):
```c
struct block_iq4_nl {
    half d;              // 2 bytes
    uint16_t qs[QK4_NL/2]; // = uint16_t qs[16] = 32 bytes (!)
};
// Total: 34 bytes — WRONG, should be 18
```

`QK4_NL` is defined as 32 in the Metal prefix. So `qs[32/2] = qs[16]` where each
element is `uint16_t` (2 bytes) = 32 bytes for the qs array alone. Total struct
size = 2 + 32 = **34 bytes**.

But the actual GGUF block is only 18 bytes (2 bytes scale + 16 bytes packed data).

### Why Q5_K Works But IQ4_NL Doesn't

The Q5_K struct in our Metal code:
```c
struct block_q5_K {
    // ... fields ...
};
// Total: 176 bytes — matches GGUF Q5_K block size exactly
```

The Q5_K kernel was carefully ported from llama.cpp with the correct block size.
The IQ4_NL kernel was written fresh with the wrong struct. The kernel reads 34
bytes per block instead of 18, so every block reads into the next block's data,
producing garbage scale factors and quant values → NaN.

### What Needs to Change

The `block_iq4_nl` struct must be:
```c
struct block_iq4_nl {
    half  d;            // 2 bytes — scale factor
    uint8_t qs[QK4_NL/2]; // = uint8_t qs[16] — 4-bit packed values, 2 per byte
};
// Total: 18 bytes ✓
```

And then every place that indexes into `qs` must be audited. The current code
treats `qs` as `uint16_t*`, packing two 4-bit values per uint16_t. After the fix,
`qs` is `uint8_t*`, packing two 4-bit values per byte. The bit extraction logic
in the dequant helpers and the matvec kernel must change accordingly.

The easiest reference for the correct IQ4_NL layout is llama.cpp's ggml source:
- `ggml/include/ggml-common.h` — the `block_iq4_nl` struct
- `ggml/src/ggml-metal/ggml-metal.metal` — the Metal kernel

### Files That Need Fixing

| File | What to fix |
|------|------------|
| `l26f_metal.m:1185` | `block_iq4_nl` struct definition in Metal prefix string |
| `metal/l26f_dense.metal` | Dequant helpers + `kernel_mul_mv_iq4_nl_f32` — all qs access |
| `l26f_metal.m:14720` | `row_bytes = blocks_per_row * 18` — already correct at 18 |

---

## Weight Type Distribution in the Model

| GGUF Type | Count | Where Used |
|-----------|-------|------------|
| F32 (0)   | 228   | All norm weights, biases, Q/K norms |
| Q8_0 (8)  | 1     | `token_embd.weight` only |
| Q5_K (13) | 106   | All attn QKV/output, FFN down projections, MoE down |
| Q6_K (14) | 6     | `output.weight` + 5 other tensors |
| IQ4_NL (20)| 199  | FFN gate/up, MoE gate/up, MoE shared experts, some attn |

**Critical observation**: 199 out of 540 tensors are IQ4_NL. This includes ALL MoE
expert gate/up projections and shared expert weights. The IQ4_NL kernel must work
correctly before we can do MoE FFN or even the dense FFN for layers 1-31.

---

## Tensor Map By Layer Type

### Layer 0 — GLA + Dense FFN (11 tensors)

| Tensor | Type | Shape | Status |
|--------|------|-------|--------|
| `attn_norm.weight` | F32 | [4096] | ✓ RMS norm works |
| `attn_qkv.weight` | Q5_K | [4096, 12288] | ✓ Q5_K matvec works |
| `attn_gate.weight` | Q5_K | [4096, 4096] | ✓ Q5_K matvec works |
| `attn_output.weight` | Q5_K | [4096, 4096] | ✓ Q5_K matvec works |
| `attn_q_norm.weight` | F32 | [128] | Not yet used |
| `attn_k_norm.weight` | F32 | [128] | Not yet used |
| `layer_output_norm.weight` | F32 | [4096] | Not yet used (GLA g_norm) |
| `ffn_norm.weight` | F32 | [4096] | ✓ RMS norm works |
| `ffn_gate.weight` | **IQ4_NL** | [4096, 9216] | ✗ NaN — block struct bug |
| `ffn_up.weight` | **IQ4_NL** | [4096, 9216] | ✗ NaN — block struct bug |
| `ffn_down.weight` | Q5_K | [9216, 4096] | Untested (depends on gate/up) |

### GLA + MoE Layers (1-6, 8-14, 16-22, 24-30) — 16 tensors each, 27 layers

Same attention tensors as layer 0 (but also IQ4_NL QKV in some layers?).
FFN has: gate_inp (F32 router), gate_exps/up_exps (IQ4_NL), down_exps (Q5_K),
plus shared expert (gate_shexp/up_shexp IQ4_NL, down_shexp IQ4_NL),
and exp_probs_b bias (F32).

**All expert FFN weights are IQ4_NL** — blocked until the block struct is fixed.

### MLA + MoE Layers (7, 15, 23, 31) — 18 tensors each, 4 layers

MLA attention: q_a, q_b, kv_a_mqa, k_b, v_b, q_a_norm, kv_a_norm.
These use ds4's MLA path. Not yet started.

### Global Tensors

| Tensor | Type | Shape | Status |
|--------|------|-------|--------|
| `token_embd.weight` | Q8_0 | [4096, 157184] | ✓ CPU dequant works |
| `output_norm.weight` | F32 | [4096] | ✓ RMS norm works |
| `output.weight` | Q6_K | [4096, 157184] | Untested |

---

## Multi-Layer Driver (test_l26f_multilayer.c)

A multi-layer inference driver was written but is blocked by the IQ4_NL bug.
The driver structure is:

```
l26f_session
  ├── l26f_compute (reusable per-layer compute buffers)
  ├── l26f_gla_state[32] (persistent GLA recurrent states)
  ├── hidden (current hidden state)
  └── logits (output projection result)

Flow:
  embed_token(token) → hidden
  for layer 0..31:
    if GLA layer: l26f_gla_layer(hidden → hidden)
    if MLA layer: skip (pass-through)
    if layer 0: l26f_dense_ffn(hidden → hidden)
    else: skip MoE FFN (just residual from attention)
  output_logits(hidden → logits)
  argmax(logits → token)
```

The Q8_0 embedding dequant was added (CPU-side, reads Q8_0 block format correctly).
The `ds4_metal_tensor_copy` function is used for tensor-to-tensor copies.
GLA states are zero-initialized via `calloc` + `ds4_metal_tensor_write`.

This file is not yet committed — it's a working draft that will be cleaned up
once the IQ4_NL blocker is resolved.

---

## Bugs Fixed So Far (chronological)

**Bug 1: GGUF tensor_data_pos calculated too early** (commit `141b72a`)

The GGUF file has a header (magic, version, KV pairs, tensor info entries) before
the tensor data section. `tensor_data_pos` marks where data begins. The code
computed it from the cursor position *before* iterating over the 540 tensor info
entries. After the loop, the cursor had advanced past all entries. Fix: compute
`tensor_data_pos` after the loop, aligned to 32 bytes.

**Bug 2: Weight offset had tensor_data_pos subtracted** (commit `b2c4d08`)

`abs_offset = tensor_data_pos + gguf_offset` — already an absolute file position.
The Metal views cover the entire file from byte 0. But test_l26f.c subtracted
`tensor_data_pos` before passing to Metal functions, shifting all reads ~34KB
into garbage. The values at the wrong offset happened to look like valid weights
(same quant block format repeats), so some operations partially worked before
producing NaN. Fix: pass `abs_offset` directly, as ds4 does.

**Bug 3: Token embedding treated as F32 when it's Q8_0** (uncommitted)

The `token_embd.weight` tensor is Q8_0 quantized (34 bytes per 32-element block).
The code read raw bytes as floats — producing values like 10^38. Fix: CPU-side
Q8_0 block dequant (half scale + 32 int8 quants per block).

**Bug 4: block_iq4_nl Metal struct was 34 bytes, should be 18** (commit `ad631b6`)

The struct had `uint16_t qs[QK4_NL/2]` = `uint16_t qs[16]` = 32 bytes for the
quant array. But IQ4_NL packs 32 4-bit values into 16 bytes (`uint8_t qs[16]`).
So the struct was 2+32=34 bytes instead of 2+16=18.

Every IQ4_NL weight read was corrupted: the kernel read 34 bytes per block but
the file stores 18 bytes per block, so each block read spilled into the next
block's scale factor, producing garbage d values and NaN. This affected 199 of
540 tensors in the quality GGUF (all FFN gate/up projections) and 304 of 540
tensors in the original GGUF (including attention QKV).

The IQ4_NL matvec kernel was also rewritten — the old one was a hand-written
kernel with incorrect qs access patterns for the 18-byte block. The new one is
ported from llama.cpp's `kernel_mul_mv_iq4_nl_f32_impl`, which uses shared memory
for the kvalues lookup table and correctly splits `qs` into two 8-byte halves.

---

## Lessons Learned

### 1. Port block structs verbatim from the reference

The IQ4_NL bug arose because the Metal struct was written from memory instead of
copied from llama.cpp's `ggml-common.h`. **Always copy struct definitions
byte-for-byte.** Verify sizeof matches GGUF type_size: 18 for IQ4_NL, 176 for
Q5_K, 210 for Q6_K, 34 for Q8_0.

### 2. `abs_offset` is absolute — never adjust for Metal

`abs_offset = tensor_data_pos + gguf_offset`. Metal views cover entire file from
byte 0. Pass `abs_offset` directly. Never subtract `tensor_data_pos`.

### 3. Match ds4's calling conventions exactly

If ds4 passes X to a function, we pass X. No "adapting" or "adjusting".

### 4. NaN ≈ wrong data being read

Before debugging kernel math, verify offsets and data at those offsets.

### 5. Two offset spaces — don't confuse them

- **File offset**: absolute position in the mmap'd file (what `abs_offset` gives)
- **GGUF offset**: position relative to tensor data section
- Metal uses file offsets. `abs_offset` converts GGUF → file.

### 6. Quant type matters for every weight

The model mixes F32, Q8_0, Q5_K, Q6_K, and IQ4_NL. Always check `tensor->type`.

### 7. When rewriting a kernel, don't break other kernels

The first attempt at fixing l26f_dense.metal replaced the Q5_K and Q6_K kernels
with incorrect scalar-loop versions, breaking the working attention path. Only
modify what needs fixing.

---

## GGUF Files

Two distinct files (the "fixed" one is a copy of the 20260505 one, verified by `cmp`):

| File | Size | Q5_K | IQ4_NL | Notes |
|------|------|------|--------|-------|
| `.../Ling-2.6-flash-IQ4_NL-bailing_hybrid-20260505-LJ.gguf` | 57 GB | 7 | 304 | Original. attn QKV is IQ4_NL. |
| `.../Ling-2.6-flash-IQ4_NL-quality-bailing_hybrid-20260508-LJ.gguf` | 58 GB | 106 | 199 | Re-quantized. attn QKV is Q5_K. |

Both are on NVME at `~/NVME_4TB_SSD_GRAUGEAR_Users_ljubomir/llama.cpp/bailing-hybrid/`.

The quality GGUF masks the IQ4_NL bug for attention (Q5_K attn weights) but still
needs it for FFN. The old GGUF hits IQ4_NL even in the attention path. Either way,
the IQ4_NL block struct fix is required.

We're using the quality GGUF for now (attn path works with Q5_K).

### ~~P0: Fix IQ4_NL block struct~~ DONE (commit `ad631b6`)

### P1: MoE FFN (current blocker for layers 1-31)

With IQ4_NL working, the multi-layer driver can run:
- Layer 0: GLA + dense FFN (all kernels working)
- Layers 1-6: GLA + MoE FFN (need MoE routing + expert matvec)
- MLA layers 7,15,23,31: skip initially or implement MLA

### P2: MoE Expert Routing + Matvec

- Expert router: `ffn_gate_inp.weight` (F32, [4096, 256]) → softmax → top-k
- Expert matvec: 256 experts, 8 active, IQ4_NL weights
- Shared expert: gate_shexp + up_shexp (IQ4_NL) + down_shexp (IQ4_NL)
- ds4 has `ds4_metal_routed_moe_one_tensor` but it's Q8_0-only

### P3: MLA Attention (layers 7, 15, 23, 31)

4 global attention layers. ds4 has MLA kernels — can likely reuse directly.

### P4: Tokenizer + Sampling + CLI

BPE (157184 vocab), temperature/top-k/top-p, CLI interface.

---

## Key Files

| File | Role |
|------|------|
| `l26f_gguf.c:156-193` | Tensor parsing with abs_offset calculation |
| `l26f.h` | Data structures, GGUF types (block_size/type_size table) |
| `l26f_metal.m:1185` | `block_iq4_nl` struct in Metal prefix (fixed: uint8_t qs[16]) |
| `l26f_metal.m:14706` | IQ4_NL matvec dispatch |
| `l26f_metal.m:14842` | Generic quantized matvec dispatch router |
| `l26f_metal.m:14592` | GLA dispatch (blit before commit) |
| `l26f_metal.m:4385` | Model map view setup |
| `metal/l26f_dense.metal:5-29` | IQ4_NL dequant helpers (depend on block struct) |
| `metal/l26f_dense.metal:95-152` | `kernel_mul_mv_iq4_nl_f32` (depends on block struct) |
| `metal/l26f_gla.metal` | GLA kernel |
| `metal/norm.metal` | RMS Norm (from ds4, working) |
| `metal/glu.metal` | SwiGLU (from ds4, working) |
| `test_l26f.c` | Single-layer GLA test (works) |
| `test_l26f_multilayer.c` | Multi-layer driver (uncommitted, blocked by IQ4_NL) |

## Build & Run

```bash
cd ~/llama.cpp/contrib/l26f
make                                    # build all
./test_l26f <model.gguf>                # single-layer test (works)
./test_l26f_multilayer <model.gguf> 1   # multi-layer test (NaN from IQ4_NL)
```
