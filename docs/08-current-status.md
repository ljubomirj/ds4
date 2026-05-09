# Status 08: Full 32-Layer Inference Working — Three Bugs Fixed

**Date**: 2026-05-09

## Summary

All three blocking bugs are fixed. The 32-layer Ling-2.6-flash inference pipeline now
runs end-to-end, deterministically, producing valid logits from token 1.

**Result**: `token 1 → next_token 124425`, logits clean (e.g., `0.76, -2.26, 3.76, ...`).

---

## Bugs Fixed

### Bug 1: GLA Kernel Race Condition (non-determinism root cause)

**File**: `l26f_metal.m:14690`

The GLA kernel dispatch grid had `S/nsg` threadgroups in the x-dimension, but each
threadgroup already covers all `S=128` values of `i` (32 threads × NSG=4 = 128).
The extra 31 threadgroups computed overlapping-but-different subsets and raced to
write the same output location. Result: output varied between runs.

**Fix**: Changed grid x-dimension from `S/nsg` to `1`.

```objc
// BEFORE: 32 threadgroups racing to same output
dispatchThreadgroups:MTLSizeMake(S/nsg, S/4, H*n_seqs)
// AFTER: 1 threadgroup per (j, head)
dispatchThreadgroups:MTLSizeMake(1, S/4, H*n_seqs)
```

### Bug 2: MoE Expert Byte Stride (NaN in expert down projection)

**File**: `test_l26f_multilayer.c` (down_exp_bytes calculation)

The `down_exp_bytes` calculation used IQ4_NL parameters (block_size=32, type_size=18)
for a Q5_K tensor (block_size=256, type_size=176). This made expert 1+ read from
wrong offsets, producing NaN.

| Parameter | Wrong (IQ4_NL) | Correct (Q5_K) |
|-----------|----------------|-----------------|
| block_size | 32 | 256 |
| type_size | 18 | 176 |
| stride/expert | 2,359,296 | 2,883,584 |

**Fix**: Compute per-expert bytes using actual tensor type from `l26f_types[]`.

### Bug 3: block_q6_K Struct Field Order (NaN in output projection)

**File**: `l26f_metal.m:1206-1211`

The `block_q6_K` Metal struct had `d` as the first field; upstream has it last.
This caused the dequantization scale to be read from `ql[0..1]` instead of the
actual `d` field, producing garbage logits.

```c
// BEFORE (wrong order):
struct block_q6_K { half d; uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; };

// AFTER (matches upstream):
struct block_q6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; half d; };
```

---

## Infrastructure Added

### ds4_metal_tensor_fill (l26f_metal.m, ds4_metal.m)

Implements the declared-but-missing fill function. All compute buffers are now
zeroed after allocation:

```c
int ds4_metal_tensor_fill(ds4_metal_tensor *tensor, float value);
```

### Tensor Checksum Infrastructure (test_l26f_multilayer.c)

Two helpers for pinpointing non-determinism:

```c
float l26f_tensor_checksum(tensor, bytes, &nans);
void l26f_checksum_print(label, tensor, bytes);
```

Layer 0 emits checksums after every kernel, allowing instant identification of
the first divergent tensor across runs.

---

## Remaining Issues

### MLA Layers (7, 15, 23, 31) Not Implemented

These 4 layers pass through unchanged. The model's quality is degraded without
them, but the pipeline doesn't crash.

### Output Projection Performance

Q6_K output matvec dispatches 157,184 threads with threadgroup size 1. This is
functional but very slow. Needs batched or tiled dispatch.

### CPU-Side MoE Routing

The expert routing loop (matvec, softmax, group top-k, accumulation) runs on CPU
with per-expert read-modify-write cycles. Functional for single-token decode but
inefficient.

---

## Verification

Two consecutive runs produce identical checksums and output:

```
Pass 1: token 1 -> next_token 124425  logits: 0.7608 -2.2612 3.7572 ...
Pass 2: token 1 -> next_token 124425  logits: 0.7608 -2.2612 3.7572 ...
```

All 32 layers produce 0 NaNs. Hidden state at output: `sum=-9507.303 nans=0/4096`.

---

## Lessons Reinforced

1. **Checksum early, checksum often.** Adding per-kernel checksums pinpointed the
   GLA race condition in minutes, after hours of kernel-micro-analysis had failed.

2. **Copy struct field order verbatim from upstream.** The IQ4_NL lesson (#1) repeated
   with Q6_K. Add `_Static_assert(sizeof(block_q6_K) == 210, ...)` immediately.

3. **Never assume all MoE experts share the same quant type.** In this model,
   gate/up are IQ4_NL (type 20) but down is Q5_K (type 13). The byte stride
   calculation must use per-tensor type parameters.
