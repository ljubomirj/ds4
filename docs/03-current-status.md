# l26f — Current Status & Remaining Blocker

**Date**: 2026-05-08
**Branch**: `l26f` (ljubomirj/l26f fork of antirez/ds4)
**Model**: Ling-2.6-flash IQ4_NL quality GGUF (58 GB)

---

## What Works

### 1. GGUF Model Loader (`l26f_gguf.c`)

Parses GGUF v3, reads metadata KV pairs (extracting model params), enumerates all
540 tensors with correct names, shapes, types, and file offsets. Verified against
the Python `gguf` library — offsets match exactly.

Key bug that was fixed: `tensor_data_pos` was computed from the position BEFORE
tensor info entries, but must be computed AFTER all entries plus alignment
padding. The fix (commit `141b72a`) computes `tensor_data_pos` as the aligned
position after parsing all tensor entries:

```c
// CORRECT: after loop, align to 32 bytes
uint64_t pos = (uint64_t)(c->ptr - m->map);
m->tensor_data_pos = (pos + 31) & ~31;
// Then: abs_offset = tensor_data_pos + gguf_offset
```

Before the fix, offsets were off by ~34KB, causing all kernel dispatches to read
garbage weight data.

### 2. Metal Shader Compilation

All kernels compile into a single Metal library:
- 18 ds4 metal files (dense, moe, norms, unary, glu, bin, cpy, etc.)
- `metal/l26f_gla.metal` — GLA kernel ported from llama.cpp
- `metal/l26f_dense.metal` — IQ4_NL/Q5_K/Q6_K dequant helpers + matvec kernels

Block struct definitions for IQ4_NL, Q5_K, Q6_K live in the Metal source prefix
(C string in `l26f_metal.m`). The IQ4_NL lookup table (`kvalues_iq4nl_f[16]`)
is also in the prefix.

### 3. Metal Runtime

- `ds4_metal_init()` succeeds
- Model map wrapping: 2 overlapping MTLBuffer views covering 58 GiB. Created via
  `newBufferWithBytesNoCopy` from the mmap'd file. Residency takes 1-16 seconds.
- Tensor allocation works (`ds4_metal_tensor_alloc`, `ds4_metal_tensor_view`)

### 4. GLA Kernel (Isolated Test)

The GLA kernel (`kernel_gla_impl<NSG>`) was tested with random Q/K/V/G inputs
(no weight lookup, no matvec) and produces correct non-NaN output:

```
Input: 4096 random floats
GLA output sum: 1686.548, 0 NaNs
State sum: 129411.375, 0 NaNs
```

The kernel writes activations + final state to a single output buffer, then a
blit encoder copies the state back to the state buffer. This was verified to
work correctly after fixing a command buffer ordering bug (blit was after commit;
must be before commit).

### 5. Q5_K Matvec Dispatch

The dispatch code routes to the correct kernel. The `l26f_metal_matvec_quant()`
function selects between ds4's Q8_0/F16 matmul and our Q5_K/Q6_K/IQ4_NL kernels.
Dispatch returns 1 (success) for qkv and gate projections.

### 6. Reusable ds4 Infrastructure

All ds4 Metal kernels are available: RMS norm, SiLU, SwiGLU, element-wise ops,
embedding lookup, softmax, RoPE, concat, argsort, set_rows, sum_rows, etc.

---

## The Remaining Blocker: NaN in RMS Norm

### Symptom

After RMS norm on a valid 4096-element input (no NaNs, values in [0, 1)), the
output has ~42 NaN values (out of 4096, about 1%). The NaN positions are not
contiguous or obviously patterned.

The input is deterministic:
```c
for (i = 0; i < 4096; i++)
    x[i] = (float)((i * 7 + 13) % 1000) / 1000.0f;  // values in [0, 1)
```

A CPU simulation of the same calculation produces zero NaNs:
```c
ss = sum(x[i] * x[i])
rms = sqrt(ss / n + 1e-6f)
out[i] = w[i] * x[i] / rms  // all valid, sum = 1413.9
```

### Dispatch Path

The RMS norm goes through:
1. `ds4_metal_rms_norm_weight_tensor(out, inp, model_map, model_size, weight_offset, n, eps)`
2. → calls `ds4_metal_rms_norm_weight_rows_tensor` with `rows=1`
3. → finds weight MTLBuffer via `ds4_metal_wrap_model_range(model_map, model_size, weight_offset, row_bytes, &inner_offset)`
4. → dispatches `kernel_rms_norm_mul_f32_4` (weighted RMS norm, float4 vectorized)
5. → grid: `(1, 1, 1)`, threads: `(ds4_metal_rms_norm_threads(n), 1, 1)`

The weight is `blk.0.attn_norm.weight`: F32, 4096 elements, at file offset
1223754240 from tensor data start (verified against Python GGUF reader).

### Weight Data Verification

The raw weight data at the correct offset appears reasonable:
```
First 10 floats: 0.342 0.307 0.434 0.320 0.369 0.395 0.379 0.291 0.334 0.377
```

These are typical LayerNorm weight values (in the 0.2-0.5 range, all positive).

### Theories for NaN Origin

**Theory A: MTLBuffer view reads wrong data**

The `ds4_metal_wrap_model_range` function finds which view covers the requested
range and returns the MTLBuffer + inner offset. For a weight at file offset
1223754240:

- View 0 covers offset [0, 57982058496) — 0 to 54 GiB
- View 1 covers offset [57277399040, 61807443968) — 53.3 GiB to 57.6 GiB

The weight at 1223754240 (1.14 GiB) falls in View 0. The inner offset should be
1223754240. The kernel accesses `wbuf + inner_offset`. If the inner offset is
wrong, the kernel reads from the wrong position in the buffer.

**Theory B: Kernel reads Out-of-Bounds**

The RMS norm weight kernel (`kernel_rms_norm_mul_f32_4`) is a ds4 kernel that
reads weight values. If it reads past the end of the weight buffer, it might hit
unmapped memory or read NaNs. The weight buffer is 16384 bytes (4096 * 4). The
kernel processes 4096 elements with float4 vectorization, which should be safe.

**Theory C: Residency Set Bug**

The ds4 Metal residency set is registered after model map wrapping. If the kernel
accesses a weight page that hasn't been faulted in yet, it might get NaN. The
residency set should prevent this, but there might be an edge case.

**Theory D: Q5_K Kernel Produces NaNs That Cascade**

The earlier test showed the GLA kernel works with random inputs but the full
pipeline fails. The Q5_K matvec kernel might produce NaN for some weight values
that then cascade through GLA. But the isolated RMS norm test also shows NaN,
so this can't be the sole cause.

### How to Reproduce

```bash
cd ~/llama.cpp/contrib/l26f
make test_l26f
./test_l26f ~/llama.cpp/models/Ling-2.6-flash-IQ4_NL-quality-bailing_hybrid-20260508-LJ.gguf
```

The `test_l26f.c` file runs a full GLA layer (RMS norm → QKV matvec → Gate matvec → GLA → Output matvec → Add) and prints the output.

### Key Files

| File | Role |
|------|------|
| `l26f_gguf.c:156-193` | Tensor parsing with abs_offset calculation |
| `l26f_metal.m:14558-14695` | GLA dispatch (blit before commit) |
| `l26f_metal.m:4385-4410` | Model map view setup |
| `metal/l26f_gla.metal` | GLA kernel |
| `metal/l26f_dense.metal` | IQ4_NL/Q5_K/Q6_K matvec + dequant helpers |
| `metal/norm.metal` | RMS Norm kernels (from ds4, unmodified) |
| `test_l26f.c` | End-to-end GLA layer test |
| `test_l26f_standalone.c` | (from /tmp) Isolated GLA kernel test |

### Build

```bash
cd ~/llama.cpp/contrib/l26f
make
# Produces: test_l26f (binary)
```

### Next Debugging Steps

1. **Verify RMS norm weight in Metal**: Write a test that copies the weight from
   the MTLBuffer back to CPU and compares byte-by-byte with the mmap data.

2. **Test RMS norm with small buffers**: Try n=32 or n=64 to narrow down if it's
   a size-dependent issue.

3. **Test RMS norm without weight** (plain RMS norm): Use
   `ds4_metal_rms_norm_plain_tensor` which doesn't load weights, to isolate
   whether the weight reading is the issue.

4. **Check Metal residency**: Add env var `DS4_METAL_NO_RESIDENCY=1` to skip
   residency and test if the issue is residency-related.

5. **Read back individual tensor data**: After each kernel dispatch, read back
   and inspect the output to find exactly which values become NaN.

6. **lldb**: Attach lldb to get a stack trace on the segfault/NaN.

### Things NOT to Change

- The GGUF loader is working correctly (verified against Python)
- The GLA kernel produces correct output with random inputs
- The model map wrapping works
- The `ds4_metal.m` file is a copy of ds4's — minimal changes only
