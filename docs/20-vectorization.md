# Vectorization in GPU Programming

## The MATLAB Analogy

If you come from MATLAB/NumPy, you already know the rule: **vectorized operations
are fast, scalar loops are slow.** A matrix multiply `A * B` is 100x faster than
the same computation written as nested `for` loops, because the vectorized call
dispatches to BLAS which uses SIMD, cache tiling, and parallelism.

GPU programming is the same game, just more extreme.

## What "Vectorized" Means on GPU

| Level | What | Speed | Example |
|-------|------|-------|---------|
| Scalar loop | T×K separate kernel launches | Slow | `for t: for k: matvec(token, expert)` |
| Batch kernel | One kernel launch, many work items | Fast | `mul_mm_id(all_tokens, all_experts)` |
| Tiled SGEMM | simdgroup 8×8 matrix ops in shared memory | Fastest | `simdgroup_multiply_accumulate()` |

The difference between "scalar loop" and "tiled SGEMM" for 2 tokens × 8 experts:
- Scalar: 16 kernel launches, each processes 1 vector, no shared-memory reuse
- SGEMM: 1 kernel launch, 256 experts in parallel, 64×32 tiles in shared memory

For T=64 the scalar approach would be 512 kernel launches per layer × 31 layers = ~16,000 launches.
The SGEMM does 31 launches. Same math, 500x fewer GPU round-trips.

## The MoE Prefill Example

In this codebase we had exactly this pattern:

### Slow (scalar fallback)
```
for each token t:
    for each selected expert k:
        matvec(mid[t,k], down_weight[expert_id])   // 1 kernel launch per (t,k)
```
Each `matvec` is a single GPU dispatch that processes one 1024→4096 multiply.
The GPU sits mostly idle waiting for the next launch.

### Fast (tiled SGEMM)
```
mul_mm_id(mid_all, down_weights_all)   // 1 kernel launch for all tokens × experts
```
One kernel launch. Inside: expert indirection map routes each work group to
the right weight slice. Shared-memory tiles amortize weight loads across
multiple output rows. SIMD matrix operations do 8×8 multiplies per instruction.

The tiled SGEMM kernel (`kernel_l26f_mul_mm_id_*`) uses:
- 64×32 output tiles (NR0=64, NR1=32)
- simdgroup_half8x8 for dequantized weight tiles (shared memory)
- simdgroup_float8x8 for activation tiles
- 4×2 arrangement of simdgroup matrices per threadgroup

## Quant Type Dispatch

The tiled SGEMM structure is identical for all quant types — only the
dequantization function changes. We have one kernel per quant type:

| Kernel | Quant type | Block size | Block bytes | `nl` |
|--------|-----------|------------|-------------|------|
| `kernel_l26f_mul_mm_id_iq4_nl_f32` | IQ4_NL (type=20) | 32 | 18 | 2 |
| `kernel_l26f_mul_mm_id_q5_k_f32`   | Q5_K (type=13)   | 256 | 176 | 8 |

The host dispatch function computes strides from the quant type's block_size
and type_size. The kernel body is copy-pasted with only the dequant function
and `nl` constant changed.

To add a new quant type (e.g. Q6_K): copy the kernel, change the dequant
function and `nl`, add host dispatch with correct strides. 15 minutes of work.

## Rules

1. **Never do per-element GPU dispatch.** If you're writing `for i: kernel(data[i])`,
   you're leaving 90%+ of GPU performance on the table.

2. **Batch first, kernel second.** Structure the data so one kernel launch processes
   all items. The expert indirection map (`kernel_l26f_mul_mm_id_map0`) is the
   pattern: preprocess scatter/gather indices, then one batched kernel.

3. **Tile for shared memory.** Inside the kernel, load weight tiles into shared
   memory once, reuse across multiple output rows. The 64×32 tiling in our SGEMM
   loads each weight tile once but uses it for 64 output elements.

4. **Match quant type to kernel.** The kernel bakes in the dequant function.
   Check the tensor type at dispatch time and route to the correct kernel.
   A type mismatch produces NaN (garbage stride calculations, wrong dequant).

5. **The MATLAB instinct is right.** If you find yourself writing a loop that
   launches GPU kernels, stop. Ask: "How would I write this as a single
   vectorized operation?" Then write that kernel.
