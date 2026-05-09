# l26f — Project Status

**Updated**: 2026-05-09 (afternoon — debug session stuck on MoE NaN)  
**Branch**: `l26f` (ljubomirj/l26f fork of antirez/ds4)  
**Model**: Ling-2.6-flash IQ4_NL quality GGUF (58 GB)

---

## The Short Version

We are blocked on a NaN explosion in the multi-layer end-to-end test when hitting
MoE FFN layers (layers 1–32). The root cause is **not** the IQ4_NL kernel itself;
the most important clue is that **layer 0 output is non-deterministic between
runs**, which strongly suggests a race condition or uninitialized-memory read
somewhere in the shared infrastructure (buffer allocations, kernel dispatch, or
the GLA kernel). We chased IQ4_NL micro-details for too long without tracing the
non-determinism back to its source.

---

## What Was Discovered

1. **MoE weights are IQ4_NL, not Q5_K**
   - `blk.*.ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`, and shared-expert
     variants are all type 20 (IQ4_NL).
   - The original code hardcoded `down_exp_bytes` using Q5_K block size (256) and
     type size (176).
   - **Fix applied**: Changed `down_exp_bytes` calculation to use IQ4_NL
     parameters (`in_dim/32 * 18`).

2. **NaN persisted after the offset fix**
   - Layer 1 expert down still shows NaN for ~6–7 of 8 selected experts.
   - Shared expert down (same kernel, same shape [1024,4096] IQ4_NL) is **always
     clean**.
   - Expert gate / up / SwiGLU outputs are clean.

3. **Layer 0 output is NON-DETERMINISTIC between runs**
   - Three back-to-back runs gave layer-0 sums: `-153.9`, `-319.7`, `-154.5`.
   - Selected experts at layer 1 change completely between runs.
   - **This is the most important clue**: the root cause is likely a race
     condition or uninitialized-memory read somewhere in layer 0 (or shared
     infrastructure), not a pure IQ4_NL math bug.

---

## What Was Tried

- ✅ Verified `block_q5_K` struct size (176 bytes) — correct.
- ✅ Verified `block_iq4_nl` struct size (18 bytes) — correct.
- ✅ Fixed `down_exp_bytes` from Q5_K stride to IQ4_NL stride.
- ✅ Added debug prints at layer 1 for gate, up, mid, down, shared expert, and
  `moe_out`.
- ✅ Inspected IQ4_NL single-token matvec kernel
  (`kernel_mul_mv_iq4_nl_f32`) line-by-line against upstream llama.cpp — logic
  matches.
- ✅ Checked buffer allocations (`ds4_metal_tensor_alloc`) — sizes look correct.
- ✅ Confirmed command buffers are waited on before CPU reads
  (`ds4_metal_finish_command_buffer`).
- ❌ Did **not** yet isolate which layer-0 kernel introduces non-determinism.
- ❌ Did **not** verify that `post_attn` and other compute buffers are fully
  written by every kernel (some are allocated but never zeroed).
- ❌ Did **not** run individual kernel unit tests to confirm deterministic
  behaviour.

---

## Current State

- `contrib/l26f/test_l26f_multilayer.c` contains the debug prints (layer 1
  gate/up/mid/down).
- `down_exp_bytes` fix is in place.
- Build succeeds; test runs but layer 0 is non-deterministic and layer 1 down
  produces NaN for most experts.
- **We are stuck** because we chased the IQ4_NL kernel micro-details instead of
  tracing the non-determinism back to its source in layer 0.

---

## Recommended Next Steps

1. **Zero all compute buffers after allocation**
   `ds4_metal_tensor_alloc` uses `MTLResourceStorageModeShared` but does **not**
   zero memory. If any kernel fails to write an element, garbage leaks through.
   Add `memset` / `ds4_metal_tensor_write` zeroing for `post_attn`, `gla_out`,
   `attn_proj`, etc.

2. **Pinpoint the non-deterministic kernel in layer 0**
   After each layer-0 operation, read the tensor and print a checksum (or simply
   `sum`). Compare across two runs. The first operation whose checksum diverges
   is the buggy kernel.
   Operations to instrument:
   - GLA attention output (`post_attn` after `l26f_gla_layer`)
   - Each matvec inside `l26f_gla_layer` (QKV, gate, output proj)
   - Dense FFN gate/up/down matvecs
   - SwiGLU output
   - Residual adds

3. **Check for out-of-bounds writes**
   The IQ4_NL dispatch uses `(out_dim + 1) / 2` threadgroups. For odd
   `out_dim`, the last threadgroup writes to `dst[out_dim]` which is
   one-past-the-end. While current `out_dim` values are even, verify this
   invariant holds for ALL tensors (including intermediate buffers).

4. **Write a minimal IQ4_NL matvec unit test**
   Generate synthetic IQ4_NL weights, run the Metal kernel, compare with CPU
   reference (`dequantize_row_iq4_nl` from `ggml-quants.c`). This definitively
   rules in/out the IQ4_NL kernel once the non-determinism is resolved.

5. **Check GLA kernel dispatch bounds**
   Ensure `l26f_metal_gla` threadgrid covers exactly the output elements and
   does not rely on uninitialized `dst` memory outside the written region.
