# Status: Q6_K LM-head + Argmax Fusion

Date: 2026-05-14

## Goal

Fuse the Q6_K LM-head matvec with the argmax operation for greedy (temp=0)
decode, saving one dispatch and the 614 KB logits buffer write. The synced
profile showed `logits_head` at ~1.58 ms/token, making it the largest single
remaining stage.

## What Changed

- Added concrete Metal kernel `kernel_l26f_logits_head_q6k_argmax` in
  `metal/l26f_dense.metal`. Based on `kernel_mul_mv_q6_K_f32` but:
  - Dispatches 1-D over vocab rows, 4 rows per threadgroup (NR0=2, NSG=2)
  - Tracks per-simdgroup max (value, index), reduces across simdgroups via
    threadgroup memory
  - Writes per-threadgroup (max_val_f32, max_idx_i32) pairs to an
    intermediate buffer instead of the full logits array
- Added host wrapper `l26f_metal_logits_head_q6k_argmax` in `l26f_metal.m`:
  - Dispatches the fused kernel
  - CPU-scans the per-tg intermediate buffer (~39K entries) to find the
    global argmax
  - Returns the token ID directly
- Added `logits_max_pairs` buffer to `l26f_session` (314 KB for 157K vocab)
- Modified `l26f_output_greedy_token` to use the fused path when
  `output.weight` is Q6_K (type 14), with fallback to the original two-step
  path
- Added `L26F_LOGITS_FUSE=0` env var for A/B testing
- Registered kernel in startup validation

## Verification

Build: `make release` — passes, existing warnings only.

Correctness:
```sh
L26F_FUSED_MOE=1 ./build_release/test_l26f_multilayer \
  ~/llama.cpp/models/Ling-2.6-flash-IQ4_NL-quality-bailing_hybrid-20260508-LJ.gguf \
  1 6 0
```
Fixed token prefix preserved: `126887 → 34661 → 1 → 96 → 53740 → 127902 → 127131`.

A/B benchmark (64-token decode):
```text
Fused ON:  65 tokens in 1.729s (37.6 tok/s)
Fused OFF: 65 tokens in 1.721s (37.8 tok/s)
```
Flat within noise.

## Conclusion

The fused head+argmax is correctness-clean but performance-neutral, joining
the pattern of previous dispatch-fusion experiments (GLA residual, logits
batching, GLA QKV+gate). The logits path is primarily limited by Q6_K
dequant-mul-add throughput (~644M FMAs for the LM head), not by dispatch
count or buffer write bandwidth.

The remaining ~5 tok/s gap to r2 (42.85 tok/s) is now distributed across all
32 layers and the logits stage. No single dispatch-removal or micro-fusion
has moved the steady decode benchmark meaningfully since the MoE route
parallelization and MLA SIMD kernel.

## Next Step

The highest-confidence next step is a **Metal GPU trace comparison** between
l26f and r2 for a single decode token. This will show per-kernel GPU times
on both sides and definitively identify where the distributed gap lives.
Without ground-truth per-kernel timing data, further inferred optimizations
risk being flat like this one.
