# 2026-05-14 Status: Route Speedup and Next Work Toward >45 tok/s

## Aim

Reach and exceed `>45 tok/s` decode on Ling-2.6-flash in `l26f-private`, matching or beating the known-fast reference branch:

```sh
~/llama.cpp/worktrees/LJ-Ling-2.6-flash-r2
```

Current target worktree:

```sh
~/llama.cpp/contrib/ds4/worktrees/l26f-private
```

Primary benchmark:

```sh
L26F_FUSED_MOE=1 ./build_release/test_l26f_multilayer \
  ~/llama.cpp/models/Ling-2.6-flash-IQ4_NL-quality-bailing_hybrid-20260508-LJ.gguf \
  1 64 0
```

Correctness guard: the first generated tokens must remain:

```text
1 -> 126887 -> 34661 -> 1 -> 96
```

## Tasks Completed

### Decode speed path

- Ported SIMD Q5_K and Q6_K Metal matvec kernels from the r2/ggml-metal style.
- Ported SIMD Q5_K fused MoE down projection.
- Added fused MoE expert gate+up+SwiGLU path for IQ4_NL.
- Added fused shared expert gate+up+SwiGLU path.
- Added fused MoE accumulation.
- Added synchronized stage profiling via `L26F_PROFILE=1 L26F_PROFILE_SYNC=1`.
- Added deep MoE substage profiling via `L26F_PROFILE_MOE=1`.

### Expert-cache experiment

- Added GPU gather for selected experts into contiguous cache buffers.
- Added cached fused MoE dispatch variants.
- Fixed correctness by replacing invalid global cache offsets with per-layer pre-baked cache offset tensors.
- Result: cached path is correct but slow, about `6.1 tok/s` for 64-token decode, because it copies selected expert weights every token.
- Current decision: keep expert cache opt-in only:

```sh
L26F_EXPERT_CACHE=1
```

Do not use it for default speed work.

### Biggest recent win: MoE route kernel

Before the latest fix, the single-token MoE route kernel did the full E=256 softmax, group selection, top-k, normalization, and scaling in one GPU thread.

Deep profile showed `moe_route` was the main MoE outlier:

```text
moe_route avg ~= 1.73 ms/layer under forced sync
```

Fix:

- Replaced `kernel_l26f_moe_route` with a single 256-thread threadgroup.
- Parallelized max reduction, exp, sum reduction, normalization, and group scoring.
- Left final tiny top-group/top-expert selection serial inside thread 0.
- Changed host dispatch from one thread to one 256-threadgroup.

Result:

```text
64-token decode: ~23 tok/s -> 34.5 tok/s
short 5-token decode: 43.5 tok/s
required token sequence preserved
```

Post-fix synchronized whole-stage profile:

```text
moe avg ~= 0.89 ms/layer
gla avg ~= 0.72 ms/layer
mla avg ~= 1.65 ms/layer
dense avg ~= 0.67 ms/layer
```

Short deep MoE profile after route fix:

```text
moe_route   avg ~= 0.53 ms/layer
moe_gate_up avg ~= 0.49 ms/layer
moe_down    avg ~= 0.42 ms/layer
moe_shared  avg ~= 0.33 ms/layer
moe_router  avg ~= 0.20 ms/layer
moe_offsets avg ~= 0.18 ms/layer
moe_accum   avg ~= 0.17 ms/layer
moe_norm    avg ~= 0.18 ms/layer
residual    avg ~= 0.18 ms/layer
```

Important caveat: synchronized profiling is intentionally perturbing, especially with many small dispatches. Use it for ranking, not absolute speed.

## Current State

### Correctness

The fast default path is correct for the fixed-token regression:

```sh
L26F_FUSED_MOE=1 ./build_release/test_l26f_multilayer <model> 1 4 0
```

Observed generated tokens include:

```text
126887 -> 34661 -> 1 -> 96 -> 53740
```

### Performance

Current release benchmark:

```text
1 prompt + 65 generated tokens
Decode before route+offset and residual fusion:
65 tokens in 1.882s (34.5 tok/s)

Decode after route+offset and residual fusion:
65 tokens in 1.850s (35.1 tok/s)
```

This is a major improvement from the previous `~20-23 tok/s`, but still short of the goal of `>45 tok/s`.

### Reference

The r2 branch remains the comparison target:

```sh
~/llama.cpp/worktrees/LJ-Ling-2.6-flash-r2
```

r2 has shown generation in the `~34-44 tok/s` range depending on command and context. The desired target for l26f-private is explicitly `>45 tok/s`, not merely parity with the lower r2 runs.

## What's Next

### 1. Fuse route + offset gather

Done after this note was first written. The default MoE decode now uses a concrete `kernel_l26f_moe_route_offsets` kernel, which combines:

```text
router matvec
moe_route
gather gate/up/down selected expert offsets
```

This replaced:

```text
l26f_metal_moe_route(...)
l26f_metal_gather_offsets_3(...)
```

with:

```text
l26f_metal_moe_route_offsets(...)
```

Result: correctness preserved, but 64-token decode was roughly flat (`34.5 -> 34.0 tok/s` in one run). This removed a dispatch, but that dispatch was not a dominant steady-state wall-clock cost.

### 2. Fold residual into final accumulation

Done after this note was first written. The default MoE tail now uses a concrete `kernel_l26f_fused_accum_residual` kernel.

This replaced:

```text
fused_accum: expert weighted sum + shared expert -> moe_out
add_tensor: hidden + moe_out -> out
```

with:

```text
fused_accum_residual: hidden + weighted expert sum + shared expert -> out
```

Result: correctness preserved, and 64-token decode improved to `35.1 tok/s`.

### 3. Re-profile whole-stage after dispatch fusion

Done after route+offset and residual fusion:

```text
moe  avg ~= 0.75 ms/layer
gla  avg ~= 0.58 ms/layer
mla  avg ~= 1.36 ms/layer
dense avg ~= 0.61 ms/layer
```

Interpretation:

- MoE is no longer a single catastrophic bottleneck.
- MLA has only 4 layers but is still relatively expensive.
- GLA is paid 28 times per token and is now the broad target.
- Further progress to `>45 tok/s` likely needs MLA/GLA graph or kernel work rather than more small MoE dispatch shaving.

Use:

```sh
L26F_FUSED_MOE=1 L26F_PROFILE=1 L26F_PROFILE_SYNC=1 \
  ./build_release/test_l26f_multilayer <model> 1 8 0
```

Then run the normal benchmark:

```sh
L26F_FUSED_MOE=1 ./build_release/test_l26f_multilayer <model> 1 64 0
```

Do not optimize based only on deep MoE profiling because nested sync boundaries distort absolute timings.

### 4. Next: attack MLA then GLA

Post-route profile suggests:

- GLA is still paid 28 times per token.
- MLA is only 4 layers, but each layer is relatively expensive.
- MoE is now near the same order as GLA, so further gains need dispatch count reduction or a larger graph-level change.

Candidate MLA work:

- Compare l26f MLA path against r2 Metal graph.
- Look for CPU fallback, excessive command-buffer boundaries, or missing fused kernels.
- Prioritize anything in MLA that still reads/writes through CPU or uses generic matvec paths where r2 uses a fused Metal op.

Candidate GLA work:

- Re-check current l26f GLA dispatch count versus r2.
- Look for remaining norm/RoPE/gate/proj dispatches that can be fused without changing output.
- Preserve the fixed-token regression after every change.

### 5. Keep prefill separate

Batch prefill work exists, including mat-mat SGEMM infrastructure, but it currently has correctness issues (`token -1` / NaN logits). Do not mix prefill debugging into the decode speed path unless explicitly switching goals.

Decode target remains:

```text
34.5 tok/s -> >45 tok/s
```

## High-Value Rules From This Round

- Any optimization must pass the fixed-token regression before speed numbers matter.
- Runtime Metal source cannot use ggml-style function-pointer template kernels; write concrete kernels.
- Synchronized profile numbers are diagnostic, not benchmark numbers.
- The cache-gather idea is not a win for decode because selected expert weights are too large to copy every token.
- Small E=256 routing still needs parallelism; one GPU thread is too slow when repeated across 31 MoE layers per token.
