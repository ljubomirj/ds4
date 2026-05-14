# 2026-05-14 Status: MLA SIMD Kernel and Next >45 tok/s Target

## Aim

Continue closing the decode gap from `~35 tok/s` toward `>45 tok/s`, using:

```sh
~/llama.cpp/worktrees/LJ-Ling-2.6-flash-r2
```

as the implementation reference.

Primary benchmark:

```sh
L26F_FUSED_MOE=1 ./build_release/test_l26f_multilayer \
  ~/llama.cpp/models/Ling-2.6-flash-IQ4_NL-quality-bailing_hybrid-20260508-LJ.gguf \
  1 64 0
```

Correctness guard remains:

```text
1 -> 126887 -> 34661 -> 1 -> 96
```

## Work Completed

### Deep MLA profiler

Added `L26F_PROFILE_MLA=1` substage profiling in `l26f_mla_gpu.c`.

With `L26F_PROFILE_SYNC=1`, each MLA substage is bracketed by Metal command
completion so the timings represent completed GPU work, not queued dispatches.

Pre-optimization synced MLA profile showed:

```text
v_decomp avg ~= 0.94 ms/layer
absorb   avg ~= 0.37 ms/layer
all other MLA stages mostly ~= 0.15-0.22 ms/layer
```

The hot path was therefore the old `kernel_l26f_batch_iq4_nl_matvec`, which did
one serial thread per output row.

### Concrete SIMD MLA IQ4_NL kernel

Added concrete Metal kernel:

```text
kernel_l26f_batch_iq4_nl_matvec_simd
```

This keeps the same logical operation as the old MLA batch IQ4_NL matvec, but
uses one SIMD group to cooperate on two output rows for one head.

Host wrapper changed:

```text
l26f_metal_batch_iq4_nl_matvec
```

now dispatches:

```text
threadgroups = (ceil(out_rows / 2), n_heads, 1)
threads      = (32, 1, 1)
```

and the new kernel is included in startup validation.

Post-optimization synced MLA profile:

```text
v_decomp avg ~= 0.26 ms/layer
absorb   avg ~= 0.27 ms/layer
```

Correctness preserved:

```text
126887 -> 34661 -> 1 -> 96 -> 53740
```

Steady 64-token benchmark improved:

```text
before MLA SIMD: 35.1 tok/s
after MLA SIMD:  38.5 tok/s
```

### GLA IQ4_NL projection + residual fusion

Added an IQ4_NL single-token matvec residual mode:

```text
l26f_metal_matvec_iq4_nl_residual
```

and used it in GLA only when:

```text
blk.N.attn_output.weight type == IQ4_NL
```

This folds:

```text
attn_proj = attn_output.weight * gated_gla
out       = inp + attn_proj
```

into the IQ4_NL output projection kernel.

Correctness preserved.

Steady 64-token benchmark:

```text
after MLA SIMD:       38.5 tok/s
after GLA residual:   38.8 tok/s
second run:           38.8 tok/s
```

Interpretation: this dispatch removal is correct, but only a small steady-state
win. It is not the main remaining gap.

## Current State

Default fast path:

```sh
L26F_FUSED_MOE=1 ./build_release/test_l26f_multilayer <model> 1 64 0
```

Current release benchmark:

```text
1 prompt + 65 generated tokens
Decode: 65 tokens in 1.674s (38.8 tok/s)
```

This is up from:

```text
~35.1 tok/s after MoE route/accum fusion
~22-23 tok/s before the parallel route kernel
~20-21 tok/s after Q5_K/Q6_K SIMD ports
```

## Next Work

The next likely path to `>45 tok/s` is GLA/MoE command count and graph-level
fusion, not isolated tiny dispatch shaving.

Highest-value candidates:

1. Compare GLA execution shape against r2/llama.cpp Metal for the full graph:
   norm, QKV, gate, Q/K norm+RoPE, recurrent GLA, epilogue, output projection.
2. Investigate whether GLA `qkv` and `gate` can share model wrapping/dispatch
   structure or be scheduled as a combined concrete path.
3. Re-check MoE after MLA speedup: MoE still runs on 28 layers and small per-layer
   overhead adds up, but route/offset/residual micro-fusions are now mostly done.
4. If using synced profiles, treat absolute timings cautiously. They are good
   for ranking stages, but repeated command-buffer breaks exaggerate dispatch-floor
   costs.

