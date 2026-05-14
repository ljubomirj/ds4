# Status 19: Correctness Harness and GLA Call Order

**Date**: 2026-05-11

## Summary

The speed work now has a small regression scaffold before further kernel changes.
It records fixed-token outputs and tensor summaries for hidden/logits, while
keeping the debug path visually separate from inference code:

- `XLOG_REG((...))` for token/tensor regression output
- `XLOG_REG_LAYER((...))` for per-layer hidden summaries
- `L26F_XLOG_LEVEL=0` compiles the regression print sites out
- runtime gates remain `L26F_REGRESSION=1` and `L26F_REGRESSION_LAYERS=1`

During the first regression runs, the GLA driver call-site was found to pass
`q, k, v` into `l26f_metal_gla()`. The Metal entrypoint expects `k, v, q, g`.
That call-site is now fixed.

## Correctness Checks

Model:
`/Users/ljubomir/llama.cpp/models/Ling-2.6-flash-IQ4_NL-quality-bailing_hybrid-20260508-LJ.gguf`

Smoke command:

```sh
DS4_METAL_EXACT_MODEL_VIEWS=1 L26F_BATCH_LAYERS=8 L26F_REGRESSION=1 \
  ./test_l26f_multilayer ... 1 1 0
```

Result:

```text
REG tensor=hidden_after_forward_1xN sum=-39046.8867 min=-12179.1807 max=2321.12109 nans=0
REG tensor=logits_1xV sum=-566444.438 min=-21.8172226 max=14.5630951 nans=0
REG token_index=0 input=1 next=118108 text="ï¼ĮæłĩåĩĨ"
```

No NaNs were reported in hidden/logits.

## Benchmark

Command:

```sh
DS4_METAL_EXACT_MODEL_VIEWS=1 L26F_BATCH_LAYERS=8 \
  ./test_l26f_multilayer ... 1 4 0
```

Result:

```text
tokens: 118108, 39226, 19651, 95731
text: ï¼ĮæłĩåĩĨĠfuckingpop+v
time: 4 tokens in 1.065s = 3.8 tok/s
```

This is still an exact-view diagnostic path, not the target performance path.
The 45 tok/s comparison target needs the fast model-view path plus fused kernels.

## Unsafe Range-View Finding

An attempted bounded model-view optimization using 64 MiB chunks produced a
different token sequence than exact page-range views. That knob is now named
`DS4_METAL_UNSAFE_RANGE_VIEW_MB` and should not be used for benchmark claims
until it passes fixed-token regression.

## Next

1. Run `L26F_DBG_COMPARE` on at least one MoE layer after the GLA fix.
2. Re-test default large shared views on a clean-memory run.
3. If large views still OOM, implement a correctness-checked bounded view cache.
4. Make fused MoE exact-view-aware by wrapping only the selected expert span.

## 2026-05-11 Follow-Up

`L26F_DBG_COMPARE` exposed a fused MoE host-dispatch bug: the fused IQ4_NL
kernel uses dynamic threadgroup memory for its lookup table, but the host did
not allocate that threadgroup memory. After adding:

```text
setThreadgroupMemoryLength:32 * sizeof(float)
```

the layer-1 compare matched:

```text
gate_8xM max_diff=0
up_8xM   max_diff=0
mid_8xM  max_diff=0
down_8xN max_diff=0
final    max_diff=0.000001
```

The driver now maps only the tensor-data byte range with
`ds4_metal_set_model_map_range()`. On this GGUF:

```text
offset: 6.21 MiB
bytes:  58937.94 MiB
largest tensor: 704.00 MiB, blk.1.ffn_down_exps.weight
```

With tensor-range mapping, the two large shared model views no longer OOM when
residency/warmup are enabled.

Benchmarks:

| Mode | Command shape | Result |
|---|---|---:|
| Exact views, non-fused | `DS4_METAL_EXACT_MODEL_VIEWS=1 L26F_BATCH_LAYERS=32`, 4 tok | 3.6 tok/s |
| Large views, non-fused | `L26F_BATCH_LAYERS=32`, 4 tok | 4.7 tok/s |
| Large views, non-fused | `L26F_BATCH_LAYERS=32`, 16 tok | 6.2 tok/s |
| Large views, hybrid MoE | `L26F_HYBRID_MOE=1 L26F_BATCH_LAYERS=32`, 16 tok | 6.4 tok/s |
| Large views, full fused MoE | `L26F_FUSED_MOE=1 L26F_BATCH_LAYERS=32`, 4 tok | 2.1 tok/s |
| Exact views, hybrid MoE | `DS4_METAL_EXACT_MODEL_VIEWS=1 L26F_HYBRID_MOE=1`, 16 tok | 0.8 tok/s |

The full fused path is correct but slower because the fused Q5_K down kernel is
naive compared with the existing optimized Q5_K matvec. The hybrid path keeps
the optimized Q5_K down matvec and fuses only IQ4_NL gate/up plus SwiGLU.

The next high-leverage speed work is to profile kernel time per MoE and MLA
stage, then either optimize the fused Q5_K down kernel or remove the CPU sync
used to read selected experts.

## 2026-05-11 Stage Profile and IQ4_NL NR4 Probe

Added `L26F_PROFILE=1` / `L26F_PROFILE_SYNC=1` profiling scaffolding and
`L26F_PROFILE_GLA=1` for deeper GLA timing. With synchronized stage timing,
the first MLA call pays a one-time cold setup cost, but steady-state decode is
dominated by GLA IQ4_NL matvecs and MoE:

```text
GLA layer, steady typical:
  qkv  ~1.13-1.16 ms
  gate ~0.82-0.84 ms
  proj ~0.81-0.88 ms
  attn ~0.18-0.22 ms
```

The recurrent GLA kernel is not the bottleneck. The next speed target is the
three IQ4_NL decode matvecs per GLA layer, followed by MoE down projection.

Tried an opt-in `L26F_IQ4_NL_NR4=1` path that computes four output rows per
threadgroup instead of two. It initially diverged at token 4 because the
four-row accumulator initialization was not explicit enough; after fixing that,
the 16-token regression sequence matched the default path:

```text
118108, 39226, 19651, 95731, 47484, 83165, 103781, 103781,
29582, 1, 83165, 39159, 123817, 39159, 39159, 0
```

Benchmarks:

| Mode | Command shape | Result |
|---|---|---:|
| Large views, default NR2 | `L26F_BATCH_LAYERS=32`, 16 tok | 6.2 tok/s |
| Large views, NR4 probe | `L26F_IQ4_NL_NR4=1 L26F_BATCH_LAYERS=32`, 16 tok | 6.0 tok/s |
| Large views, NR4 + hybrid MoE | `L26F_IQ4_NL_NR4=1 L26F_HYBRID_MOE=1 L26F_BATCH_LAYERS=32`, 16 tok | 6.4 tok/s |

Conclusion: NR4 is correctness-checked but not faster on this workload. Keep it
as an opt-in probe only. Do not make it default.

Next recommended implementation target: a correctness-checked IQ4_NL
multi-output dispatch for GLA that combines adjacent QKV/gate/proj work where
possible, or a better Q5_K fused-down kernel for MoE. Simple row grouping alone
does not move throughput.

## 2026-05-11 r2 llama.cpp Comparison

The working reference is:

```text
~/llama.cpp/worktrees/LJ-Ling-2.6-flash-r2
```

Its `README.HF` records the same model family running on M2 Max at:

```text
Prompt:     96.1 t/s
Generation: 33.3 t/s

batched-bench eval: 896 runs, 29.44 ms/token, 33.97 t/s
```

The relevant graph is `src/models/bailing-hybrid.cpp`. For GLA layers, the
llama.cpp path does more than l26f currently does:

```text
attn_norm
qkv matmul
Q RMS norm + K RMS norm
RoPE on Q and K
ggml_gated_linear_attn(K, V, Q, g_full, recurrent_state)
post-GLA group RMS norm
layer_out_norm multiply
attn_gate matmul + sigmoid
gla_out * sigmoid(gate)
attn_output matmul
residual
```

l26f currently does:

```text
attn_norm
qkv matmul
attn_gate matmul
gated linear attention using attn_gate as decay/gate input
attn_output matmul
residual
```

So l26f's GLA block is not just slower; it is not yet the same computation as
the fast llama.cpp branch. Correctness work should now use the r2 graph as the
source of truth.

One small Metal-kernel delta was tested: llama.cpp uses `N_SG_IQ4_NL=2` for
`kernel_mul_mv_iq4_nl_f32`, i.e. two simdgroups per threadgroup. l26f now has
an opt-in `L26F_IQ4_NL_NSG2=1` mode matching that row mapping. It passes the
16-token regression sequence, but does not materially improve throughput:

| Mode | Command shape | Result |
|---|---|---:|
| l26f default | `L26F_BATCH_LAYERS=32`, 16 tok | ~6.2 tok/s |
| l26f IQ4_NL NSG2 | `L26F_IQ4_NL_NSG2=1 L26F_BATCH_LAYERS=32`, 16 tok | ~6.1-6.2 tok/s |

Conclusion: the 5x gap is not solved by the single IQ4_NL row-mapping knob.
Next work should be either:

1. make l26f GLA match the r2 graph exactly, then re-check token sequence; or
2. port the r2/ggml execution strategy for the hot graph pieces instead of
   continuing one-off kernel tweaks.

## 2026-05-12 r2 GLA Port

Ported the GLA layer order from
`~/llama.cpp/worktrees/LJ-Ling-2.6-flash-r2/src/models/bailing-hybrid.cpp`:

```text
attn_norm
qkv matmul
Q/K per-head RMSNorm
NeoX RoPE on Q/K
precomputed r2 slope tensor into recurrent GLA
post-GLA 4-group RMSNorm
layer_output_norm multiply
attn_gate sigmoid
gated GLA output
attn_output matmul
residual
```

Two small scaffolding fusions were added to avoid leaving the correctness port
as many tiny kernels:

- `kernel_l26f_gla_qk_norm_rope`: Q/K RMSNorm + NeoX RoPE.
- `kernel_l26f_gla_epilogue`: group RMSNorm + `layer_output_norm` + sigmoid gate.

Validation:

| Check | Result |
|---|---|
| Normal build | `make test_l26f_multilayer` succeeds |
| 4-token smoke | no NaNs/crash, token sequence stable: `126887, 34661, 1, 96` before epilogue fusion and after |
| 16-token benchmark | `./test_l26f_multilayer ... 1 16 0` => 5.9-6.0 tok/s after scratch cleanup |
| MoE compare | separate `-DL26F_DBG_COMPARE` binary, layer 1, `gate/up/mid/down/final` all `max_diff=0` |

This is an important correctness alignment, but not yet a speed win. The
post-port steady speed remains close to the prior large-view baseline. The
remaining 5x gap to r2 is now more clearly in the execution/model-buffer
strategy and hot IQ4_NL/logits matvec path rather than the recurrent GLA
kernel itself.

Next target: port or approximate llama.cpp Metal's weight residency strategy.
l26f still runs matvecs directly from mmap-backed shared model views; r2 keeps
its graph weights in the ggml Metal backend's buffer scheme. A correctness-
checked bounded tensor weight cache, or an explicit copied Metal buffer mode
with fixed-token regression, is the next likely lever.
