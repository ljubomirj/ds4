# Status: GLA QKV+Gate Fusion

Date: 2026-05-14

## Goal

Test the proposed GLA fusion:

- `attn_qkv.weight`: normed hidden -> Q/K/V (`4096 x 12288`)
- `attn_gate.weight`: same normed hidden -> gate (`4096 x 4096`)

These share `normed_1xN`, so one concrete Metal kernel can compute both outputs
and remove one dispatch per GLA layer.

## What Changed

- Added concrete `kernel_l26f_gla_qkv_gate_iq4_nl_f32`.
- Confirmed the current 20260508 quality GGUF does **not** use IQ4_NL here:
  every GLA `attn_qkv.weight` and `attn_gate.weight` tensor is `q5_k`.
- Added concrete `kernel_l26f_gla_qkv_gate_q5_k_f32`.
- Added host wrappers:
  - `l26f_metal_gla_qkv_gate_iq4_nl`
  - `l26f_metal_gla_qkv_gate_q5_k`
- Wired `l26f_gla_layer()` to choose fused IQ4_NL, fused Q5_K, or the original
  two-dispatch fallback based on tensor type and shape.
- Added env controls:
  - `L26F_GLA_QKV_GATE_FUSE=0` disables this fusion for A/B tests.
  - `L26F_DEBUG_GLA_FUSE=1` prints selected GLA tensor types and fusion path.

## Verification

Build:

```sh
make release
```

Correctness:

```sh
L26F_FUSED_MOE=1 ./build_release/test_l26f_multilayer \
  ~/llama.cpp/models/Ling-2.6-flash-IQ4_NL-quality-bailing_hybrid-20260508-LJ.gguf \
  1 64 0
```

Fixed token prefix preserved:

```text
126887 -> 34661 -> 1 -> 96 -> 53740
```

Deep profile confirms the fused path is active: `gla_gate` no longer appears
when fusion is enabled; its work is included in `gla_qkv`.

## Benchmark Result

Standard 64-token decode:

```text
fusion enabled:  Decode: 65 tokens in 1.685s (38.6 tok/s)
fusion disabled: Decode: 65 tokens in 1.688s (38.5 tok/s)
```

This is correctness-clean but performance-neutral. The dispatch removal does
not translate into a meaningful steady decode gain; the fused Q5_K kernel simply
moves the gate work into the larger QKV stage.

## Conclusion

GLA QKV+gate fusion is not the missing >45 tok/s lever. Keep it guarded for now
because it is concrete, correct, and useful for profiling, but the next speed
work should move to a larger target:

1. Faster Q6_K LM-head + max reduction, because `logits_head` is still around
   2 ms/token in synced profiles.
2. GPU trace comparison against r2 to locate the remaining distributed gap.
3. Broader GLA/MoE graph-level fusion only if trace evidence shows dispatch
   overhead is material in the non-synced path.
