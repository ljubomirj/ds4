# Status 18: Speed Work, Metal OOM, Exact View Fallback

**Date**: 2026-05-11

## Summary

Speed work started from the current `l26f-private` state. The branch already has
GPU MLA enabled by default and a production fused-MoE path behind
`L26F_FUSED_MOE=1`.

On this run, the default two-overlapping-view Metal model map could create the
57.6 GiB shared views, but command execution OOMed before a usable benchmark:

```text
kIOGPUCommandBufferCallbackErrorOutOfMemory
```

Disabling residency/warmup (`DS4_METAL_NO_RESIDENCY=1`) avoided map-time OOM, but
the first layer still OOMed when the full shared views were used.

## Changes

- Added `L26F_BATCH_LAYERS=N` to `test_l26f_multilayer.c`.
  - Default remains whole-token batching (`32` layers).
  - Values `1..31` commit command buffers after that many layers.
- Added `DS4_METAL_EXACT_MODEL_VIEWS=1` to `l26f_metal.m`.
  - Skips the giant overlapping model views.
  - Wraps only the exact page-aligned model byte range needed by each dispatch.
  - Caches exact views by `(model_map, page_offset, view_bytes)`.

## Benchmarks From This Session

Model:
`/Users/ljubomir/llama.cpp/models/Ling-2.6-flash-IQ4_NL-quality-bailing_hybrid-20260508-LJ.gguf`

Commands:

```sh
DS4_METAL_EXACT_MODEL_VIEWS=1 L26F_BATCH_LAYERS=8 ./test_l26f_multilayer ... 1 4 0
DS4_METAL_EXACT_MODEL_VIEWS=1 L26F_FUSED_MOE=1 L26F_BATCH_LAYERS=8 ./test_l26f_multilayer ... 1 4 0
```

Results:

| Mode | Result |
|---|---:|
| Exact views, non-fused MoE, 4 tokens | 2.621s = 1.5 tok/s |
| Exact views, fused MoE, 4 tokens | 87.279s = ~0.05 tok/s |

The fused-MoE path is much slower with exact views because it wraps each whole
256-expert tensor range for the fused gate/up/down calls. The non-fused path only
wraps selected expert ranges and is therefore faster in exact-view mode.

## Next Speed Path

1. Re-test on a clean-memory machine with the default large shared views:
   `DS4_METAL_NO_RESIDENCY=1 L26F_BATCH_LAYERS=1`.
2. If large views run, benchmark:
   - non-fused MoE
   - `L26F_FUSED_MOE=1`
   - `L26F_BATCH_LAYERS=4/8/32`
3. If large views keep OOMing, make fused MoE exact-view-aware by wrapping only
   the min/max selected expert subrange and subtracting the subrange base from
   gathered offsets.
