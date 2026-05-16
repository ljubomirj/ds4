# Status: Metal GPU Trace Capture

Date: 2026-05-14

## Goal

Capture short decode traces for l26f-private and r2 so the remaining ~10% decode
gap can be inspected with Xcode's Metal debugger instead of inferred from
synchronized CPU-side profiling.

## Captures

### l26f-private

Command:

```sh
MTL_CAPTURE_ENABLED=1 \
L26F_GPU_TRACE="1:$PWD/traces/l26f-gen1-20260514-205617.gputrace" \
L26F_FUSED_MOE=1 \
./build_release/test_l26f_multilayer \
  ~/llama.cpp/models/Ling-2.6-flash-IQ4_NL-quality-bailing_hybrid-20260508-LJ.gguf \
  1 4 0
```

Output:

```text
GPU trace capture started (GPUTraceDocument) -> traces/l26f-gen1-20260514-205617.gputrace
GPU trace capture stopped
Decode: 5 tokens in 2.160s (2.3 tok/s)
```

Trace:

```text
traces/l26f-gen1-20260514-205617.gputrace  (~2.3 GiB)
traces/l26f-gen1-20260514-205617.log
```

The token prefix remained correct:

```text
126887 -> 34661 -> 1 -> 96 -> 53740
```

### r2

Command:

```sh
cd ~/llama.cpp/worktrees/LJ-Ling-2.6-flash-r2
MTL_CAPTURE_ENABLED=1 \
GGML_METAL_CAPTURE_COMPUTE=2 \
build/bin/llama-batched-bench \
  -m ~/llama.cpp/models/Ling-2.6-flash-IQ4_NL-quality-bailing_hybrid-20260508-LJ.gguf \
  -npp 1 -ntg 4 -npl 1 -c 512
```

Output:

```text
ggml_metal_graph_compute: capturing graph in /tmp/perf-metal-67188.gputrace
|     1 |      4 |    1 |      5 | ... | S_TG t/s 37.70 |
```

The trace was copied into this worktree:

```text
traces/r2-tg-capture-20260514-205644.gputrace  (~5.3 GiB)
traces/r2-capture-20260514-205644.log
```

`GGML_METAL_CAPTURE_COMPUTE=2` was used so the capture targets the first TG
decode graph rather than the initial PP graph.

## Notes

- `MTL_CAPTURE_ENABLED=1` is required for file-backed `.gputrace` output.
  Without it, l26f falls back to a live Xcode capture and no trace document is
  written.
- `xcrun xctrace export --toc` does not work on these `.gputrace` bundles in
  this environment; it returns `Document Missing Template Error`. Open them in
  Xcode directly.
- `traces/` is intentionally ignored by git because these captures are multi-GB.

## What To Inspect In Xcode

Compare a decode graph in each trace:

1. Total GPU duration for one decode token.
2. Per-kernel duration and counts for:
   - GLA recurrent / gated-delta kernels
   - Q5_K GLA QKV/gate/proj matvecs
   - MoE route/top-k
   - IQ4_NL expert gate/up/down kernels
   - Q6_K LM-head kernel
3. Command-buffer count and gaps between command buffers.
4. Whether r2 overlaps command encoding or has shorter idle gaps between large
   compute dispatches.

The expected outcome is a ranked list of kernels or scheduling gaps accounting
for the remaining ~10% difference: l26f ~39 tok/s vs r2 ~43.5 tok/s on the
same 20260508 model.
