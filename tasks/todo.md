# l26f speed work - 2026-05-11

- [x] Read current status docs and code paths for MLA/MoE speed work.
- [x] Build `test_l26f_multilayer`.
- [x] Reproduce current benchmark blocker: full-token Metal command batch OOMs on this machine.
- [x] Add command-buffer chunking so short benchmarks can run under current memory pressure.
- [x] Add exact model-view fallback for Metal OOM diagnosis.
- [x] Benchmark exact-view baseline versus fused MoE and record results.

# Proposed next plan - speed and correctness

- [x] Establish a correctness harness before more speed changes.
  - [x] Add a fixed-token regression mode that runs 1-4 tokens and records token IDs plus tensor summaries.
  - [x] Run twice and compare checksums for determinism.
  - [x] Build `L26F_DBG_COMPARE` for one MoE layer and verify per-expert vs fused stage outputs.
- [ ] Restore a fast Metal model-view mode.
  - [x] Retest default overlapping shared views after reducing memory pressure.
  - [ ] If still OOM, replace two 58 GiB overlapping views with a bounded LRU of page-aligned tensor-range views.
  - [ ] Keep `DS4_METAL_EXACT_MODEL_VIEWS=1` as diagnostic fallback, not performance path.
- [ ] Make fused MoE correct and fast.
  - [x] Use docs/17 side-by-side checkpoints for router, selected experts, gate/up/mid/down, accumulated output.
  - [ ] Make fused MoE exact-view-aware by wrapping only the min/max selected expert subrange and rebasing offsets.
  - [x] Benchmark fused vs per-expert with `L26F_BATCH_LAYERS=1/4/8/32`.
- [ ] Profile and optimize MLA.
  - [x] Add synchronized coarse stage profiler for GLA, MLA, MoE, dense, logits, sample.
  - [x] Time each MLA sub-stage: q_a, q_b, kv_a, RoPE, absorption, attention, V-decompress, output projection.
  - [x] Replace the naive MLA batch IQ4_NL matvec with a concrete SIMD-group kernel.
  - [ ] Compare current dedicated MLA kernels against llama.cpp r2 generic composition documented in docs/16.
  - [x] Prioritize 3D IQ4_NL absorption/V-decompress and attention only after correctness is locked.
- [ ] Optimize GLA IQ4_NL decode matvecs.
  - [x] Add deep GLA profiling for norm, qkv, gate, recurrent attention, projection.
  - [x] Test `L26F_IQ4_NL_NR4=1` four-row probe against fixed-token regression.
  - [x] Reject NR4 as default: correct after explicit accumulator init, but not faster.
  - [x] Compare llama.cpp r2 `N_SG_IQ4_NL=2`; keep l26f `L26F_IQ4_NL_NSG2=1` as a probe because it is correctness-safe but not a speed win.
  - [ ] Design a correctness-checked multi-output IQ4_NL dispatch for adjacent GLA matvecs, or prove it is not worth the added complexity.
- [ ] Align l26f correctness with the fast llama.cpp r2 branch.
  - [x] Implement GLA Q/K RMS norm, Q/K RoPE, post-GLA group RMS norm, `layer_out_norm`, sigmoid gate, and gated output ordering from `src/models/bailing-hybrid.cpp`.
  - [ ] Compare l26f per-layer tensor summaries against r2 graph outputs or a small exported trace.
  - [ ] Treat the r2 branch's `README.HF` 33-45 tok/s numbers as the local target, not a generic MLX/llama.cpp estimate.
- [ ] Reduce dispatch count and CPU syncs.
  - [ ] Remove nonessential tokenizer/debug stderr in benchmark runs.
  - [ ] Keep GPU-side routing; avoid CPU readback in production paths.
  - [x] Combine adjacent small kernels only where tensor summaries prove identical results.
  - [ ] Continue only where the dispatch removal is large enough to move steady 64-token decode.
- [x] Record every benchmark in a new docs/19-speed-log.md.
  - [x] Command, env vars, generated tokens, tok/s, memory mode, batch layer count, correctness status.

# Current findings

- [x] Fixed GLA call-site argument order: `l26f_metal_gla()` expects `(k, v, q, g)`, not `(q, k, v, g)`.
- [x] Converted the new regression prints to XLOG-style scaffolding macros (`XLOG_REG`, `XLOG_REG_LAYER`) with `L26F_XLOG_LEVEL` compile-time gating.
- [x] Renamed chunked model-view env to `DS4_METAL_UNSAFE_RANGE_VIEW_MB`; the 64 MiB range-view experiment changed token output and is not a correctness-safe speed path yet.
- [x] Fixed fused IQ4_NL MoE dispatch by allocating its dynamic threadgroup memory.
- [x] Mapped only the tensor-data range through `ds4_metal_set_model_map_range`; large views now run with residency/warmup enabled.
- [x] Added `L26F_HYBRID_MOE=1`: fused IQ4_NL gate/up + fused SwiGLU, optimized per-expert Q5_K down.
- [x] Profiling shows steady GLA time is dominated by IQ4_NL matvecs (`qkv`, `gate`, `proj`), not the recurrent GLA kernel.
- [x] r2 comparison shows l26f's GLA computation is incomplete relative to the known-fast llama.cpp graph.
- [x] Ported the r2 GLA block structure into l26f and fused its Q/K norm+NeoX RoPE and post-GLA epilogue scaffolding into two narrow Metal kernels.
- [x] Post-port 16-token benchmark is correctness-stable and around 5.9-6.0 tok/s; the remaining gap is still the model-buffer/execution strategy and hot IQ4_NL matvec/logits path, not the recurrent GLA kernel.
