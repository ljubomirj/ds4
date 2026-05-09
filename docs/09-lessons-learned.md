# Lessons Learned — l26f Debugging Sessions

**Date**: 2026-05-09  
**Context**: Ling-2.6-flash Metal inference engine (l26f), built on the ds4 Metal runtime

---

## Background: The itrade Codebase Study

Before the debugging session, we studied the itrade codebase at `~/z/itrade/src` and `quik-lt/xcommon.h`.
This is a ~1000-line header that serves as the engineering backbone for a production trading system.
It codifies 39 conventions for writing robust C code.

### Key Patterns Extracted from itrade

| Pattern | itrade Name | What It Does |
|---------|-------------|--------------|
| Fail-fast checks | `XCHECK`, `XERROR` | Print file:line + message, abort or return |
| Debug assertions | `XASSERT` | Active in debug builds, compiles out in release |
| Compile-time checks | `XASSERT_STATIC` | `_Static_assert` wrapper |
| Trace logging | `XLOG` | Tiered logging (error, warn, info, trace, verbose) |
| Fill-on-alloc | `FILL_ALLOC` | Fill allocated memory with `0xCC` pattern to catch uninitialized reads |
| Direction prefixes | `in_`, `out_`, `io_` | Make data flow visible at call sites |
| Dimension encoding | `_NxS` suffixes | Variable names encode matrix shapes |
| No nested includes | Convention #5 | Headers don't include headers; `.c` files do all including |

### What We Applied

We wrote a manifesto (`docs/07-manifesto.md`) distilling the 12 most applicable rules for l26f.
Of those, two directly contributed to finding the bugs:

1. **Checksum every tensor after every kernel** (manifesto rule #2) — this was the decisive
   technique. We added `l26f_tensor_checksum()` and `l26f_checksum_print()` helpers, then
   instrumented layer 0 to print checksums after every GPU operation. Running twice and
   comparing instantly revealed that the GLA kernel was the first divergent tensor.

2. **Fill-on-alloc** (manifesto rule #1) — we implemented the long-missing
   `ds4_metal_tensor_fill()` and zeroed all compute buffers after allocation. While this
   turned out not to be the root cause of our non-determinism (the GLA race condition was),
   it eliminated a whole class of potential issues and made the debugging cleaner.

### What Was Less Directly Useful

The `XCHECK`/`XASSERT` macro framework (manifesto rule #3) and direction-prefix conventions
(rules #4-5) are excellent engineering practices that we should adopt, but they didn't directly
help find the bugs this session. They're preventive, not diagnostic.

---

## The Three Bugs

### Bug 1: GLA Kernel Race Condition

**Symptom**: Layer 0 output varied between runs. Three consecutive runs gave sums of
`-153.9`, `-319.7`, `-154.5`. This non-determinism propagated through all 32 layers.

**Root cause**: The GLA kernel dispatch at `l26f_metal.m:14690` used a grid of
`(S/nsg, S/4, H*n_seqs) = (32, 32, 32)`. The x-dimension was 32x too large. Each
threadgroup already covers all `S=128` values of the `i` index (32 threads × NSG=4 = 128).
The extra 31 threadgroups computed overlapping but *different* subsets of `i` (because
boundary elements are clamped to zero) and all wrote to the same output location. The
threadgroup whose write was visible last determined the output — a classic
write-after-write race condition dependent on GPU scheduling.

**How we found it**: The checksum infrastructure. We added checksums after every kernel
in layer 0 and ran twice:

```
Pass 1: comp.gla_out = -227.741638
Pass 2: comp.gla_out = -14.214802
```

All upstream tensors (`hidden_in`, `normed`, `qkv`, `gate_out`) were identical between
runs. The divergence started exactly at `gla_out`. This pinpointed the GLA kernel in
seconds. We then read the kernel source and immediately spotted the too-large grid.

**Fix**: Changed `MTLSizeMake(S/nsg, ...)` to `MTLSizeMake(1, ...)`.

**Time spent**: ~10 minutes with checksums. We had previously spent **hours** analyzing
IQ4_NL kernel arithmetic without finding this, because we were looking in the wrong place.

**Lesson**: When output is non-deterministic, checksum every intermediate tensor and
binary-search for the first divergence. Don't guess which kernel is wrong.

---

### Bug 2: MoE Expert Byte Stride (Wrong Quant-Type Parameters)

**Symptom**: Layer 1 MoE experts 0–7 all produced NaN in the down projection, despite
gate/up/mid being clean. Shared expert down (same shape, different data) was always clean.

**Root cause**: The `down_exp_bytes` calculation assumed all MoE weights were IQ4_NL
(block_size=32, type_size=18), but `ffn_down_exps.weight` is actually Q5_K (block_size=256,
type_size=176). The stride per expert was:

```
Wrong:  4096 × (1024/32) × 18 = 2,359,296 bytes
Correct: 4096 × (1024/256) × 176 = 2,883,584 bytes
```

For expert 0 (offset = base + 0), the starting offset was correct. But for experts 1+,
the code read from `base + n × 2,359,296` instead of `base + n × 2,883,584`, landing in
the middle of the preceding expert's data.

**Why shared expert worked**: The shared expert uses a separate tensor (`ffn_down_shexp.weight`)
with its own `abs_offset`. No stride calculation involved — it just reads from the correct
offset directly.

**How we found it**: After fixing the GLA race condition, we added debug prints showing
all tensor dimensions:

```
down_exps: ndim=3 dims=[1024,4096,256] type=13    ← Q5_K, not IQ4_NL!
gate_exps: ndim=3 dims=[4096,1024,256] type=20    ← IQ4_NL
```

The `type=13` (Q5_K) for `down_exps` was the giveaway. The byte calculation was using
IQ4_NL parameters on Q5_K data.

**Fix**: Computed per-expert bytes using the actual tensor type's `block_size` and
`type_size` from the `l26f_types[]` table, rather than hardcoding IQ4_NL parameters.

**Lesson**: Never assume all tensors in a group share the same quantization type. In this
model, gate/up experts are IQ4_NL but down experts are Q5_K. Always derive byte strides
from the actual tensor metadata.

---

### Bug 3: block_q6_K Struct Field Order

**Symptom**: Output logits contained NaN and extreme values (millions, billions). Hidden
state before output projection was clean (0 NaNs). The output projection is the only
place Q6_K is used (the `output.weight` tensor, shape [4096, 157184], type=14).

**Root cause**: The `block_q6_K` Metal struct had fields in the wrong order:

```c
// Our (wrong):     d first
struct block_q6_K { half d; uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; };

// Upstream (correct): d last
struct block_q6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; half d; };
```

The `d` field (super-block scale) was being read from byte offset 0 (which contains
`ql[0..1]`, raw quantization data) instead of byte offset 208. This made every
dequantization produce values scaled by garbage, causing NaN and extreme logits.

**How we found it**: After fixing bugs 1 and 2, all 32 layers produced clean output but
logits still had NaN. We printed the output.weight tensor type (`type=14` = Q6_K) and
compared our `block_q6_K` struct against the upstream definition in
`ggml/src/ggml-common.h:352-357`. The field order mismatch was immediately visible.

**Fix**: Reordered fields to match upstream: `ql, qh, scales, d`.

**Lesson**: This is the *third time* we've hit a quant-block struct ordering bug in this
project (IQ4_NL was first, now Q6_K). The pattern is always the same: copying a struct
from upstream but accidentally reordering fields. The preventative measure from the
manifesto (rule #6: add `_Static_assert` immediately after copying) would have caught
this at compile time. We need to actually implement those assertions.

---

## Debugging Timeline

| Phase | Duration | What Happened |
|-------|----------|---------------|
| Pre-session | ~1hr | Studied itrade `xcommon.h`, wrote 07-manifesto.md |
| Checksum infra | ~15min | Implemented `l26f_tensor_checksum`, `l26f_checksum_print`, `ds4_metal_tensor_fill`. Zeroed all buffers. Instrumented layer 0. |
| Bug 1 (GLA) | ~10min | Ran twice, compared checksums. GLA was first divergence. Read kernel source, spotted grid bug. Fixed. Verified determinism. |
| Bug 2 (stride) | ~15min | Still NaN at layer 1. Added dimension/type debug prints. Saw `type=13` for down_exps. Fixed byte calculation. NaN gone. |
| Bug 3 (Q6_K) | ~10min | Still NaN in logits. Checked output.weight type (Q6_K). Compared struct against upstream. Wrong field order. Fixed. Clean logits. |
| Verification | ~5min | Ran twice, all checksums identical, same token output. |
| **Total** | **~2hr** | Three bugs found and fixed |

Compare this to the *previous* debugging sessions (documented in `docs/06-current-status.md`)
where we spent hours analyzing IQ4_NL kernel arithmetic, checking GGUF offsets against Python,
inspecting buffer sizes — all without finding the bugs. The difference: structured diagnostics
(checksums) vs. speculative investigation.

---

## What Worked

1. **Tensor checksums after every kernel** — the single most effective technique. Binary
   search for the first divergent tensor across two runs. Found the GLA bug instantly.

2. **Printing tensor metadata (ndim, dims, type)** — revealed that `down_exps` is Q5_K,
   not IQ4_NL. Simple debug prints, huge diagnostic value.

3. **Zeroing all buffers** — eliminated uninitialized memory as a variable. Not the root
   cause this time, but removed an entire class of suspects.

4. **Comparing against upstream source** — for the Q6_K bug, diffing our struct against
   `ggml-common.h` was immediate. Should be the first step for any quant-type issue.

5. **Running twice, comparing** — the simplest possible determinism test. We should have
   been doing this from the start.

## What Didn't Work (Earlier Sessions)

1. **Kernel arithmetic analysis** — we spent hours line-by-line checking the IQ4_NL
   dequantization math. It was correct. The bugs were elsewhere.

2. **GGUF offset verification** — we verified offsets against a Python GGUF reader. The
   offsets were correct. The bugs were in how we *used* those offsets.

3. **Assuming "all MoE weights are IQ4_NL"** — we wrote this assumption into comments
   and code. It was wrong. Down experts are Q5_K.

4. **Not running twice early enough** — if we had run the test twice and compared outputs
   on day one, we would have known immediately that the problem was non-determinism, not
   a math bug. The NaN was a red herring; the real issue was that the same input produced
   different outputs each time.

---

## Rules for Future Sessions

### When output is wrong:

1. **Is it deterministic?** Run twice, compare. If different → race condition or
   uninitialized memory. Stop analyzing math.

2. **Where does it diverge?** Checksum every intermediate tensor. Binary search for
   the first bad one.

3. **What changed?** Compare tensor metadata (ndim, dims, type) against what you expect.

### When NaN appears:

1. **Which tensor first?** Checksum chain finds it.
2. **Is the struct field order correct?** Compare against upstream `ggml-common.h`.
3. **Is the byte stride correct?** Use the actual tensor type, not an assumption.
4. **Is the offset correct?** Print it, compare against Python GGUF reader.

### When adding a new quant type:

1. Copy struct definition verbatim from upstream `ggml-common.h`.
2. Add `_Static_assert(sizeof(block_XXX) == N, "...")` immediately.
3. Verify field order matches by reading upstream definition carefully.
4. Compute byte strides from the type table, not hardcoded constants.
5. Write a CPU reference dequant and compare against GPU output before trusting the kernel.

---

## Remaining Work

- **MLA layers (7, 15, 23, 31)**: Currently skipped (pass-through). Need multi-head
  latent attention implementation.
- **Output projection performance**: Q6_K matvec dispatches 157K threads with
  threadgroup size 1. Needs batched dispatch.
- **CPU MoE routing**: The expert selection loop runs on CPU with per-expert
  read-modify-write. Adequate for single-token decode but wasteful.
- **Static asserts**: We still haven't added `_Static_assert` for quant block structs
  in the Metal prefix. This would have caught the Q6_K bug at compile time.
- **CPU reference**: No CPU-side dequantization reference exists for IQ4_NL or Q6_K.
  Writing one would allow element-by-element comparison against GPU output.
