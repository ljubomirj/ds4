# Debugging Guidelines — Differential Path Comparison

**Date**: 2026-05-11  
**Context**: Debugging fused MoE kernel NaNs against known-good per-expert path.

---

## Principle: Run Both Paths, Compare at Every Stage

When you have a **working path** (A) and a **broken path** (B), instrument BOTH
identically and run them in the same process. Compare outputs at every pipeline
stage. The first divergence point IS the bug location.

### Why This Works

- Binary search over pipeline stages: O(log N) comparisons instead of O(N) guessing
- Both paths share the same input data (no "different random seed" confusions)
- NaN/Inf propagation is contagious — the first NaN taints everything downstream
- If values explode to Inf before becoming NaN, you see the explosion point

---

## Rule 1: Debugging Code Is ADDITIONAL, Not Replacing

Debugging instrumentation must be **added alongside** business logic, never
replacing or modifying it. The production code path must remain untouched.

```c
// BAD — modifies control flow
if (L26F_DBG) { /* ... debug stuff ... */ } else { /* ... real code ... */ }

// GOOD — adds alongside, zero effect when disabled
L26F_DBG_CHECKPOINT("gate", expert_gate_buf, 1024);
result = l26f_metal_matvec_quant(...);  // unchanged
L26F_DBG_CHECKPOINT("after_matvec", result_buf, 4096);
```

---

## Rule 2: ALL CAPS for Debug Code

All debugging macros and variables use ALL_CAPS so they are mentally filtered
when reading business logic. See `docs/14-code-style.md` for the full convention.

```c
L26F_DBG_CHECKSUM(...)    // not: dbg_checksum(...)
L26F_DBG_COMPARE(...)     // not: compare_debug(...)
L26F_DBG_ASSERT_EQ(...)   // not: assert_equal(...)
```

---

## Rule 3: Preprocessor ON/OFF, Never Edit-Delete

Use `#define` / `#undef` to control debug output. Never add/remove logging
by editing source — that introduces typos, missing semicolons, and merge conflicts.

### Compile-Time Switches

```c
// Enable with: -DL26F_DBG_FUSED (add to CFLAGS in Makefile)
// Disable by:  omitting the flag (default)

#ifdef L26F_DBG_FUSED
#define L26F_DBG_CHECKPOINT(LABEL, TENSOR, NFLOATS) \
    l26f_dbg_checkpoint((LABEL), (TENSOR), (NFLOATS), __FILE__, __LINE__)
#else
#define L26F_DBG_CHECKPOINT(LABEL, TENSOR, NFLOATS) ((void)0)
#endif
```

### Layered Control

Multiple flags for granular control:

| Flag | What It Enables |
|------|----------------|
| `L26F_DBG_FUSED` | All fused MoE debug checkpoints |
| `L26F_DBG_FUSED_STAGE1` | Stage 1 only (routing + offset gather) |
| `L26F_DBG_FUSED_STAGE2` | Stage 2 only (gate + up matmecs) |
| `L26F_DBG_FUSED_STAGE3` | Stage 3 only (swiglu + down + accumulate) |
| `L26F_DBG_COMPARE` | Side-by-side comparison of fused vs per-expert |

Outer flag enables all; inner flags enable subsets for manageable output.

```c
#if defined(L26F_DBG_FUSED_STAGE1) || defined(L26F_DBG_FUSED)
// ... stage 1 checkpoints ...
#endif
```

---

## Rule 4: Check NaN, Inf (Positive and Negative), and Insane Values

Never check only for NaN. A value can become +Inf or -Inf before propagating
as NaN. Also check for "insane" values that are technically finite but indicate
a problem (e.g., a hidden state value of 1e30 in a model where typical range
is [-10, 10]).

### The L26F_DBG_CLASSIFY Macro

```c
typedef enum {
    L26F_VAL_OK,       // finite, within expected range
    L26F_VAL_LARGE,    // finite but |x| > threshold (e.g., 1e4)
    L26F_VAL_POS_INF,  // +Inf
    L26F_VAL_NEG_INF,  // -Inf
    L26F_VAL_NAN,      // NaN (any sign)
    L26F_VAL_ZERO      // exactly 0.0 (may indicate uninit if unexpected)
} l26f_dbg_val_class;

// Classify a single float value
static l26f_dbg_val_class L26F_DBG_CLASSIFY(float v, float insane_threshold) {
    if (isnan(v))                        return L26F_VAL_NAN;
    if (isinf(v) && v > 0)               return L26F_VAL_POS_INF;
    if (isinf(v) && v < 0)               return L26F_VAL_NEG_INF;
    if (v == 0.0f)                        return L26F_VAL_ZERO;
    if (fabsf(v) > insane_threshold)      return L26F_VAL_LARGE;
    return L26F_VAL_OK;
}
```

### The L26F_DBG_CHECKPOINT Function

```c
static void L26F_DBG_CHECKPOINT_IMPL(
        const char *label,
        const float *data, int n,
        const char *file, int line,
        float insane_threshold)
{
    int n_ok = 0, n_nan = 0, n_pinf = 0, n_ninf = 0, n_large = 0, n_zero = 0;
    float sum = 0.0f, max_abs = 0.0f;
    int first_bad_idx = -1;
    l26f_dbg_val_class first_bad_class = L26F_VAL_OK;

    for (int i = 0; i < n; i++) {
        l26f_dbg_val_class c = L26F_DBG_CLASSIFY(data[i], insane_threshold);
        switch (c) {
            case L26F_VAL_OK:      n_ok++;    sum += data[i]; break;
            case L26F_VAL_NAN:     n_nan++;   break;
            case L26F_VAL_POS_INF: n_pinf++;  break;
            case L26F_VAL_NEG_INF: n_ninf++;  break;
            case L26F_VAL_LARGE:   n_large++; break;
            case L26F_VAL_ZERO:    n_zero++;  sum += data[i]; break;
        }
        float a = fabsf(data[i]);
        if (a > max_abs) max_abs = a;
        if (first_bad_idx < 0 && c != L26F_VAL_OK && c != L26F_VAL_ZERO) {
            first_bad_idx = i;
            first_bad_class = c;
        }
    }

    fprintf(stderr, "DBG %s:%d %-30s n=%d ok=%d nan=%d +inf=%d -inf=%d large=%d zero=%d sum=%.4f max_abs=%.4f",
            file, line, label, n, n_ok, n_nan, n_pinf, n_ninf, n_large, n_zero, sum, max_abs);
    if (first_bad_idx >= 0) {
        fprintf(stderr, " FIRST_BAD=[%d]=%f(%d)", first_bad_idx,
                data[first_bad_idx], (int)first_bad_class);
    }
    fprintf(stderr, "\n");
}
```

---

## Rule 5: Side-by-Side Comparison Pattern

Run BOTH paths (working A and broken B) for the same input, compare at each stage:

```c
// In the MoE function, when L26F_DBG_COMPARE is active:

// Step 1: Both paths share the same RMS norm + routing
L26F_DBG_CHECKPOINT("rms_norm", ffn_normed, n_embd);

// Step 2: Run WORKING path for expert 0
old_per_expert_matvec(gate_buf_old, ...);
L26F_DBG_CHECKPOINT("A_gate_e0", gate_buf_old, n_ff_exp);

// Step 3: Run BROKEN path for expert 0
fused_moe_iq4nl(gate_buf_fused, ...);
L26F_DBG_CHECKPOINT("B_gate_e0", gate_buf_fused, n_ff_exp);

// Step 4: Compare
L26F_DBG_ASSERT_EQ("gate_e0", gate_buf_old, gate_buf_fused, n_ff_exp);
```

### The L26F_DBG_ASSERT_EQ Macro

```c
#ifdef L26F_DBG_COMPARE
#define L26F_DBG_ASSERT_EQ(LABEL, A, B, N) \
    l26f_dbg_assert_eq((LABEL), (A), (B), (N), __FILE__, __LINE__)
#else
#define L26F_DBG_ASSERT_EQ(LABEL, A, B, N) ((void)0)
#endif

static void l26f_dbg_assert_eq(
        const char *label, const float *a, const float *b, int n,
        const char *file, int line)
{
    float max_diff = 0.0f;
    int first_diff_idx = -1;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > max_diff) {
            max_diff = d;
            if (first_diff_idx < 0 && d > 1e-4f) first_diff_idx = i;
        }
    }
    if (max_diff > 1e-4f) {
        fprintf(stderr, "DBG DIFF %s:%d %-30s MAX_DIFF=%.6f first_diff=[%d] a=%.6f b=%.6f\n",
                file, line, label, max_diff,
                first_diff_idx,
                first_diff_idx >= 0 ? a[first_diff_idx] : 0.0f,
                first_diff_idx >= 0 ? b[first_diff_idx] : 0.0f);
    }
}
```

---

## Rule 6: First-Bad-Element Reporting

When a checkpoint finds a problem, report the INDEX and VALUE of the FIRST bad
element. This is often enough to identify the root cause (e.g., "element 43 of
1024 is NaN" — 43 is suspiciously close to expert index 43, suggesting an offset
calculation bug).

---

## Workflow for Debugging the Fused MoE NaN

1. **Add the debug infrastructure** (macros, checkpoint function, assert_eq)
2. **Instrument BOTH paths** at matching pipeline stages
3. **Build with `-DL26F_DBG_FUSED -DL26F_DBG_COMPARE`**
4. **Run 1 token**, compare output
5. **Binary search**: if stage N matches but stage N+1 doesn't, the bug is
   between stages N and N+1
6. **Zoom in**: add more checkpoints around the bad stage
7. **Fix the bug**, verify both paths now match
8. **Remove `-DL26F_DBG_*` flags** — code compiles clean, no debug overhead

### Pipeline Stages for MoE

| Stage | Working Path | Fused Path | Expected Match? |
|-------|-------------|-----------|----------------|
| RMS norm output | Same | Same | Yes (identical) |
| Router logits | Same | Same | Yes |
| Selected indices | Same | Same | Yes |
| Selected weights | Same | Same | Yes |
| Expert 0 gate | per-expert matvec | fused iq4nl | Yes (must match) |
| Expert 0 up | per-expert matvec | fused iq4nl | Yes |
| Expert 0 swiglu | per-element | fused swiglu | Yes |
| Expert 0 down | per-expert matvec | fused q5k | Yes |
| All 8 experts accumulated | axpy loop | fused accum | Yes |
| Shared expert | same code | same code | Yes |
| Final moe_out | axpy + residual | fused accum + residual | Yes |

---

## Anti-Patterns to Avoid

1. **Don't add `end_commands()` for debug reads** — this breaks the batch and
   changes timing/ordering. Instead, run both paths fully, then compare after
   the final `end_commands()`.

2. **Don't modify business logic for debugging** — if you add an `if (debug)`
   branch, you're testing a different code path than production.

3. **Don't delete debug code after fixing** — wrap it in `#ifdef` and leave it.
   The next bug will come, and the instrumentation is already there.

4. **Don't print every float** — use summary statistics (sum, max_abs, counts)
   for first pass. Only dump raw data after you've narrowed the location.

5. **Don't trust "first run" results** — pipeline compilation can affect the
   first token. Compare from token 1 onward.

---

## Makefile Integration

```makefile
# Debug flags (add to CFLAGS as needed)
CFLAGS_DBG_FUSED = -DL26F_DBG_FUSED
CFLAGS_DBG_COMPARE = -DL26F_DBG_COMPARE
CFLAGS_DBG_ALL = $(CFLAGS_DBG_FUSED) $(CFLAGS_DBG_COMPARE)

# Usage:
#   make CFLAGS="$(CFLAGS_DBG_ALL)" test_l26f_multilayer
#   make CFLAGS="-DL26F_DBG_FUSED" test_l26f_multilayer
```

No changes to the default build. Debug flags are opt-in via command line.
