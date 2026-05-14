# Code Style & Development Process

Adapted from itrade's `xcommon.h` conventions and l26f's hard-won debugging lessons.

---

## 1. ALL_CAPS for Scaffolding, camelCase for Logic

When reading inference code, your eye should skip `XASSERT`, `XCHECK`, `XLOG` and land on
the algorithm between them. UPPERCASE = machinery you can tune out. camelCase = business
logic you must reason about.

```c
XCHECK(in_dim > 0, ("in_dim must be positive, got %zu", in_dim));
float sum = 0;
for (int i = 0; i < in_dim; i++) sum += weights[i] * input[i];
XASSERT(!isnan(sum), ("matvec produced NaN"));
```

### Macro Quick Reference

| Macro | When Active | Purpose |
|-------|-------------|---------|
| `XCHECK(cond, (fmt, args))` | Always | Fatal runtime check, calls `XerrorRaise` |
| `XASSERT(cond)` | Debug only | Non-fatal invariant, `__builtin_trap` in debug |
| `XASSERT_STOP(cond)` | Always | Fatal assertion, always traps |
| `XASSERT_STATIC(expr, msg)` | Compile time | Type/array size checks |
| `XLOG(fmt, ...)` | Always | Diagnostic logging to stderr |
| `XERROR(x)` | Always | Fatal error, calls `XerrorRaise` (note: parenthesized printf args, e.g. `XERROR(("bad %d", n))`) |
| `XMEMSET_INIT(ptr, bytes)` | Always | 0xC9 in debug, 0x00 in release |
| `XLOG_EVERY(n, i, total, ...)` | Always | Periodic logging inside loops |

---

## 2. Memory Fill Patterns

| Pattern | Value | When | Meaning |
|---------|-------|------|---------|
| `FILL_ALLOC` | 0xCC | After malloc (debug) | Fresh allocation, not yet written |
| `FILL_FREE` | 0xC9 | Before free (debug) | Dead memory, use-after-free detector |
| `XMEMSET_DEAD` | 0xC9 | GPU buffer init (debug) | Uninitialized GPU buffer sentinel |

As float: `0xCCCCCCCC` → `-1.07374e+08`. If you see this in output, a kernel didn't
write that element.

---

## 3. Argument Naming by Direction

| Prefix | Meaning | Example |
|--------|---------|---------|
| `in_` | Input (read-only) | `in_dim`, `in_hidden` |
| `out_` | Output (write) | `out_logits`, `out_proj` |
| `io_` | Input-output (read-write) | `io_state` |
| (none) | Member / local | `layer`, `n_embd` |

Makes data flow visible at the call site without jumping to the definition.

---

## 4. Encode Dimensions in Variable Names

From itrade convention #30. Suffix pattern: `_RxC` where R = rows, C = columns.

```c
float hidden_1xN[4096];      // 1 token × hidden dim
float moe_out_TxN;           // T tokens × hidden dim
float router_logits_TxE;     // T tokens × E experts
float sel_idx_TxK;           // T tokens × K selected
float q_a_1xQ;               // 1 token × Q lora rank
```

Constants: `N=4096`, `E=256`, `K=8`, `V=157184`, `S=128`, `C=512`, `R=64`, `Q=1536`.

---

## 5. No Headers Including Headers

All `#include` goes in `.c` files only. Headers declare types and functions but include
nothing. This makes dependencies explicit and prevents hidden coupling.

---

## 6. Three Tiers of Checking

| Tier | Macro | When Active | Cost |
|------|-------|-------------|------|
| 1 | `XCHECK(cond, (msg))` | Always | Minimal: branch + format string |
| 2 | `XASSERT(cond)` | Debug only | Can be expensive (tensor scans) |
| 3 | `XASSERT_STATIC(expr, msg)` | Compile time | Zero runtime cost |

Every offset, every buffer size, every kernel return value gets an `XCHECK`.
No silent failures. Every check prints file:line and what went wrong.

---

## 7. Debugging Protocol

### NaN appears
1. Is the input NaN? Read the input tensor. If yes, go upstream.
2. Are the weights NaN? Dequantize one block on CPU.
3. Is the offset correct? Print absolute byte offset, verify against Python GGUF reader.
4. Is the output deterministic? Run twice, compare checksums. If different → race condition.

### Data flow is invisible
1. Insert `XLOG()` at function entry/exit and after major blocks.
2. Use `XLOG_EVERY(10000, i, total, ...)` inside large loops.
3. For GPU tensors: checksum after every kernel, compare across runs.
4. Build with `make debug` to get `XASSERT` active and 0xC9 fill patterns.

### Reproducibility
1. Cache all external data sources.
2. Any experiment must be rerunnable from cache alone.
3. Version data in git where possible (ASCII), filesystem where not (binary).

---

## 8. Build Commands

```sh
make release    # → build_release/test_l26f_multilayer  (-O3, no XASSERT)
make debug      # → build_debug/test_l26f_multilayer    (-O0 -g -DL26F_DEBUG)
make clean      # removes both build dirs
```

Use `debug` for development and correctness work. Use `release` for benchmarking.

---

## 9. Static Asserts for Struct Sizes

Every quant block struct gets a compile-time size assertion in C and Metal:

```c
_Static_assert(sizeof(block_iq4_nl) == 18, "IQ4_NL block must be 18 bytes");
_Static_assert(sizeof(block_q5_K)   == 176, "Q5_K block must be 176 bytes");
_Static_assert(sizeof(block_q6_K)   == 210, "Q6_K block must be 210 bytes");
```

Or equivalently: `XASSERT_STATIC(sizeof(block_iq4_nl) == 18, IQ4_NL_must_be_18_bytes);`

These prevent the single most common class of bug: wrong struct definitions copied
between C host and Metal kernel source.

## Key Dimensions (Ling-2.6-flash)

```
N = 4096   hidden dim         S = 128    GLA state dim
F = ~      dense FFN int      M = 1024   expert FFN int
V = 157184 vocab size         H = 32     attention heads
D = 192    head dim           P = 128    qk_nope (= D - R)
Q = 1536   q_lora_rank        C = 512    kv_lora_rank
R = 64     n_rot (RoPE)       CR = 576   kv_dim (= C + R)
E = 256    total experts      K = 8      selected experts
G = 8      expert groups      T = (var)  sequence position
```

See `docs/13-naming-conventions.md` for the full suffix system.
