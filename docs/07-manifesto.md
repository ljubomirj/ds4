# l26f Engineering Manifesto

**Purpose**: Practical rules for writing robust, debuggable code in l26f — drawn from
hard-won lessons in this project and patterns proven in the itrade codebase.

---

## 1. Uninitialized Memory Is Enemy #1

Our current non-determinism bug almost certainly stems from reading uninitialized GPU
buffers. This is the single most important rule:

**Every `ds4_metal_tensor_alloc` must be followed by a zero-fill.**

```c
// WRONG (current):
s->comp.post_attn = ds4_metal_tensor_alloc(act_bytes);

// RIGHT:
s->comp.post_attn = ds4_metal_tensor_alloc(act_bytes);
l26f_tensor_zero(s->comp.post_attn, act_bytes);
```

Metal's `MTLResourceStorageModeShared` does NOT zero memory. If any kernel skips even
one element (e.g., due to a thread-count rounding error), garbage leaks through and
propagates to every downstream operation.

**Fill pattern for debugging**: When debugging non-determinism, fill buffers with
`0xCCCCCC...` (the itrade `FILL_ALLOC` pattern). If you see `0xCCCCCCCC` as a float
(`-1.07374e+08`) in output, you know a kernel didn't write that element.

---

## 2. Checksum Every Tensor After Every Kernel

The fastest way to pinpoint a non-deterministic kernel is to checksum the output of
every GPU operation and compare across runs.

```c
static float l26f_tensor_checksum(ds4_metal_tensor *t, uint64_t bytes) {
    float *data = malloc(bytes);
    ds4_metal_tensor_read(t, 0, data, bytes);
    float sum = 0;
    uint32_t nans = 0;
    for (uint64_t i = 0; i < bytes / sizeof(float); i++) {
        if (isnan(data[i])) { nans++; continue; }
        sum += data[i];
    }
    free(data);
    return sum;
}
```

After each kernel call, compute the checksum and log it. When a checksum diverges
between runs, the kernel that produced it is the culprit. This would have found our
non-determinism bug in minutes instead of hours of kernel micro-analysis.

---

## 3. Three Tiers of Checking

Adapted from itrade's `XCHECK` / `XASSERT` / `XASSERT_STATIC` pattern:

| Tier | Macro | When Active | Purpose |
|------|-------|-------------|---------|
| 1 | `L26F_CHECK(cond, msg)` | Always | Runtime invariants: buffer sizes, offsets, return codes |
| 2 | `L26F_ASSERT(cond, msg)` | Debug builds | Expensive checks: tensor contents, value ranges |
| 3 | `_Static_assert(cond, msg)` | Compile time | Type sizes, array sizes, constants |

```c
#define L26F_CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "l26f CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, msg); \
        return 0; \
    } \
} while (0)

#ifdef NDEBUG
  #define L26F_ASSERT(cond, msg) ((void)0)
#else
  #define L26F_ASSERT(cond, msg) do { \
      if (!(cond)) { \
          fprintf(stderr, "l26f ASSERT failed at %s:%d: %s\n", __FILE__, __LINE__, msg); \
          __builtin_trap(); \
      } \
  } while (0)
#endif
```

**Every offset, every buffer size, every kernel return value gets a `L26F_CHECK`.**
No silent failures. The `return 0` pattern we currently use is fine for flow control,
but it must always print what went wrong and where.

---

## 4. Encode Dimensions in Variable Names

From itrade convention #19-20. When you read `mClosePrc_NxS` you know the shape
without looking it up. We deal with shapes constantly:

```c
// WRONG:
uint64_t gate_exp_bytes, up_exp_bytes, down_exp_bytes;

// RIGHT — encode the shape and quant type:
const uint64_t gate_exp_bytes  = 1024 * (4096 / 32) * 18;   // [4096,1024,256] IQ4_NL
const uint64_t down_exp_bytes  = 4096 * (1024 / 32) * 18;    // [1024,4096,256] IQ4_NL
const uint64_t attn_qkv_bytes  = 4096 * (12288 / 256) * 176;  // [4096,12288]   Q5_K
```

Or better, derive them from the tensor metadata and name the intermediate:

```c
const uint32_t blocks_per_row_down = wt_down_exps->dim[0] / 32;  // IQ4_NL: block_size=32
const uint64_t down_exp_bytes      = wt_down_exps->dim[1] * blocks_per_row_down * 18;
```

---

## 5. Prefix Arguments by Direction

From itrade convention #6:

| Prefix | Meaning | Example |
|--------|---------|---------|
| `in_` | Input (read-only) | `in_dim`, `in_hidden` |
| `out_` | Output (write) | `out_logits`, `out_proj` |
| `io_` | Input-output (read-write) | `io_state` |
| (none) | Member / local | `layer`, `n_embd` |

This makes data flow visible at the call site:

```c
int l26f_metal_matvec_quant(
    ds4_metal_tensor       *out_dst,
    const ds4_metal_tensor *in_src,
    const void             *in_model_map,
    uint64_t                in_model_size,
    uint64_t                in_weight_offset,
    uint64_t                in_dim,
    uint64_t                out_dim,
    uint32_t                in_weight_type,
    uint64_t                in_n_tok);
```

---

## 6. One Struct Size Bug = One Static Assert

Every time we fix a struct size bug (IQ4_NL was 34 instead of 18, etc.), add a
compile-time assertion so it can never regress:

```c
_Static_assert(sizeof(block_iq4_nl) == 18, "IQ4_NL block must be 18 bytes");
_Static_assert(sizeof(block_q5_K)   == 176, "Q5_K block must be 176 bytes");
_Static_assert(sizeof(block_q6_K)   == 210, "Q6_K block must be 210 bytes");
_Static_assert(sizeof(block_q8_0)   == 34,  "Q8_0 block must be 34 bytes");
```

These go in the Metal prefix string AND in the C header. Both sides must agree.

---

## 7. Copy Block Structs Verbatim — Then Assert

This is already lesson #1 from our history. The reinforcement: **always copy the
struct definition from the upstream `ggml-metal.metal`**, then immediately add the
`_Static_assert`. If the assertion fires, the copy was wrong.

---

## 8. No Headers Including Headers

From itrade convention #5. Our `l26f.h` currently includes nothing, which is correct.
Keep it that way. All `#include` goes in the `.c` files. This makes dependencies
explicit and prevents hidden coupling.

---

## 9. Uppercase Macros for Scaffolding, camelCase for Logic

From itrade convention #10. When reading the algorithm between the checks, your eye
should glide past the `L26F_CHECK`, `L26F_ASSERT`, `VERB` calls because they're
UPPERCASE. The actual logic is the camelCase code between them.

```
L26F_CHECK(out_dim > 0, "out_dim must be positive");
float sum = 0;
for (int i = 0; i < in_dim; i++) sum += weights[i] * input[i];
L26F_ASSERT(!isnan(sum), "matvec produced NaN");
```

---

## 10. Verify First, Optimize Never (Until Correct)

We spent hours micro-analyzing the IQ4_NL kernel's arithmetic when the real bug was
non-determinism from uninitialized memory. The rule:

**Correctness before performance. Always.**

Before optimizing a kernel, verify it produces deterministic, correct output against a
CPU reference. Only then optimize.

The concrete workflow:
1. Write a CPU reference dequant + matvec for each quant type.
2. Run the Metal kernel on the same data.
3. Compare outputs element-by-element.
4. Only after match, consider the kernel "verified."

---

## 11. The NaN Debugging Protocol

When NaN appears:

1. **Is the input NaN?** Read the input tensor. If yes, go upstream.
2. **Are the weights NaN?** Dequantize one block on CPU. If the half `d` field is
   NaN or infinity, the GGUF offset is wrong.
3. **Is the offset correct?** Print the absolute byte offset and verify against the
   Python GGUF reader.
4. **Is the output deterministic?** Run twice, compare checksums. If they differ,
   you have a race condition or uninitialized read — stop debugging kernel math.

We violated rule 4 for hours. Never again.

---

## 12. Tensor Summary, Not Just Sum

The itrade `xvecsummary` pattern — print a one-line statistical summary:

```
nans=0/4096 sum=-153.9 min=-36.0 max=13.9 p1=-12.1 p99=8.3 mu=-0.04 sd=3.2
```

This is far more diagnostic than just `sum`. The `nans` count catches NaN. The
`min`/`max` catch explosions. The quantiles catch distributional shifts. We already
have `sum`/`min`/`max`/`nans` — adding percentiles is a small effort for large
diagnostic gain.

---

## 13. Data debugging
1. Piping is useful for running things in parallel, especially with the multi-core machines we all use nowadays. Pipes provide the simplest dependency graph, and they have automatic IPC synchronization courtesy of the pipes FIFO-s flowing from one process to the next. That allows for individual processes in the pipeline to be run independently on different cores with the data flowing between allowing them to keep busy and not stall waiting.
2. Drawback of data flowing through the pipelines is that data flow is invisible to us and no record remains of the data available for inspection and debugging. Replacing a pipe "|" with "| tee tempfile |" allows for subsequent inspection of the "tempfile" to see what the data looked like.
   Example - debug:
```
     $ for a in {1..10}; do echo $a; done | wc -l
```
   as
```
     $ for a in {1..10}; do echo $a; done | tee tempfile | wc -l
     $ cat tempfile
```
3. Vectorised processing via numpy or in languages like matlab or array languages where the default data type is not a scalar (but an array or Ndim array in general) makes peeking into the individual values being aggregated or processed hard and needing extra effort.
  * Use hardware support for NaN to your advantage to prevent accidently using missing data. Convert missing data on input into NaN in memory where possible (data is float or double). Any operation with a NaN will result in a NaN, so it will be detected in the end.
  * If possible structure processing in the form of apply(function,array) so to use a kernel function that data will flow through. Then insert logging inside it to inspect data flowing through, turn it on/off where insight/speed is needed.
4. Use any data invariants to your advantage. Add asserts liberally to catch data breaking invariants as early as possible. If possible designate not-a-value for other types (analogous to NaN for floats) where possible, even if the hardware support is lacking: 0 for indices (assuming 1-based), INT_MIN as INT_NAN for integers, nullprt for pointers.
5. This book has been useful to me for coding, but lessons apply to data too - "Writing Solid Code" by Steve Maguire.
6. Data is used to run experiemnts, and experiments *must* be reproducable (b/c - science!). So:
  * Any external data source used (and thus outside our control) must be cached.
  * It must be possible to run the whole experiment without touching any external data source, only using cached data - so any run is reproducable.
  * Often it's possbile to leverage binary native de/serialization capabilities with a bit of care and prologue/epilogue code blocks.
    Example - python:
```
       name = cache_name()
       try:
           with open(name, 'rb') as f: st = pickle.load(f)
       except IOError:
           st = fetch_from_external_source()
           ... (maybe reprocessing cleaning) ...
           with open(name, 'wb') as f: pickle.dump(st, f)
```
    Example - matlab:
```
       name = cache_name();
       st = load_struct_from_mat(name);
       if isempty(st)
         st = fetch_from_external_source();
         ... (maybe preprocessing, cleaining) ...
         save_struct_to_mat(name, st);
       end
```
7. Version data: where possible/space allowing, and often where the data is ASCII (e.g. .csv or .tsv files) - add it straight to a git repository.
8. Version data: where not possible b/c too much space - version the binary blobs outside a repository using the filesystem.
9. Keep ASCII files (.csv, .tsv) on disk compressed in a streaming-friendly format, not only to (a) save space, but also to ensure (b) integrity via checksuming, and possibly (c) speed up reading.
10. Use formats like .zstd and .gz that are append and rsync friendly, e.g. where
```
    if $ zstd -c file1 >file.zst; zstd -c file2 >>file.zst, then $ diff <(zstd -dc file.zst) <(cat file{1,2})
```
  produces no diffs.
11. Use --rsyncable in gzip and zstd (NB bzip2 lacks that) to create rsync-friendly compressed file.
12. If keeping ASCII data compressed (e.g. .csv.zst) in git can enable comfortable diffing -
    in .gitattributes add:
```
      *.csv.zst diff=csv.zst
    while in .gitconfig add:
      [diff "csv.zst"]
      textconv = zstdcat
```

