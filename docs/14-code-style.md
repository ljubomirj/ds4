# Code Style: Separating Logic from Machinery

## ALL_CAPS for Non-Business-Logic

Code that is orthogonal to the main inference path — logging, assertions, memory
poisoning, debug checks — uses ALL_CAPS identifiers so the human eye can filter
it out when reading the core algorithm.

### Categories

| Category | Pattern | Purpose |
|----------|---------|---------|
| Logging | `XLOG_`, `LOG_` | Informational output during inference |
| Warnings | `WARN_` | Non-fatal anomalies |
| Errors | `XERR_` | Fatal or unexpected conditions |
| Assertions | `XASSERT`, `ASSERT_` | Invariant checks |
| Memory | `XMEMSET_` | Memory initialization/poisoning |
| Debug | `XDUMP_`, `XTRACE_` | Temporary debug output |
| Checksum | `CHECKSUM_` | Tensor validation |

### Rationale

When reading inference code, I want to see:

```c
cpu_matvec(model_base + t_q_a->abs_offset, normed_1xN, q_a_1xQ,
           t_q_a->type, q_lora_rank, n_embd);
```

Not:

```c
ASSERT(t_q_a != NULL);
LOG_DEBUG("computing Q compression, rank=%d", q_lora_rank);
cpu_matvec(model_base + t_q_a->abs_offset, normed_1xN, q_a_1xQ,
           t_q_a->type, q_lora_rank, n_embd);
CHECK_TENSOR("q_a_1xQ", q_a_1xQ, q_lora_rank);
```

UPPERCASE makes the scaffolding visually distinct from the algorithm. Your eye
jumps to the lowercase business logic and skips the uppercase machinery.

### Examples

```c
// Good — diagnostic code stands out
fprintf(stderr, "l26f: layer %u missing tensors\n", layer);
// → would become:
XERR("l26f: layer %u missing tensors\n", layer);

// Good — optional safety
#define XMEMSET_DEAD 0xC9
memset(kv_cache->data, XMEMSET_DEAD, kv_bytes);

// Good — invariant
assert(n_tokens < max_seq_len);
// → XASSERT(n_tokens < max_seq_len);

// Good — checksum
l26f_checksum_print("gla_out", t, bytes);
// → CHECKSUM_TENSOR("gla_out", t, bytes);
```

## Memory Initialization: 0xC9 Pattern

All allocated memory is initialized to `0xC9` (the "dead" pattern) in debug
builds and `0x00` in release builds:

```c
#ifdef NDEBUG
#define XMEMSET_INIT(ptr, bytes) memset((ptr), 0x00, (bytes))
#else
#define XMEMSET_INIT(ptr, bytes) memset((ptr), 0xC9, (bytes))
#endif
```

Rationale for 0xC9:
- Non-zero: catches "forgot to write to this buffer" bugs immediately
- Distinctive: easy to spot in debugger dumps
- Produces NaN in float (helps catch uninitialized float arrays)
- Not 0xCC (MSVC debug fill) or 0xCD (MSVC heap fill) to avoid confusion
- Not 0xFF (could be mistaken for actual -NaN)

All GPU buffers get `0xC9` filled after allocation in debug mode:

```c
void *buf = calloc(1, n_embd * sizeof(float));
#ifdef L26F_DEBUG
memset(buf, 0xC9, n_embd * sizeof(float));
#endif
ds4_metal_tensor_write(tensor, 0, buf, n_embd * sizeof(float));
free(buf);
```

## Existing Code to Convert

Current patterns that should become ALL_CAPS (but haven't yet):

| Current | Should Become | File |
|---------|---------------|------|
| `fprintf(stderr, "l26f: ...")` | `XERR("l26f: ...")` | all |
| `printf("Loading model...\n")` | `XLOG("Loading model...\n")` | test |
| `printf("  layer %u: GLA\n", il)` | `XLOG("  layer %u: GLA\n", il)` | test |
| `assert(...)` | `XASSERT(...)` | all |
| `calloc(1, bytes)` → zero-fill | `XMEMSET_INIT(buf, bytes)` | session init |
| `ds4_metal_tensor_fill(t, 0.0f)` | `XZERO_GPU(t)` | session init |
| `l26f_checksum_print(...)` | `XCHECKSUM(...)` | test |
| Hardcoded `0xC9` in memset | `XMEMSET_DEAD` macro | all |
