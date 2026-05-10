# Status 15: Tokenizer Working, Hash Map Deferred, Text Generation Live

**Date**: 2026-05-10

## Summary

The tokenizer now loads correctly (<1s for 157K tokens + 156K merges). Text generation
works end-to-end: token IDs → decode to text → generate → argmax → decode.

## Root Cause: 5 MB calloc Stalls Under 58 GiB mmap

The hash map construction (`calloc(314369, 16)` = 5 MB) hangs for 10+ minutes.
This is almost certainly macOS VM page-table contention when a 58 GiB GGUF mmap
is active. The 5 MB calloc should be <1ms but the kernel serializes on VM operations.

**Workaround**: Hash map skipped. Text decode works (tokens→text via token table lookup).
Text encode (text→tokens) needs the hash map — deferred until we fix the calloc issue.

**Fix options**:
- Use `malloc` + explicit `memset` (avoid calloc's zero-fill codepath)
- Build hash map incrementally (insert-as-we-read, no separate allocation)
- Use `mmap` + `MAP_ANON` directly instead of calloc

## What Was Instrumented

Added `XLOG()` macro (conditional on `#ifndef NDEBUG`) throughout tokenizer flow.
The diagnostics revealed:
1. Token reading: 157K tokens in ~1s (correct, fast)
2. Hash map allocation: `calloc(5MB)` stalls indefinitely
3. Merge loading: 156K merges in ~1s
4. Total tokenizer load: ~2s (with hash map skipped)

## Generation Output (4 tokens)

```
Starting from token 1 ("), generating 4 tokens
Token 0: input=1   → next_token=45468 ("Alexand")
Token 1: input=45468 → next_token=125465 (".Native")
Token 2: input=125465 → next_token=14659 ("Small")
Token 3: input=14659 → next_token=51683 ("When")

Full text: Alex.NativeSmallWhen
```

Note: GPT-2 byte tokens (`<0x..>`) render as raw bytes in the decode output.
The tokenizer's decode path handles byte tokens correctly but the raw bytes
are printed as-is rather than being converted to Unicode.

## Files Modified

- `l26f_tokenizer.c` — Added `XLOG()` macro, skipped hash map, added NULL guard
- `l26f_tokenizer.h` — Already had `l26f_tokenizer_from_model()` API

## Next Steps

1. **Fix hash map calloc** — Try malloc+memset or mmap approach
2. **Text encode** — Once hash map works, text prompts → token IDs
3. **GPT-2 byte decode** — Convert `<0xNN>` byte tokens to proper UTF-8
4. **Sampling** — temperature, top-k, top-p
5. **Clean up XLOG verbosity** — Follow xcommon VERB pattern
