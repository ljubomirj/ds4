# Status 12: Tokenizer Integration — Partial, Renaming Pending

**Date**: 2026-05-10

## Summary

Tokenizer integration is partially working. The GGUF metadata parser now captures
tokenizer positions during the existing scan (no separate file open needed). A debug
probe confirmed the positions are correct:

```
tok_found=1 tok_tokens_pos=2276 tok_tokens_count=157184
tok_merges_pos=3480810 tok_merges_count=156635
bos=156891 eos=156895
```

The `l26f_tokenizer_from_model()` function reads tokens from the model's mmap and
builds the token map, but it hangs during the 157K token allocation loop. The hang
is NOT in the GGUF parsing (that completes in <1s) — it's in the tokenizer's own
construction. Likely cause: malloc overhead for 157K tiny strings, or a bug in the
token reader consuming data at the wrong offset.

## What Changed Since Status 11

1. **Tokenizer rewritten to use model's existing mmap** — `l26f_tokenizer_open(path)`
   replaced with `l26f_tokenizer_from_model(&model)`. No separate GGUF file open.
   No external library dependency.

2. **GGUF parser extended** — `l26f_parse_metadata` now captures tokenizer array
   positions (`tok_tokens_pos`, `tok_tokens_count`, etc.) during its existing KV
   scan. Added fields to `l26f_model` struct.

3. **Critical bug found and fixed** — After reading array header (item_type + count),
   was incorrectly calling `l26f_skip_value(c, L26F_GGUF_ARRAY, 0)` which re-reads
   the header, corrupting the cursor. Fixed: skip only array body elements.

## Current Problem

`l26f_tokenizer_from_model()` hangs. The GGUF positions are correct (verified with
debug binary). The hang is inside the 157K-iteration loop that reads token strings
and builds the hash map. Needs debugging — add progress prints or sample the first
few tokens to see where it stalls.

## Next Steps (Priority Order)

1. **Rename/rewrite variables with dimension annotations** — User requested.
   Weights: `_NxK`, activations: `_BxN`, unknown dims: `_xU`, `_xUxU`, etc.
2. **Debug tokenizer hang** — Add stderr prints inside the token-reading loop.
3. **Test text I/O** — Once tokenizer works, run `"Hello"` through encode/decode.
4. **End-to-end text generation** — Prompt → tokenize → generate → detokenize.

## Files Modified in This Session

- `l26f.h` — Added tokenizer position fields to `l26f_model`
- `l26f_gguf.c` — Extended `l26f_parse_metadata` to capture tokenizer KV data
- `l26f_tokenizer.h` — Rewritten; includes `l26f.h`, added `l26f_tokenizer_from_model()`
- `l26f_tokenizer.c` — Rewritten to read from model mmap instead of re-opening GGUF
- `test_l26f_multilayer.c` — Reordered: model load → tokenizer → inference
- `Makefile` — Reverted (no llama lib needed)

## Key Data Points

- Tokenizer metadata positions: tokens at byte 2276, merges at byte 3480810
- 157184 tokens, 156635 merges, BOS=156891, EOS=156895
- Token data is in first ~3.5 MB of the GGUF (well within first page cache window)
