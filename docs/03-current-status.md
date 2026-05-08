# l26f — Current Status

**Date**: 2026-05-08
**Branch**: `l26f` (ljubomirj/l26f fork of antirez/ds4)
**Model**: Ling-2.6-flash IQ4_NL quality GGUF (58 GB)

---

## What Works (All Verified)

### 1. GGUF Model Loader (`l26f_gguf.c`)

Parses GGUF v3, reads metadata KV pairs (extracting model params), enumerates all
540 tensors with correct names, shapes, types, and file offsets. Verified against
the Python `gguf` library — offsets match exactly.

### 2. Metal Shader Compilation

All kernels compile into a single Metal library:
- 18 ds4 metal files (dense, moe, norms, unary, glu, bin, cpy, etc.)
- `metal/l26f_gla.metal` — GLA kernel ported from llama.cpp
- `metal/l26f_dense.metal` — IQ4_NL/Q5_K/Q6_K dequant helpers + matvec kernels

### 3. Metal Runtime

- `ds4_metal_init()` succeeds
- Model map wrapping: 2 overlapping MTLBuffer views covering 58 GiB
- Tensor allocation/view/free all work

### 4. Full GLA Layer Pipeline (End-to-End)

**This is the big milestone.** The complete GLA layer pipeline runs correctly:

```
Input (4096 floats) → RMS Norm → QKV Matvec → Gate Matvec → GLA Attention → Output Matvec → Residual Add → Output
```

Test result:
```
Result: sum=2026.004883 min=-1.412428 max=1.580171 avg=0.494630
First 10 values: 0.2035 -0.0913 0.0195 0.1277 0.0624 0.1691 -0.1161 0.1925 -0.0783 0.0954
State: sum=28.941093
```

Zero NaNs. All values in reasonable range. State updated correctly.

### Bugs Fixed So Far

1. **GGUF tensor_data_pos calculation** (commit `141b72a`): Was computed before
   tensor info entries; fixed to compute after all entries + alignment padding.

2. **Weight offset bug**: `test_l26f.c` was subtracting `tensor_data_pos` from
   `abs_offset` before passing to Metal functions. `abs_offset` is already the
   absolute file position (`tensor_data_pos + gguf_offset`). The Metal views
   cover the entire file from byte 0, so the offset must be passed directly —
   exactly as ds4 does it. Subtracting `tensor_data_pos` shifted all reads by
   ~34KB into garbage, producing ~42 NaN values out of 4096 in RMS norm output.

---

## Lessons Learned

### 1. `abs_offset` is absolute — never adjust it for Metal

The GGUF loader computes `abs_offset = tensor_data_pos + gguf_offset`. This is
the byte offset from the start of the mmap'd file. The Metal views (created by
`ds4_metal_set_model_map`) cover the entire file from byte 0. Therefore:
- **Always pass `abs_offset` directly** to `ds4_metal_*` functions.
- **Never subtract `tensor_data_pos`** — it's already baked in.
- ds4's own code confirms this pattern (grep for `abs_offset` in ds4.c).

### 2. When adapting ds4 code, follow ds4's calling conventions exactly

ds4 was the reference implementation. Every time we wrapped a ds4 function call,
we should have copied the argument pattern verbatim from ds4.c. The NaN bug
arose because we "adapted" the offset calculation differently from ds4.
**Rule: if ds4 passes X, we pass X — no clever arithmetic.**

### 3. NaN in kernel output ≈ wrong data being read

When a Metal kernel produces NaN on valid inputs, the most likely cause is that
it's reading from the wrong memory. Before debugging kernel math, verify:
1. The weight offset passed to `ds4_metal_wrap_model_range` is correct
2. The data at that file offset is what you expect (read back and print)
3. The inner_offset returned by the view lookup makes sense

### 4. The 34KB header offset is a constant trap

GGUF files have a ~34KB header (magic, version, KV pairs, tensor info entries)
before tensor data begins. Two offset spaces exist:
- **File offset**: absolute position in the file (what `abs_offset` gives you)
- **GGUF offset**: position relative to the tensor data section (what the GGUF
  spec calls "offset" in the tensor info entry)

Never confuse the two. `abs_offset` resolves GGUF offset → file offset. The
Metal mmap starts at file byte 0. Everything going to Metal uses file offsets.

---

## What's Next

### P0: Multi-Layer Transformer Loop

Build the full 32-layer inference loop. Each GLA layer (28 of 32) does:

```
for each token position:
    1. RMS norm (attn)
    2. QKV matvec (Q5_K, 4096→12288)
    3. Gate matvec (Q5_K, 4096→4096)
    4. GLA attention (S=128, H=32)
    5. Output matvec (Q5_K, 4096→4096)
    6. Residual add
    7. RMS norm (FFN)
    8. FFN (MoE or shared — needs MoE kernel)
    9. Residual add
```

### P1: MLA Layers (layers 7, 15, 23, 31)

4 global attention layers use DeepSeek2-style MLA instead of GLA.
Can reuse ds4's MLA kernels directly.

### P2: MoE Expert Routing + Matvec

256 experts, 8 active per token, group expert selection (n_group=8, topk_group=4).
Need routing kernel + per-expert matvec with IQ4_NL/Q5_K weights.
ds4 has MoE kernels for Q8_0 — need extending for our quant types.

### P3: Tokenizer + Sampling + CLI

BPE tokenizer (157184 vocab), temperature/top-k/top-p sampling, CLI interface.

---

## Key Files

| File | Role |
|------|------|
| `l26f_gguf.c:156-193` | Tensor parsing with abs_offset calculation |
| `l26f_metal.m:14558-14695` | GLA dispatch (blit before commit) |
| `l26f_metal.m:4385-4410` | Model map view setup |
| `l26f_metal.m:14842` | Generic quantized matvec dispatch |
| `metal/l26f_gla.metal` | GLA kernel |
| `metal/l26f_dense.metal` | IQ4_NL/Q5_K/Q6_K matvec + dequant helpers |
| `metal/norm.metal` | RMS Norm kernels (from ds4) |
| `test_l26f.c` | End-to-end GLA layer test |

## Build & Run

```bash
cd ~/llama.cpp/contrib/l26f
make
./test_l26f ~/llama.cpp/models/Ling-2.6-flash-IQ4_NL-quality-bailing_hybrid-20260508-LJ.gguf
```
