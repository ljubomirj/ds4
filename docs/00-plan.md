# l26f: Ling-2.6-flash Narrow-Metal Inference Engine

A narrow-cut inference engine for Ling-2.6-flash (104B MoE, 7.4B active, hybrid MLA+GLA),
modeled after [ds4](https://github.com/antirez/ds4) (DeepSeek-V4 Flash narrow engine).

## Current Status (2026-05-08)

### Builds & Runs
- GGUF model loader: loads 540 tensors, resolves all names ✅
- Metal shader library: compiles all kernels including GLA ✅
- Metal init: `ds4_metal_init()` succeeds ✅
- Tensor allocation: `ds4_metal_tensor_alloc` works ✅

### Current Pain Point: Bus Error on Model Map Wrapping

`ds4_metal_set_model_map()` crashes with SIGBUS when wrapping the 58 GB NVME mmap
into no-copy MTLBuffers via `newBufferWithBytesNoCopy`. The crash occurs at:

```
ds4_metal_map_model_views() line 446:
    id<MTLBuffer> buffer = [g_device newBufferWithBytesNoCopy:
        (void *)(model_addr + page_model_offset + off) ...];
```

This is likely a macOS + Metal interaction with NVME-backed mmap pages.
The DS4 code assumes local-SSD-backed `MAP_SHARED` mapping.

Possible causes:
- The NVME volume uses a different VM policy that Metal's driver rejects
- Pages from the mmap aren't faulted in before Metal tries to wrap them
- `maxBufferLength` is smaller on M2 Max than expected for 58 GB file
- Sparse file: the GGUF has unallocated holes that Metal's driver hits

### Files

```
l26f/
├── l26f.h              # Data structures, GGUF types, constants
├── l26f_gguf.c         # GGUF model loader (metadata + tensor parsing)
├── l26f_metal.h        # Public Metal C API
├── l26f_metal.m         # ObjC Metal glue (extended ds4_metal.m)
├── l26f.c              # Inference driver (stub)
├── test_l26f.c         # Metal pipeline test
├── Makefile            # Build system
├── metal/
│   ├── l26f_gla.metal  # GLA kernel (ported from llama.cpp)
│   ├── l26f_dense.metal # IQ4_NL/Q5_K/Q6_K dequant + matvec kernels
│   ├── dense.metal     # ds4: Q8_0/F16/F32 matmul
│   ├── moe.metal       # ds4: MoE expert matmul
│   ├── norm.metal      # ds4: RMSNorm
│   ├── (14 more ds4 metal files)
├── docs/
│   ├── 00-plan.md      # This file
│   ├── 01-caching-analysis.md
│   ├── 02-quantization-analysis.md
```

### What's Done (detailed)

| Component | Details |
|-----------|---------|
| GGUF loader | Parses GGUF v3, extracts model metadata (33 layers, 4096 embd, 157K vocab, 256 experts), parses 540 tensors with correct types |
| GLA kernel | Ported from llama.cpp ggml-metal.metal:2754-2873. Chunkwise parallel GLA with recurrent state update. NSG variants 1/2/4/8. grid=(S/NSG, S/4, H*n_seqs) |
| IQ4_NL matvec | Single-token decode kernel + Q5_K/Q6_K matvec kernels. Generic dispatch router by GGUF type |
| Block types | IQ4_NL (18-byte blocks), Q5_K (176-byte), Q6_K (210-byte), Q8_0 all defined in Metal prefix |
| Dequant helpers | All from llama.cpp ggml-metal.metal: dequantize_iq4_nl, dequantize_q5_K, dequantize_q6_K, dequantize_f16_t4 |
| Reusable ds4 kernels | RMSNorm, SiLU, SwiGLU, element-wise ops, copy, embedding lookup, softmax, RoPE, concat, argsort, set_rows, sum_rows |

### What's Next (remaining)

| Priority | Task | Dependencies |
|----------|------|-------------|
| P0 | Fix model map bus error | None |
| P0 | RMS norm Metal test (first op on GPU) | P0 |
| P1 | GLA layer end-to-end test | P0 |
| P1 | Full 32-layer transformer loop | P1 |
| P2 | MLA kernel (DeepSeek2-style compressed attention) | P0 |
| P2 | MoE expert matmul with IQ4_NL weights | P0 |
| P3 | BPE tokenizer | None |
| P3 | CLI / server | P1, P3 |

## Architecture (reference)

### Ling-2.6-flash Model Specs

| Parameter | Value |
|---|---|
| Total params | 107.49 B |
| Active params | ~7.4B (8/256 experts) |
| Hidden size | 4096 |
| FFN size | 9216 |
| Expert FFN | 1024 |
| Experts | 256 (8 active, 8 groups, 4 group active) |
| Layers | 33 (32 transformer + 1 MTP) |
| MLA layers | 7, 15, 23, 31 (kv_lora_rank=512, q_lora_rank=1536) |
| GLA layers | All others (32 heads, head_dim=128) |
| Vocabulary | 157184 (BPE) |
| Max context | 131072 |
| RoPE | freq_base=6M, dim=64 |
| GGUF | 58 GB, IQ4_NL + Q5_K + Q6_K + Q8_0 + F32 |

### Memory Budget (M2 Max, 96 GB)

| Component | Size |
|---|---|
| Model weights | ~58 GB |
| KV cache (ring buffer, MLA layers) | ~50 MB |
| GLA recurrent state | ~57 MB |
| Compute buffers | ~200 MB |
| **Total** | **~59 GB** |

### Key Differences from llama.cpp

| Aspect | llama.cpp | l26f |
|---|---|---|
| KV cache | Cell-based full prefix | Sliding window ring + compression |
| Prompt cache | General-purpose serialization | Self-contained snapshot |
| GLA state | `llama_memory_hybrid` | Explicit state blob |
| Model dispatch | Per-architecture branches | Hardcoded Ling-2.6 dimensions |
| Weight loading | GGML backend buffers | Direct Metal no-copy MTLBuffers |

### Recurrent State Cutoff Strategy

GLA's S state is fixed-size (128×128×32 = 2 MB per layer in F32, ~56 MB for
28 layers). Snapshot at checkpoint intervals, replay only since last snapshot
on restore. All weights ≤ 8-bit, snapshots are tiny.
