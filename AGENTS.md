# AGENTS.md — l26f: Ling-2.6-flash Narrow-Metal Inference Engine

For AI agents and humans working in this directory.

## What We're Building

A single-file-dispatch Metal inference engine for Ling-2.6-flash (104B MoE, 7.4B active).
Hybrid architecture: 28 GLA layers + 4 MLA layers + MoE FFN. Modeled after ds4
(DeepSeek-V4 Flash narrow engine by antirez).

GGUF file: `models/Ling-2.6-flash-IQ4_NL-quality-bailing_hybrid-20260508-LJ.gguf` (58 GiB).

## Directory Map

```
l26f.h, l26f.c              Model struct, type table, shared defs
l26f_gguf.c                 GGUF loader (mmap, 540 tensors, metadata)
l26f_metal.h, l26f_metal.m  Metal kernels: GLA, matvec, RMS norm, embeddings
ds4_metal.h                 DS4 Metal runtime (shared with ds4, read-only)
l26f_mla_cpu.c              CPU-only MLA decode (Phase A: correctness first)
l26f_tokenizer.c, .h        BPE tokenizer (reads from model's existing mmap)
test_l26f_multilayer.c      Main driver: 32-layer end-to-end, multi-token gen
test_l26f.c                 Minimal GLA-only test
metal/l26f_gla.metal        GLA recurrent attention kernel
metal/l26f_dense.metal      Q5_K, Q6_K, IQ4_NL matvec kernels
ds4.c, ds4_cli.c, ds4_server.c  Original ds4 engine (read for reference)
docs/                        All design, status, and convention documents
```

## Documents in docs/

### Status Journal (time-ordered, append-only)

| File | When | What |
|------|------|------|
| `00-plan.md` | 2026-05-08 | Initial architecture, build plan |
| `01-caching-analysis.md` | ~ | KV cache strategy |
| `02-quantization-analysis.md` | ~ | Block structs, type table |
| `03-current-status.md` | ~ | GLA kernel working |
| `04-current-status.md` | ~ | Q5_K/Q6_K/IQ4_NL matvec working |
| `05-current-status.md` | ~ | Multi-layer testing |
| `06-current-status.md` | ~ | MoE integration |
| `08-current-status.md` | ~ | MLA plan forming |
| `11-status-tasks-complete.md` | 2026-05-09 | All 32 layers working, multi-token gen |
| `12-status-tokenizer-partial.md` | 2026-05-10 | Tokenizer integration partial |

### Reference Documents

| File | Purpose |
|------|---------|
| `07-manifesto.md` | 12 engineering rules (zero-fill, checksums, static asserts) |
| `09-lessons-learned.md` | Debugging timeline, what worked/didn't |
| `10-mla-plan.md` | MLA data flow, absorption trick, RoPE |
| `13-naming-conventions.md` | Dimension-annotated variable names (`_NxM`, `_1xCR`) |
| `14-code-style.md` | ALL_CAPS for non-business-logic, 0xC9 memory fill |

### Reference Code

| File | Purpose |
|------|---------|
| `xcommon.c`, `.h`, `.hpp` | Extract from itrade's xcommon.h (debug macros, safer allocs) |

## Current State (2026-05-10)

**Working**: All 32 layers end-to-end, zero NaNs, fully deterministic.
Multi-token auto-regressive generation with argmax selection.
GLA + MLA (CPU-only) + MoE FFN with correct expert routing and per-type byte strides.

**Partial**: Tokenizer loads metadata positions correctly but hangs during token
string allocation loop (157K tokens). See `12-status-tokenizer-partial.md`.

**Not yet**: Text I/O (blocked on tokenizer fix), GPU-accelerated MLA,
sampling (temperature/top-k/top-p), prefill, benchmarking.

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

## Critical Bugs Found (and How We'd Catch Them Next Time)

1. **GLA grid dispatch race**: `x=N/nsg=32` instead of `x=1`. 32 threadgroups wrote
   to same output. Found via checksum diff between two runs. Manifesto rule #2.
2. **MoE expert byte stride**: Used IQ4_NL params (bs=32,ts=18) for Q5_K data
   (bs=256,ts=176). `ffn_down_exps` is type=13 not type=20. Fixed by computing
   stride from actual tensor type.
3. **Q6_K struct field order**: `d` was first, should be last (after ql,qh,scales).
   Match `ggml-common.h:352-357`. Fixed with `_Static_assert`.

## Workflow

- Agent modifies code. LJ commits to git and pushes to GitHub.
- Commits and PRs are NOT the agent's responsibility.
- Status updates go in `docs/` as numbered files (12, 13, 14, ...).
- Time-dependent info goes in `docs/##_description.md` with ever-increasing numeric prefix.
- Conventions and reference docs go in `docs/` with descriptive names.
- AGENTS.md (this file) is the authoritative index of everything in docs/.

## Build

```sh
cd contrib/l26f
make clean && make test_l26f_multilayer
```

No external libraries needed. No llama.cpp linking. Everything in this directory.
Metal framework required (macOS only).
