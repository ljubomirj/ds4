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
| `15-status-tokenizer-working.md` | 2026-05-10 | Tokenizer working, text gen live, hash map deferred |
| `16-resources-on-disk.md` | 2026-05-10 | Full index of on-disk reference resources |

### Reference Documents

| File | Purpose |
|------|---------|
| `07-manifesto.md` | 12 engineering rules (zero-fill, checksums, static asserts) |
| `09-lessons-learned.md` | Debugging timeline, **Lesson 5: instrument before reasoning** |
| `10-mla-plan.md` | MLA data flow, absorption trick, RoPE |
| `13-naming-conventions.md` | Dimension-annotated variable names (`_NxM`, `_1xCR`) |
| `14-code-style.md` | ALL_CAPS for non-business-logic, 0xC9 memory fill, XLOG pattern |

### Reference Code

| File | Purpose |
|------|---------|
| `xcommon.c`, `.h`, `.hpp` | Extract from itrade's xcommon.h (debug macros, safer allocs) |

### Resource Map

| File | Purpose |
|------|---------|
| `docs/16-resources-on-disk.md` | Full index of on-disk resources: worktrees, MLX repos, Metal references |

## Current State (2026-05-10)

**Working**: All 32 layers end-to-end, zero NaNs, fully deterministic.
Multi-token auto-regressive generation with argmax selection.
GLA + MLA (CPU-only) + MoE FFN with correct expert routing and per-type byte strides.
**Tokenizer** loads tokens + merges in ~2s, decodes token IDs to text.
**Text generation** works: prompt token → generate → decode → print.

**Partial**: Text encode (text→tokens) needs hash map lookup.
Hash map construction (`calloc 5MB`) stalls under 58 GiB mmap — macOS VM issue.
GPT-2 byte token rendering has artifacts (raw bytes in output).

**Not yet**: GPU-accelerated MLA, sampling (temperature/top-k/top-p),
prefill, benchmarking, text prompt input.

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
- **Commits and PRs are NOT the agent's responsibility.** Agents write files
  but do not commit to git unless LJ explicitly asks. If LJ says "commit",
  the agent may commit only the changes LJ specified.
- Status updates go in `docs/` as numbered files (12, 13, 14, ...).
- Time-dependent info goes in `docs/##_description.md` with ever-increasing numeric prefix.
- Conventions and reference docs go in `docs/` with descriptive names.
- AGENTS.md (this file) is the authoritative index of everything in docs/.

### README.LJ — LJ's personal log

LJ writes into `README.LJ` files. Agents may read these files but should
know that their content may change at any time as LJ edits them. These are
LJ's personal notes — discussions, plans, deliberations, conversation
history, and records of what was done. Do not write to `README.LJ` unless
explicitly asked.

## Debugging Pattern: XLOG Before Reasoning

When something hangs or produces wrong output:

1. Add `XLOG()` at entry/exit of suspect functions and after major blocks
2. Add `XLOG_EVERY(10000, i, total, ...)` inside large loops
3. `make clean && make`, run, observe stderr
4. The data tells you exactly where behavior diverges from expectation
5. Now investigate that one line — not the whole codebase

See `docs/09-lessons-learned.md` Lesson 5 for the full story.
See `docs/14-code-style.md` for XLOG macro definition and conventions.

## Build

```sh
cd worktrees/l26f   # or just `make` if already in the worktree
make clean && make test_l26f_multilayer
```

No external libraries needed. No llama.cpp linking. Everything in this directory.
Metal framework required (macOS only).

---

## Git / GitHub Repository Setup

### Remotes

| Remote | URL | Direction |
|--------|-----|-----------|
| `origin` | `https://github.com/ljubomirj/l26f.git` | Public fork (you) |
| `upstream` | `https://github.com/antirez/ds4` | Original project |

`origin` is **your** public repo. `upstream` is antirez's ds4.

### Worktrees

This directory (`worktrees/l26f/`) is a **git worktree** of the ds4 repo at
`../..` (i.e. `~/llama.cpp/contrib/ds4/`). Three worktrees exist:

| Worktree path | Branch | Tracks | Purpose |
|---|---|---|---|
| `~/llama.cpp/contrib/ds4` (main repo) | `main` | `origin/main` | Pinned at the commit where `l26f` was branched (d615ab0). Preserved for reference. |
| `.../worktrees/l26f` | `l26f` | `origin/l26f` | **Your working branch.** All development happens here. |
| `.../worktrees/ds4-main` | `ds4-main` | `upstream/main` | Latest upstream HEAD. Updated via `git pull` to track antirez's changes. |

### Push Safety: Only `l26f` Goes to Origin

The `origin` remote is **locked** to only ever push the `l26f` branch:

```ini
[remote "origin"]
	push = refs/heads/l26f:refs/heads/l26f
```

This means from **any** worktree or branch, `git push origin` (no refspec) will
only push `l26f`. An explicit `git push origin main:main` would still work, but
you'd have to type it deliberately. The configured refspec prevents accidental
pushes via bare `git push`, `git push --all`, or `git push origin` from the wrong
worktree.

### Daily Workflow

```sh
# Work in the l26f worktree
cd ~/llama.cpp/contrib/ds4/worktrees/l26f

# Make changes, commit, push
git add -A
git commit -m "some message"
git push origin

# To see upstream changes without touching your work:
cd ~/llama.cpp/contrib/ds4/worktrees/ds4-main
git pull

# View all worktrees from anywhere:
git worktree list
```

### Why This Setup

- The `main` branch in the main repo is **frozen** at the ds4 revision from which
  `l26f` was forked. This preserves the baseline context.
- The `ds4-main` worktree tracks the latest upstream for comparison, merging,
  or rebasing.
- The push refspec is a **safety net**: even a fat-fingered `git push --all` or
  `git push origin` from the wrong worktree can't overwrite `main` on your
  public GitHub.

### Adding the Upstream Remote (if setting up fresh)

```sh
git remote add upstream https://github.com/antirez/ds4.git
git fetch upstream
```
