# REAP25 Speed Campaign Report — M2 Max, 2026-08-22/23

Autoresearch campaign to speed up DS4F REAP-ed 0731 (DeepSeek-V4-Flash
REAP25 IQ2XXS, 63.87 GiB) on Apple M2 Max 96GB, Metal backend. Goal: 8
active experts/token if fast enough, fallback to 6. 34 benchmark runs
logged, all gold-gated.

## Deliverables (in this branch)

Commit `20847cd` contains three kept source patches on top of `9e2edca`.

### Patch 1 — Hash-aware fused router finalize+weights

Files: `ds4_metal.m`, `metal/dsv4_misc.metal`

`kernel_dsv4_router_finalize_weights_one_simd` gains a `hash_mode` branch:
threads 0-5 read the token's `tid2eid` row directly instead of running the
bitonic selection; the weight-normalization epilogue is identical to the
top-k path. Host gate changed from `use_simd_finalize &&` to
`(use_simd_finalize || hash_mode) &&`.

Why: the three hash-routed layers (0-2) were excluded from the fused
router+weights dispatch and paid a generic two-dispatch path.

Interleaved A/B pairs: +1.9 / +1.9 / +2.9 t/s (candidate won 3/3).

### Patch 2 — Static attn-out-low Q8 decode kernel

Files: `metal/moe.metal`, `ds4_metal.m`

Ported from the ds4-metal fork (ivanfioravanti, commit 43ce3f9):
`kernel_dsv4_attn_out_low_q8_0_flash_decode_static_f32` — compile-time
literals for the exact Flash decode shape (8 groups x [4096->1024] Q8_0,
row bytes 4352). Same NR0=2/NSG=4 lane mapping and reduction tree as the
generic kernel — bit-exact by construction.

Kill switch: `DS4_METAL_DISABLE_ATTN_OUT_LOW_Q8_STATIC`. Falls back to
the generic kernel if PSO creation fails.

Interleaved pairs: +2.3 / +0.13 / +0.05 t/s (candidate won 3/3).
Note: only wired at the batch-decode site (~ds4_metal.m:24532); the TP
k-slice site (~25114) still uses generic.

### Patch 3 — REAP-aware router+shared fusion gate

File: `ds4.c`

The router+shared-gate/up fusion required `ffn_gate_inp dim[1] ==
DS4_N_EXPERT` (global max 256), silently excluding all 40 compact layers
(192 experts) from the fused path that hash layers enjoyed. Changed to
`reap_layer_expert_count(il)` and passes per-layer count as
`router_out_dim`; the kernel was already fully parameterized via
`args.ne01`.

Stage ledger confirms fusion engages (`shared_gate_up` 0.24 -> 0.024 ms/L
on compact layers). Wall-neutral in 4 interleaved pairs, but removes a
stock-model assumption that would bite any future REAP variant.

Also in working tree (uncommitted, stash "q-path instrumentation"):
2-line decode q-path sub-stage profiling (`q_a_norm` / `q_b_mat`
boundaries). Zero cost when off; kept for future sessions.

## The expert-cap decision matrix

The original goal asked for "8 active experts per token". Measured:
**DS4F top-k = 6** (`--inspect`: used=6). `--n-active-experts 8` is a
NO-OP by design (clamp returns n when k >= n). Early runs suggesting "8
experts faster" were a baseline-vs-baseline drift artifact, corrected in
run #12.

### Speed (tg128, non-thinking deepseek-chat)

| Depth | k=6 | k=4 | k=3 | Source |
|---|---|---|---|---|
| d1024 | 16.5–17.5 t/s | 17–19 t/s | 17–20+ t/s | runs #22-#23 |
| d8192 | 15.39 ± 0.38 | — | **18.22 ± 0.12** (+18%) | run #25 |

### Quality (perplexity, wikitext2)

| Corpus | Ctx | k=6 | k=4 | k=3 |
|---|---|---|---|---|
| prose | 8192 | 6.494 | 6.921 (+6.6%) | 7.981 (+22.9%) |
| prose | 32768 | 5.250 | 5.469 (+4.2%) | 6.030 (+14.8%) |
| code (ds4 sources) | 8192 | 2.489 | 2.548 (+2.4%) | 2.656 (+6.7%) |

### Key findings

- The speed advantage of lower caps GROWS with depth (+8% at d1024 ->
  +18% at d8192 for k=3) because routed-MoE bandwidth pressure compounds
  with KV attention.
- The quality penalty does NOT grow with depth; it is depth-stable or
  slightly improving (k=3: 22.9% -> 14.8% from ctx 8192 to 32768).
- On CODE, the penalty is much milder than on prose (+6.7% for k=3 vs
  +22.9% on prose at ctx 8192). Agent workloads may tolerate k=3 better
  than the prose numbers suggest.
- k <= 2 breaks coherence outright (benchy 'Paris' test fails at k=2;
  k=1 produces gibberish).

### Recommendation for LJ

- k=6: quality-safe default.
- k=4: defensible sweet spot (+5-8% speed, mild quality cost).
- k=3: attractive at agent-session depths (+18% speed at d8192, only
  +14.8% PPL / +6.7% code-PPL), but requires LJ's manual long-form A/B
  before adopting as a launcher default.
- k <= 2: not viable.

## What was tried and rejected

All rejected with evidence; full details in autoresearch run log #1-#34.

| Trial | Run | Result | Why |
|---|---|---|---|
| '8 experts faster' | #3-#5, corrected #12 | NO-OP | DS4F top-k=6; clamp returns n when k>=n. Early delta was drift artifact. |
| Split-layer sweep {0,2,4,8,16} | #6 | Default best | Interleaved A/B: default mean 17.36 vs split16 16.53. Knob closed on new code. |
| DS4_METAL_MODEL_UNTRACKED=1 | #7 | Regresses | Default 14.29 vs untracked 11.93 interleaved; drags PP too. |
| IQ2 row tile N_R0_IQ2_XXS 4->8 | #8 | FAILED GOLD | Output text diverged step 0, 319/320 logprobs wrong. Second rejection of this knob. |
| Greedy logits-readback skip | #11 | FAILED GOLD | s->logits is live API surface (--dump-logprobs reads it after every eval). Chain port must add GPU top-logprobs. |
| Attn inv-rope fuse on M2 Max | #20 | Neutral, reverted | Fuse engages (stage drops 0.33->0.295 ms/L) but wall time unchanged; upstream M3/M5 gate justified. |
| Static q_b matvec kernel | #29 | Neutral, reverted | Stage timing identical (0.326 vs 0.327 ms/L). Proved the 3.8x-over-floor gap is DRAM-pattern related, NOT address-arithmetic overhead. |
| DS4_METAL_Q8_MV_NSG {2,8} | #30 | FAILED GOLD | Non-default NSG reorders FP accumulation; only NSG=4 is bit-exact. |

## What was confirmed (no change needed)

| Check | Run | Result |
|---|---|---|
| Readback-summary (full-memory path) | #9 | Zero selected calls, zero splits, zero loads — generic GPU-ID MoE path confirmed. |
| Router-select fusion ON vs OFF | #14 | Fusion wins 3/3 interleaved; stays on. |
| Thinking-mode smoke | #31 | Coherent reasoning + correct answer through all patches; both workloads safe. |
| Cap equivalence (k=6 no-op check) | #10 | Confirmed k>=6 is true no-op; zero overhead when inactive. |

## Instrumentation added

Decode q-path sub-stage boundaries (working tree, uncommitted):

```
q_a_norm  0.235 ms/L   (qkv rms-norm rows)
q_b_mat   0.327 ms/L   ([1024->32768] Q8_0 matvec)
rope      0.162 ms/L   (head rms-norm + inv-rope tail)
```

Roofline finding: q_b_mat runs at 3.8x its bandwidth floor — worst ratio
in the token. Run #29 proved this is NOT address-arithmetic overhead;
it is DRAM access-pattern / bandwidth-contention related. Any future fix
needs novel lane mappings or K-splitting across threadgroups.

Full stage ledger after fixes (per compact layer, warm):

```
routed_moe 0.42 | q_path 0.41 | attn_output 0.36 (static kernel)
attn_inv_rope 0.32 | router 0.25 | shared_down 0.18 | kv_path 0.18 ms
```

## Lessons learned

1. **Verify semantics before attributing speed.** The biggest early error
   (runs #3-#5) was attributing a +17% delta to "8 experts" when the cap
   was a no-op. Always check what an option actually does at the model's
   shape before running benchmarks.
2. **Interleaved paired A/B is mandatory on this box.** Load average
   swings 3-6 from co-tenants; same-config TG ranged 9-19 t/s across the
   day. Only within-window pairs are trustworthy. Single runs lie.
3. **Gold gate catches things you can't see.** Three separate trials
   passed build and looked plausible but failed gold (tile-8, NSG != 4,
   logits-skip). The gate earned its keep every time.
4. **Instrumentation pays for itself immediately.** Adding two profiling
   boundaries (run #28) revealed the q_b_mat roofline anomaly that
   redirected the entire structural analysis. Cheap instrumentation first
   is always right.
5. **Static-trip specialization works when address arithmetic dominates**
   (attn-out-low: won), but not when the bottleneck is memory patterns
   (q_b_mat: neutral). Know why a trick works before porting it.
6. **REAP models break stock-model assumptions silently.** Two latent
   bugs found (fusion gate excluding compact layers; Metal kernels not
   handling -1 sentinel IDs). Every shape assumption needs auditing
   against tensor-authoritative counts.
7. **s->logits is API surface, not scratch.** Multiple consumers read it
   post-eval; any optimization touching the logits readback must account
   for all of them or add GPU-side equivalents first.

## What remains for next time

Prioritized; all are multi-day projects beyond this loop's scope:

1. **Greedy chain decode port** (from ds4-metal fork, commit 78269ce).
   GPU argmax ring removes per-token wait + 517 KiB logits readback
   (~0.5 ms/token on M3 Ultra; est ~1 t/s here since our CPU overhead is
   ~1.5/46 ms). Scope: devtoken router plumbing, hash-layer token view,
   session-chain entry, GPU top-logprobs (mandatory — see logits-skip
   rejection), kill switch. Est 1-2 days careful work.

2. **MoE multi-slot stream sharing** (structural kernel redesign).
   routed_moe runs at 0.24 TB/s effective vs ~400 GB/s peak (1.6x gap =
   ~7 ms/token theoretical max). Requires grouped expert-stream reads
   adapted for single-token decode. High effort, uncertain payoff.

3. **Novel q_b matvec mappings.** 3.8x over floor confirmed as
   memory-pattern, not addressing. Candidate approaches: vectorized
   float4 loads, K-splitting across threadgroups, different lane-to-row
   assignments. All multi-day, uncertain.

4. **Wire static out-low kernel into TP k-slice site** (~ds4_metal.m
   :25114) if TP decode matters.

5. **NPU/ANE exploration**: parked. ~/LJ-asi-mlx/ANE has private-API
   bridge with IOSurface zero-copy. ANE ~12 TFLOPS FP16 peak but tiny
   SRAM and conv-style layout limit it to small fixed ops. Low expected
   win vs GPU; high engineering cost.

6. **k=3 long-form quality evaluation**: requires LJ's manual A/B on
   representative tasks. Objective proxies (PPL, coherence) are done.

## Git topology note

This report lives on branch `reap-compact-speed` in the public repo
(`~/ds4/github/.git`). The top-level `~/ds4/.git` is the private repo
(notes/docs/harness); its `autoresearch/*` branch holds only the harness
commit. All campaign source work is in the public repo's worktrees.

Local `reap-compact-speed` @ `20847cd` has diverged from
`fork/reap-compact-speed` @ `4dfb564` (184 ahead / 17 behind). Pushing
requires LJ's decision on rebase vs merge.
