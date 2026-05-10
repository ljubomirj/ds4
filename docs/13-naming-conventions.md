# Naming Conventions: Dimension-Annotated Variables

All tensor-like variables (weights, activations, states, buffers) carry their
dimensionality in the name as a suffix using `_DIM1xDIM2x...`.

## Dimension Key for Ling-2.6-flash (104B MoE)

| Symbol | Value | Meaning |
|--------|-------|---------|
| `N` | 4096 | Hidden dimension (n_embd) |
| `F` | ~    | Dense FFN intermediate (n_ff, model-dependent) |
| `M` | 1024 | Expert FFN intermediate (n_ff_exp) |
| `V` | 157184 | Vocabulary size (n_vocab) |
| `S` | 128 | GLA state dimension |
| `H` | 32 | Attention heads |
| `D` | 192 | Head dimension (head_dim = n_embd * 3 / n_head = 6144/32) |
| `P` | 128 | Non-positional part of head dim (qk_nope = D - R) |
| `Q` | 1536 | Q LoRA compression rank (q_lora_rank) |
| `C` | 512 | KV LoRA compression rank (kv_lora_rank) |
| `R` | 64 | RoPE dimension (n_rot) |
| `E` | 256 | Total MoE experts |
| `K` | 8 | Selected experts per token |
| `G` | 8 | Expert groups |
| `B` | (var) | Batch size / sequence length |
| `T` | (var) | Sequence position / token count |

## Suffix Rules

### 1. Separator: lowercase `x`

Dimensions are separated by lowercase `x`:

```
weight_NxM         -- 2D matrix: 4096 rows × 1024 cols
hidden_1xN         -- 1D vector: batch=1 × 4096
logits_1xV         -- 1D vector: batch=1 × 157184
gate_exps_NxMxE    -- 3D tensor: 4096 × 1024 × 256 experts
```

### 2. Leading `1` for batch-decode vectors

Single-token activations use `1` for the batch dimension:

```
normed_1xN         -- 1 token × 4096
qkv_1x3N          -- 1 token × 12288 (packed Q/K/V)
gla_out_1xNxSxSxH  -- 1 token × 4096 attn out + 128×128×32 state
```

### 3. Compound dimensions: CAPS concatenation, no separator

When a dimension is composed of two known quantities, concatenate their
symbols in ALL CAPS. This visually distinguishes a SINGLE merged dimension
from two separate dimensions:

```
kv_a_1xCR           -- [1, C+R] = [1, 576]    CR: one dimension
k_t_1xCR            -- [1, 576]
t_kv_a_mqa_NxCR     -- [4096, 576]
attn_q_kv.weight     would be [Q, CR] = [1536, 576]
```

**NOT**: `kv_a_1xCxR` — that reads as TWO dimensions [1, 512, 64], which is wrong.
**NOT**: `kv_a_1xCpR` — lowercase `p` is ambiguous ("C plus R" vs "C parallel R").

Examples of compound dimensions:
```
CR  = C + R  = 512 + 64 = 576            (kv_dim)
HD  = H * D  = 32 * 192 = 6144           (per-head Q total)
3N  = 3 * N  = 12288                      (packed QKV)
PE  = P + E  (if we had such a thing)
```

### 4. Unknown dimensions: lowercase `xU`

When a dimension exists but its symbolic letter hasn't been assigned:

```
buffer_1xU          -- 1-dim array, size unknown
scratch_NxUxU       -- 3-dim tensor, 2 dims unknown
```

If we know it's 2D/3D/4D but absolutely nothing about the dimensions:

```
temp_xUUU           -- unknown rank and all dims unknown
```

### 5. Omission and abbreviation

Never omit `1` for vectors. Always `hidden_1xN`, not `hidden_N`.
Never abbreviate when a single letter exists. Use `qkv_1x3N`, not `qkv_1x3xN`.

## Weight Variable Naming

Weight tensors (l26f_tensor lookups) use `wt_` prefix followed by name and shape:

```
wt_norm_N               -- [4096]      RMS norm weight
wt_qkv_Nx3N             -- [4096, 12288]  Packed Q/K/V projection
wt_gate_NxN             -- [4096, 4096]  GLA gate projection
wt_down_FxN             -- [F, 4096]     Dense FFN down projection
wt_gate_exps_NxMxE      -- [4096, 1024, 256]  MoE expert gates
```

MLA tensors use `t_` prefix (CPU-side lookup):

```
t_attn_norm_N           -- [4096]
t_q_a_NxQ               -- [4096, 1536]
t_q_b_QxHxD             -- [1536, 6144]
t_kv_a_mqa_NxCR         -- [4096, 576]
t_k_b_PxCxH             -- [128, 512, 32]
t_v_b_CxPxH             -- [512, 128, 32]
t_output_NxN            -- [4096, 4096]
```

## Buffer / Compute Struct Naming

GPU buffers in `l26f_compute` carry dimension suffixes:

```
normed_1xN              -- RMS-normed input
qkv_1x3N               -- Packed Q/K/V
gate_out_1xN            -- GLA gate output
gla_out_1xNxSxSxH       -- GLA attention output + state
attn_proj_1xN           -- Attention output projection
ffn_gate_1xF            -- FFN gate projection (dense)
ffn_gate_1xM            -- FFN gate projection (expert)
moe_out_1xN             -- Accumulated MoE output
shexp_out_1xN           -- Shared expert output
```

Session buffers:

```
hidden_1xN              -- Current hidden state
output_normed_1xN       -- Final RMS-normed state
logits_1xV              -- Output logits
```

## CPU-Side Activation Naming

In `l26f_mla_cpu.c`, local scratch arrays:

```
normed_1xN              -- RMS-normed input
q_a_1xQ                -- Compressed Q
q_a_normed_1xQ          -- RMS-normed compressed Q
q_b_1xHxD              -- Expanded Q per head
kv_a_1xCR              -- Compressed KV
kv_cmpr_1xC             -- KV compressed part only
q_absorbed_HxC          -- Q after K-absorption (per head)
v_decomp_HxP            -- V after decompression (per head)
attn_result_HxC         -- Attention result (per head)
scores_1xT              -- Attention scores [1, n_cached]
```

## Pointer Slices Within Named Buffers

When taking a pointer to a sub-region of a dimension-annotated buffer,
annotate the pointer with the slice shape:

```
k_pe_R = kv_a_1xCR + C;         -- last R=64 elements of kv_a
q_nope_h_P = q_b_1xHxD + h*D;   -- first P=128 elements of head h
k_t_1xCR = data_TxCR + t*CR;    -- row t of KV cache
```

## MoE Routing Arrays

```
router_logits_1xE[256]          -- Router logits [1, n_expert]
probs_1xE[256]                  -- Softmax'd probabilities
group_scores_1xG[8]             -- Per-group top-2 sum
selected_groups_1xG[8]          -- Top-4 group indices
masked_probs_1xE[256]           -- Group-masked probs
selected_experts_K[8]           -- Final top-8 expert indices
selected_weights_K[8]           -- Normalized expert weights
```

## Constants and Locals

Shape constants keep short names when the meaning is clear from context:

```
const uint32_t S = 128, H = 32;          -- GLA shape
const uint32_t n_embd = m->n_embd;       -- alias for convenience
const uint32_t n_ff_exp = 1024;          -- expert FFN intermediate
const uint32_t n_expert = 256;           -- total experts
const uint32_t n_groups = 8;             -- expert groups
const uint32_t n_exp_per_group = 32;     -- experts per group
```

These are local aliases, NOT dimensional annotations on the variable name itself.
They avoid repeating `m->n_embd` everywhere while keeping functions readable.

## Anti-Patterns

```
// WRONG — dimension not in suffix
float *q_a;

// WRONG — `p` looks like an operator or "parallel"
float *kv_a_1xCpR;

// WRONG — compound dimension uses `x` like it's two separate dims
float *kv_a_1xCxR;

// WRONG — missing leading `1`
float *hidden_N;

// WRONG — ambiguous compound vs separate
float *weight_QxHxD;   // Is this Q × H × D, or (Q×H) × D?
                        // Answer: it's 3 separate dims, so this is correct.
                        // Compound only when it's ONE dimension that happens
                        // to be the sum/product of two known quantities.
```
