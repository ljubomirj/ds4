# Quantization Analysis: Ling-2.6-flash IQ4_NL

**File**: `Ling-2.6-flash-IQ4_NL-bailing_hybrid-20260505-LJ.gguf`
**Size**: 57 GB | **BPW**: 4.50 | **Tensors**: 540

---

## Current Quantization Mix

| Quant Type | Bit Width | Count | What |
|---|---|---|---|
| F32 | 32.00 | 228 | All LayerNorms, MoE routers (`ffn_gate_inp`), expert biases (`exp_probs_b`) |
| IQ4_NL | ~4.50 | 304 | Token embeddings, all attention tensors, most MoE FFN |
| Q5_K | ~5.40 | 7 | First 4 layers' `ffn_down` tensors (automatic upgrade) |
| Q6_K | ~6.60 | 1 | `output.weight` (LM head, automatic upgrade) |

---

## What's Protected (Correctly)

- **All norms** (attn_norm, ffn_norm, q_a_norm, kv_a_norm, etc.) → F32
- **MoE routers** (`ffn_gate_inp.weight`) → F32
- **Expert biases** (`exp_probs_b.bias`) → F32
- **Output head** (`output.weight`) → Q6_K (auto-upgraded from IQ4_NL)
- **First 4 layers' `ffn_down`** → Q5_K (auto-upgraded from IQ4_NL)

---

## What's NOT Protected (at IQ4_NL, should be higher)

### Token Embeddings — HIGHEST PRIORITY

| Tensor | Shape | Current | Recommended | Size Increase |
|---|---|---|---|---|
| `token_embd.weight` | 4096 × 157184 | IQ4_NL (~360MB) | Q8_0 (~600MB) | +240MB |

157K vocabulary at 4.5 bpw means each embedding vector is heavily compressed.
For long chain-of-thought and code generation, semantic precision in the embedding
layer is foundational. Q8_0 is the minimum acceptable precision.

### MLA Attention Tensors — COMPLETELY UNPROTECTED

The llama.cpp quantizer doesn't recognize MLA tensor naming patterns:

| Tensor | Pattern | Why Critical |
|---|---|---|
| `attn_kv_a_mqa.weight` | 4096 × 576 | MLA KV compression bottleneck into 512-dim latent space. Errors here directly degrade attention. |
| `attn_q_a.weight` | 4096 × 1536 | MLA Q down-projection. Compresses Q into 1536-dim latent space. |
| `attn_q_b.weight` | 1536 × 6144 | MLA Q up-projection. Expands Q from latent to full attention dimension. |
| `attn_k_b.weight` | 128 × 512 × 32 size 1 | Key expansion from latent. |
| `attn_v_b.weight` | 512 × 128 × 32 | Value expansion from latent. |

**Root cause**: The tensor category string matcher (`llama-quant.cpp:115-150`)
uses patterns like `attn_q.weight`, `attn_kv_b.weight`. MLA tensors use
`attn_q_a.weight`, `attn_kv_a_mqa.weight` — none match.

### GLA Attention Tensors

| Tensor | Shape | Layers |
|---|---|---|
| `attn_qkv.weight` (fused QKV) | 4096 × 12288 | 28 GLA layers |
| `attn_gate.weight` | 4096 × 4096 | 28 GLA layers |
| `attn_output.weight` | 4096 × 4096 | 33 (all layers) |

---

## Recommended Quality Upgrades

### Priority 1 (highest impact, +250MB)

```
--token-embedding-type q8_0       # +240MB — vocab precision
--tensor-type "attn_kv_a_mqa=q5_k" # +3MB — MLA KV bottleneck
--tensor-type "attn_q_a=q5_k"      # +10MB — MLA Q bottleneck
```

### Priority 2 (+400MB, total +650MB)

```
--tensor-type "attn_q_b=q6_k"     # +20MB
--tensor-type "attn_qkv=q5_k"     # +200MB — 28 GLA layers
--tensor-type "attn_output=q5_k"  # +200MB — 33 layers
```

### Priority 3 (+250MB, total +900MB)

```
--tensor-type "attn_gate=q5_k"    # +30MB
--output-tensor-type q8_0         # +240MB over current Q6_K
```

---

## Implementation Plan for l26f

1. **Re-quantize from F16 source** (not from IQ4_NL) to avoid double-quantization artifacts
2. Apply all Priority 1 overrides
3. Apply Priority 2 overrides where memory budget allows (we have ~2-3 GB headroom)
4. Build the l26f loader to work only with this specific quantization mix
5. No generic quantization support — ds4-style hardcoded weight loading

### Changed Tensors from Current GGUF (estimate)

| Change | Tensors Affected | Size Delta |
|---|---|---|
| token_embd: IQ4_NL → Q8_0 | 1 | +240 MB |
| all attn_kv_a_mqa: IQ4_NL → Q5_K | 5 | +3 MB |
| all attn_q_a: IQ4_NL → Q5_K | 5 | +10 MB |
| all attn_q_b: IQ4_NL → Q6_K | 5 | +20 MB |
| all attn_qkv (GLA): IQ4_NL → Q5_K | 28 | +200 MB |
| all attn_output: IQ4_NL → Q5_K | 33 | +200 MB |
| all attn_gate (GLA): IQ4_NL → Q5_K | 28 | +30 MB |
| **Total** | **105** | **~700 MB** (1.2% of 57 GB) |

### Total Model Size After Upgrades

From 57 GB → ~58 GB. Fits comfortably in 96 GB RAM.

---

## Note on Quantization Quality per Tensor Type

- **Embeddings**: Most sensitive. Every token lookup uses the embedding. 4.5 bpw
  loses significant semantic precision.
- **MLA latent projections** (q_a, kv_a): These compress into latent spaces of
  1536 and 512 dimensions respectively. Quantization noise in the compression
  projects through the expansion layers (q_b, k_b, v_b), amplifying errors.
- **GLA QKV**: Fused query/key/value, but GLA uses gating which may mask some
  quantization noise.
- **MoE FFN experts**: Acceptable at IQ4_NL. Large volume (256 experts) and
  routing (top-8 activation) dilutes individual expert quantization noise.
