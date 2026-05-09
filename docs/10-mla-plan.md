# MLA Implementation Plan for l26f

## Architecture Parameters

```
n_embd       = 4096
n_head       = 32
q_lora_rank  = 1536
kv_lora_rank = 512
n_rot        = 64
n_embd_head_qk_nope = 128  (= n_embd / n_head - n_rot = 192 - 64)
n_embd_head_v        = 128
```

Wait: n_head=32, n_embd=4096. The attn_q_b output dim is 6144 = 32 * 192.
So head_dim = 6144/32 = 192. qk_nope = 192 - 64 = 128. That matches attn_k_b dims [128,512,32].

## MLA Layer Tensors

| Tensor | Shape | Type | Bytes |
|--------|-------|------|-------|
| attn_norm | [4096] | F32 | 16 KiB |
| attn_q_a | [4096, 1536] | Q5_K (13) | 4.1 MiB |
| attn_q_a_norm | [1536] | F32 | 6 KiB |
| attn_q_b | [1536, 6144] | Q6_K (14) | 7.4 MiB |
| attn_kv_a_mqa | [4096, 576] | Q5_K (13) | 1.5 MiB |
| attn_kv_a_norm | [512] | F32 | 2 KiB |
| attn_k_b | [128, 512, 32] | IQ4_NL (20) | 1.1 MiB |
| attn_v_b | [512, 128, 32] | IQ4_NL (20) | 1.1 MiB |
| attn_output | [4096, 4096] | Q5_K (13) | 11 MiB |
| FFN (same MoE as GLA layers) | | | |

## Decode Flow (single token, MLA absorbed form)

```
Input: hidden [4096]

1. RMS norm: normed = RMSNorm(hidden, attn_norm)              [4096]

2. Q compression:
   q_a = matvec(normed, wq_a)                                 [1536]     Q5_K
   q_a_normed = RMSNorm(q_a, attn_q_a_norm)                   [1536]
   q_b = matvec(q_a_normed, wq_b)                             [6144]     Q6_K
   Split: q_nope = q_b[0:4096]  → [128, 32]  (reshape)
          q_pe   = q_b[4096:6144] → [64, 32]  (reshape)

3. KV compression:
   kv_a = matvec(normed, wkv_a_mqa)                            [576]      Q5_K
   Split: kv_cmpr = kv_a[0:512]  → [512]
          k_pe    = kv_a[512:576] → [64]
   kv_cmpr = RMSNorm(kv_cmpr, attn_kv_a_norm)                  [512]

4. RoPE:
   q_pe = RoPE(q_pe, position)     → [64, 32]
   k_pe = RoPE(k_pe, position)     → [64]

5. Absorption (key MLA trick — avoid decompressing K for every cached token):
   For each head h:
     q_nope_absorbed[h] = wk_b[:,:,h] × q_nope[:,h]    → [512]
     // wk_b is [128, 512], q_nope[:,h] is [128], result is [512]
   // This is a batched matvec: 32 heads × [512,128] × [128] = [512,32]

6. KV cache update:
   K_new = concat(kv_cmpr, k_pe)  → [576]    (only 576 floats per token!)
   // NO V cache — V is derived from the compressed latent during attention

7. Attention (MQA — 1 KV head, 32 Q heads):
   For each head h:
     Q_h = concat(q_nope_absorbed[h], q_pe[:,h])  → [576]
     scores = Q_h · K_cache^T / sqrt(576)          → [n_cached]
     weights = softmax(scores)                      → [n_cached]
     attn_out_h = weights · V_cache                 → [512]  (V_cache = kv_cmpr part of K)

8. V decompression:
   For each head h:
     v_decomp[h] = wv_b[:,:,h] × attn_out_h        → [128]
     // wv_b is [512, 128], attn_out is [512], result is [128]

9. Output projection:
   concat all v_decomp → [4096]
   out = matvec(concat_v, wo)                       [4096]     Q5_K

10. Residual: hidden = hidden + out
11. MoE FFN (same as GLA layers)
```

## What We Already Have

- ✅ RMS norm kernel
- ✅ Q5_K matvec (for wq_a, wkv_a_mqa, wo)
- ✅ Q6_K matvec (for wq_b)
- ✅ IQ4_NL matvec (for wk_b, wv_b — but these are per-head batched, need special handling)
- ✅ MoE FFN (same as GLA layers)
- ❌ RoPE kernel
- ❌ KV cache (simple CPU-side ring buffer for decode)
- ❌ Softmax (CPU is fine for single-token decode)
- ❌ Absorption matmul (batched [512,128]×[128] per head — can be 32 individual matvecs or one batched)
- ❌ V decompression (batched [512,128]×[512] per head)

## Implementation Strategy

### Phase A: CPU-only MLA decode
Get correctness first. Do all MLA operations on CPU using the mmap'd model data.
This avoids writing Metal kernels for RoPE, softmax, attention, and batched matmul.

Steps:
1. Add MLA-specific compute buffers to `l26f_session` (q_a, q_b, kv_a, etc.)
2. Implement `l26f_mla_layer()` that does steps 1–10 on CPU
   - Read quantized weights from mmap
   - Dequantize on CPU (reference implementations from ggml-quants.c)
   - All compute on CPU
   - Write result back to `hidden` GPU tensor
3. Wire into main loop (replace `continue` for MLA layers)
4. Verify with checksums (compare against llama.cpp output if possible)

### Phase B: GPU-accelerate the hot path
Once CPU version is correct, move the expensive ops to GPU:
1. RoPE Metal kernel
2. Batched IQ4_NL matmul for absorption and V decompression
3. Flash attention for MLA (or simple CPU attention for single-token decode)

### Phase C: KV cache
Add a proper KV cache for multi-token generation (task 5).

## Compute Buffers Needed for MLA

```
normed:       [4096]        F32    (reuse existing)
q_a:          [1536]        F32
q_a_normed:   [1536]        F32
q_b:          [6144]        F32
q_nope:       [4096]        F32    (view into q_b first 4096)
q_pe:         [2048]        F32    (view into q_b last 2048)
kv_a:         [576]         F32
kv_cmpr:      [512]         F32    (view into kv_a first 512)
k_pe:         [64]           F32   (view into kv_a last 64)
kv_cmpr_norm: [512]         F32
q_absorbed:   [512*32]      F32    (absorption output, per-head)
kv_cache:     [576*max_seq] F32    (per MLA layer, persists across tokens)
attn_out:     [512*32]      F32    (attention output before V decompress)
v_decomp:     [128*32]      F32    (= 4096, output projection input)
attn_proj:    [4096]        F32    (reuse existing)
```

For Phase A (CPU-only), we can skip GPU tensors entirely and just use CPU arrays.
