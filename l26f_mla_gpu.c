// l26f: GPU-accelerated MLA (Multi-head Latent Attention) decode
// Replaces l26f_mla_cpu.c for GPU execution path.
//
// Data flow (all on GPU unless noted):
//   1. RMS norm (attn_norm)     → normed_1xN
//   2. Q compression (Q5_K)     → q_a_1xQ
//   3. RMS norm (q_a_norm)      → q_a_normed_1xQ
//   4. Q expansion (Q6_K)       → q_b_1xHxD
//   5. RoPE on q_pe              → q_pe_1xHxR  (batched)
//   6. KV compression (Q5_K)    → kv_a_1xCR
//   7. Split kv_a → kv_cmpr + k_pe, RMS norm on kv_cmpr, RoPE on k_pe
//   8. Absorption (batched IQ4_NL) → q_absorbed_HxC
//   9. KV cache update (CPU: memcpy into cache, copy cache to GPU)
//  10. Attention (MQA)            → attn_result_HxC
//  11. V decompression (batched IQ4_NL) → v_decomp_HxP
//  12. Concat heads → out_proj_in_1xN
//  13. Output projection (Q5_K)  → attn_proj_1xN
//  14. Residual add

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "l26f.h"
#include "l26f_metal.h"
#include "ds4_metal.h"

// Debug checksum helper — only reads GPU when L26F_MLA_DEBUG is defined
#ifndef L26F_MLA_DEBUG
static float mla_ckpt_sum(const ds4_metal_tensor *t, uint64_t bytes, const char *label, uint32_t layer) {
    (void)t; (void)bytes; (void)label; (void)layer;
    return 0.0f;
}
#else
static float mla_ckpt_sum(const ds4_metal_tensor *t, uint64_t bytes, const char *label, uint32_t layer) {
    if (!t || bytes == 0) return 0.0f;
    float *d = (float *)malloc(bytes);
    if (!d) return 0.0f;
    ds4_metal_tensor_read(t, 0, d, bytes);
    float sum = 0;
    for (uint64_t i = 0; i < bytes / sizeof(float); i++) sum += d[i];
    if (layer == 7) fprintf(stderr, "    MLA-GPU L%u %-20s sum=%.6f\n", layer, label, sum);
    free(d);
    return sum;
}
#endif

// ---- MLA KV cache (CPU-side, copied to GPU for attention) ----

typedef struct {
    float *data_TxCR;
    int    max_seq_len;
    int    kv_dim;
    int    n_tokens;
    ds4_metal_tensor *gpu_cache;
} l26f_mla_kv_cache_gpu;

l26f_mla_kv_cache_gpu *l26f_mla_kv_cache_gpu_alloc(int max_seq, int kv_dim) {
    l26f_mla_kv_cache_gpu *c = (l26f_mla_kv_cache_gpu *)calloc(1, sizeof(*c));
    c->data_TxCR = (float *)calloc((uint64_t)max_seq * kv_dim, sizeof(float));
    c->max_seq_len = max_seq;
    c->kv_dim = kv_dim;
    c->n_tokens = 0;
    c->gpu_cache = ds4_metal_tensor_alloc((uint64_t)max_seq * kv_dim * sizeof(float));
    if (c->gpu_cache) {
        void *zeros = calloc(1, (uint64_t)max_seq * kv_dim * sizeof(float));
        if (zeros) {
            ds4_metal_tensor_write(c->gpu_cache, 0, zeros, (uint64_t)max_seq * kv_dim * sizeof(float));
            free(zeros);
        }
    }
    return c;
}

void l26f_mla_kv_cache_gpu_free(l26f_mla_kv_cache_gpu *c) {
    if (!c) return;
    free(c->data_TxCR);
    if (c->gpu_cache) ds4_metal_tensor_free(c->gpu_cache);
    free(c);
}

// ---- MLA GPU compute buffers (allocated once, reused across layers/tokens) ----

typedef struct l26f_mla_compute {
    ds4_metal_tensor *q_a_1xQ;
    ds4_metal_tensor *q_a_normed_1xQ;
    ds4_metal_tensor *q_b_1xHxD;
    ds4_metal_tensor *q_pe_1xHxR;
    ds4_metal_tensor *kv_a_1xCR;
    ds4_metal_tensor *kv_cmpr_1xC;
    ds4_metal_tensor *k_pe_1xR;
    ds4_metal_tensor *q_absorbed_HxC;
    ds4_metal_tensor *attn_result_HxC;
    ds4_metal_tensor *v_decomp_HxP;
    ds4_metal_tensor *out_proj_in_1xN;
    ds4_metal_tensor *attn_proj_1xN;
    ds4_metal_tensor *normed_1xN;
} l26f_mla_compute;

l26f_mla_compute *l26f_mla_compute_alloc(uint32_t n_embd, uint32_t n_head,
                                                  uint32_t q_lora_rank, uint32_t kv_lora_rank,
                                                  uint32_t n_rot, uint32_t head_dim) {
    l26f_mla_compute *mc = (l26f_mla_compute *)calloc(1, sizeof(*mc));
    if (!mc) return NULL;

    const uint32_t qk_nope = head_dim - n_rot;
    const uint32_t kv_dim = kv_lora_rank + n_rot;

    mc->q_a_1xQ         = ds4_metal_tensor_alloc((uint64_t)q_lora_rank * sizeof(float));
    mc->q_a_normed_1xQ  = ds4_metal_tensor_alloc((uint64_t)q_lora_rank * sizeof(float));
    mc->q_b_1xHxD       = ds4_metal_tensor_alloc((uint64_t)n_head * head_dim * sizeof(float));
    mc->q_pe_1xHxR      = ds4_metal_tensor_alloc((uint64_t)n_head * n_rot * sizeof(float));
    mc->kv_a_1xCR       = ds4_metal_tensor_alloc((uint64_t)kv_dim * sizeof(float));
    mc->kv_cmpr_1xC     = ds4_metal_tensor_alloc((uint64_t)kv_lora_rank * sizeof(float));
    mc->k_pe_1xR        = ds4_metal_tensor_alloc((uint64_t)n_rot * sizeof(float));
    mc->q_absorbed_HxC  = ds4_metal_tensor_alloc((uint64_t)n_head * kv_lora_rank * sizeof(float));
    mc->attn_result_HxC = ds4_metal_tensor_alloc((uint64_t)n_head * kv_lora_rank * sizeof(float));
    mc->v_decomp_HxP    = ds4_metal_tensor_alloc((uint64_t)n_head * qk_nope * sizeof(float));
    mc->out_proj_in_1xN = ds4_metal_tensor_alloc((uint64_t)n_embd * sizeof(float));
    mc->attn_proj_1xN   = ds4_metal_tensor_alloc((uint64_t)n_embd * sizeof(float));
    mc->normed_1xN      = ds4_metal_tensor_alloc((uint64_t)n_embd * sizeof(float));

    if (!mc->q_a_1xQ || !mc->q_a_normed_1xQ || !mc->q_b_1xHxD || !mc->q_pe_1xHxR ||
        !mc->kv_a_1xCR || !mc->kv_cmpr_1xC || !mc->k_pe_1xR || !mc->q_absorbed_HxC ||
        !mc->attn_result_HxC || !mc->v_decomp_HxP || !mc->out_proj_in_1xN ||
        !mc->attn_proj_1xN || !mc->normed_1xN) {
        fprintf(stderr, "l26f_mla: OOM compute buffers\n");
        free(mc);
        return NULL;
    }

    ds4_metal_tensor_fill(mc->q_a_1xQ, 0.0f);
    ds4_metal_tensor_fill(mc->q_a_normed_1xQ, 0.0f);
    ds4_metal_tensor_fill(mc->q_b_1xHxD, 0.0f);
    ds4_metal_tensor_fill(mc->q_pe_1xHxR, 0.0f);
    ds4_metal_tensor_fill(mc->kv_a_1xCR, 0.0f);
    ds4_metal_tensor_fill(mc->kv_cmpr_1xC, 0.0f);
    ds4_metal_tensor_fill(mc->k_pe_1xR, 0.0f);
    ds4_metal_tensor_fill(mc->q_absorbed_HxC, 0.0f);
    ds4_metal_tensor_fill(mc->attn_result_HxC, 0.0f);
    ds4_metal_tensor_fill(mc->v_decomp_HxP, 0.0f);
    ds4_metal_tensor_fill(mc->out_proj_in_1xN, 0.0f);
    ds4_metal_tensor_fill(mc->attn_proj_1xN, 0.0f);
    ds4_metal_tensor_fill(mc->normed_1xN, 0.0f);

    return mc;
}

void l26f_mla_compute_free(l26f_mla_compute *mc) {
    if (!mc) return;
    ds4_metal_tensor_free(mc->q_a_1xQ);
    ds4_metal_tensor_free(mc->q_a_normed_1xQ);
    ds4_metal_tensor_free(mc->q_b_1xHxD);
    ds4_metal_tensor_free(mc->q_pe_1xHxR);
    ds4_metal_tensor_free(mc->kv_a_1xCR);
    ds4_metal_tensor_free(mc->kv_cmpr_1xC);
    ds4_metal_tensor_free(mc->k_pe_1xR);
    ds4_metal_tensor_free(mc->q_absorbed_HxC);
    ds4_metal_tensor_free(mc->attn_result_HxC);
    ds4_metal_tensor_free(mc->v_decomp_HxP);
    ds4_metal_tensor_free(mc->out_proj_in_1xN);
    ds4_metal_tensor_free(mc->attn_proj_1xN);
    ds4_metal_tensor_free(mc->normed_1xN);
    free(mc);
}

// ---- Main GPU MLA decode function ----
//
// hidden_1xN: GPU tensor with input hidden state [n_embd]
// out_1xN:    GPU tensor for output (after residual add) [n_embd]
// Returns 1 on success.

int l26f_mla_layer_gpu(
        l26f_model *m, uint32_t layer, int position,
        l26f_mla_kv_cache_gpu *kv_cache,
        l26f_mla_compute *mc,
        ds4_metal_tensor *hidden_1xN,
        ds4_metal_tensor *out_1xN)
{
    const uint32_t n_embd       = m->n_embd;
    const uint32_t n_head       = m->n_head;
    const uint32_t q_lora_rank  = m->q_lora_rank;
    const uint32_t kv_lora_rank = m->kv_lora_rank;
    const uint32_t n_rot        = m->n_rot;
    const uint32_t head_dim     = 192;
    const uint32_t qk_nope      = head_dim - n_rot;
    const uint32_t kv_dim       = kv_lora_rank + n_rot;
    const float eps             = m->rms_norm_eps;
    const float theta           = m->rope_theta;

    // Find tensors
    char name[128];
    l26f_tensor *t;

    snprintf(name, sizeof(name), "blk.%u.attn_norm.weight", layer);
    t = l26f_model_find_tensor(m, name);
    l26f_tensor *t_attn_norm_N = t;

    snprintf(name, sizeof(name), "blk.%u.attn_q_a.weight", layer);
    t = l26f_model_find_tensor(m, name);
    l26f_tensor *t_q_a_NxQ = t;

    snprintf(name, sizeof(name), "blk.%u.attn_q_a_norm.weight", layer);
    t = l26f_model_find_tensor(m, name);
    l26f_tensor *t_q_a_norm_Q = t;

    snprintf(name, sizeof(name), "blk.%u.attn_q_b.weight", layer);
    t = l26f_model_find_tensor(m, name);
    l26f_tensor *t_q_b_QxHxD = t;

    snprintf(name, sizeof(name), "blk.%u.attn_kv_a_mqa.weight", layer);
    t = l26f_model_find_tensor(m, name);
    l26f_tensor *t_kv_a_mqa_NxCR = t;

    snprintf(name, sizeof(name), "blk.%u.attn_kv_a_norm.weight", layer);
    t = l26f_model_find_tensor(m, name);
    l26f_tensor *t_kv_a_norm_C = t;

    snprintf(name, sizeof(name), "blk.%u.attn_k_b.weight", layer);
    t = l26f_model_find_tensor(m, name);
    l26f_tensor *t_k_b_PxCxH = t;

    snprintf(name, sizeof(name), "blk.%u.attn_v_b.weight", layer);
    t = l26f_model_find_tensor(m, name);
    l26f_tensor *t_v_b_CxPxH = t;

    snprintf(name, sizeof(name), "blk.%u.attn_output.weight", layer);
    t = l26f_model_find_tensor(m, name);
    l26f_tensor *t_output_NxN = t;

    if (!t_attn_norm_N || !t_q_a_NxQ || !t_q_a_norm_Q || !t_q_b_QxHxD ||
        !t_kv_a_mqa_NxCR || !t_kv_a_norm_C || !t_k_b_PxCxH ||
        !t_v_b_CxPxH || !t_output_NxN) {
        fprintf(stderr, "l26f: MLA GPU layer %u missing tensors\n", layer);
        return 0;
    }

    // --- Step 1: RMS norm ---
    if (!ds4_metal_rms_norm_weight_tensor(mc->normed_1xN, hidden_1xN,
            m->map, m->size, t_attn_norm_N->abs_offset, n_embd, eps))
        return 0;
    mla_ckpt_sum(mc->normed_1xN, n_embd * sizeof(float), "normed", layer);

    // --- Step 2: Q compression (Q5_K matvec) ---
    if (!l26f_metal_matvec_quant(mc->q_a_1xQ, mc->normed_1xN,
            m->map, m->size, t_q_a_NxQ->abs_offset,
            t_q_a_NxQ->dim[0], t_q_a_NxQ->dim[1], t_q_a_NxQ->type, 1))
        return 0;
    mla_ckpt_sum(mc->q_a_1xQ, q_lora_rank * sizeof(float), "q_a", layer);

    // --- Step 3: RMS norm on Q ---
    if (!ds4_metal_rms_norm_weight_tensor(mc->q_a_normed_1xQ, mc->q_a_1xQ,
            m->map, m->size, t_q_a_norm_Q->abs_offset, q_lora_rank, eps))
        return 0;
    mla_ckpt_sum(mc->q_a_normed_1xQ, q_lora_rank * sizeof(float), "q_a_normed", layer);

    // --- Step 4: Q expansion (Q6_K matvec) ---
    if (!l26f_metal_matvec_quant(mc->q_b_1xHxD, mc->q_a_normed_1xQ,
            m->map, m->size, t_q_b_QxHxD->abs_offset,
            t_q_b_QxHxD->dim[0], t_q_b_QxHxD->dim[1], t_q_b_QxHxD->type, 1))
        return 0;
    mla_ckpt_sum(mc->q_b_1xHxD, (uint64_t)n_head * head_dim * sizeof(float), "q_b", layer);

    // --- Step 5: RoPE on q_pe ---
    // q_b layout: [n_head * head_dim] = for each head, first qk_nope=128 then n_rot=64
    // We need to extract q_pe for each head and apply RoPE.
    // q_pe[h] starts at q_b[h * head_dim + qk_nope], length n_rot.
    // For simplicity, we copy q_pe to a contiguous buffer first.
    // TODO: could write a fused kernel that extracts + RoPEs in one pass.
    {
        float *q_b_cpu = (float *)malloc((uint64_t)n_head * head_dim * sizeof(float));
        float *q_pe_cpu = (float *)malloc((uint64_t)n_head * n_rot * sizeof(float));
        if (!q_b_cpu || !q_pe_cpu) { free(q_b_cpu); free(q_pe_cpu); return 0; }

        ds4_metal_tensor_read(mc->q_b_1xHxD, 0, q_b_cpu,
                              (uint64_t)n_head * head_dim * sizeof(float));

        for (uint32_t h = 0; h < n_head; h++) {
            memcpy(q_pe_cpu + h * n_rot,
                   q_b_cpu + h * head_dim + qk_nope,
                   n_rot * sizeof(float));
        }

        ds4_metal_tensor_write(mc->q_pe_1xHxR, 0, q_pe_cpu,
                               (uint64_t)n_head * n_rot * sizeof(float));

        free(q_b_cpu);
        free(q_pe_cpu);
    }

    // Apply RoPE to q_pe (batched: n_head vectors of n_rot dims each)
    if (!l26f_metal_rope_batch(mc->q_pe_1xHxR, mc->q_pe_1xHxR,
                                n_rot, n_head, position, theta))
        return 0;
    mla_ckpt_sum(mc->q_pe_1xHxR, (uint64_t)n_head * n_rot * sizeof(float), "q_pe_roped", layer);

    // --- Step 6: KV compression (Q5_K matvec) ---
    if (!l26f_metal_matvec_quant(mc->kv_a_1xCR, mc->normed_1xN,
            m->map, m->size, t_kv_a_mqa_NxCR->abs_offset,
            t_kv_a_mqa_NxCR->dim[0], t_kv_a_mqa_NxCR->dim[1], t_kv_a_mqa_NxCR->type, 1))
        return 0;
    mla_ckpt_sum(mc->kv_a_1xCR, kv_dim * sizeof(float), "kv_a", layer);

    // --- Step 7: Split kv_a → kv_cmpr + k_pe, RMS norm kv_cmpr, RoPE k_pe ---
    // kv_a layout: [kv_dim] = first kv_lora_rank=512 floats are kv_cmpr, last n_rot=64 are k_pe
    {
        float *kv_a_cpu = (float *)malloc(kv_dim * sizeof(float));
        if (!kv_a_cpu) return 0;

        ds4_metal_tensor_read(mc->kv_a_1xCR, 0, kv_a_cpu, kv_dim * sizeof(float));

        float *kv_cmpr_cpu = kv_a_cpu;
        float *k_pe_cpu    = kv_a_cpu + kv_lora_rank;

        ds4_metal_tensor_write(mc->kv_cmpr_1xC, 0, kv_cmpr_cpu,
                               kv_lora_rank * sizeof(float));
        ds4_metal_tensor_write(mc->k_pe_1xR, 0, k_pe_cpu,
                               n_rot * sizeof(float));

        free(kv_a_cpu);
    }

    if (!ds4_metal_rms_norm_weight_tensor(mc->kv_cmpr_1xC, mc->kv_cmpr_1xC,
            m->map, m->size, t_kv_a_norm_C->abs_offset, kv_lora_rank, eps))
        return 0;
    mla_ckpt_sum(mc->kv_cmpr_1xC, kv_lora_rank * sizeof(float), "kv_cmpr_normed", layer);

    if (!l26f_metal_rope(mc->k_pe_1xR, mc->k_pe_1xR, n_rot, position, theta))
        return 0;
    mla_ckpt_sum(mc->k_pe_1xR, n_rot * sizeof(float), "k_pe_roped", layer);

    // --- Step 8: Absorption ---
    // wk_b layout: [P, C, H] = [128, 512, 32] IQ4_NL
    // For each head h: q_absorbed[h] = wk_b[h] × q_nope[h]
    // Input: q_nope per head = q_b[h * head_dim ... h * head_dim + qk_nope - 1]
    // We need contiguous per-head input for the batched kernel.
    {
        float *q_b_cpu = (float *)malloc((uint64_t)n_head * head_dim * sizeof(float));
        float *q_nope_cpu = (float *)malloc((uint64_t)n_head * qk_nope * sizeof(float));
        if (!q_b_cpu || !q_nope_cpu) { free(q_b_cpu); free(q_nope_cpu); return 0; }

        ds4_metal_tensor_read(mc->q_b_1xHxD, 0, q_b_cpu,
                              (uint64_t)n_head * head_dim * sizeof(float));

        for (uint32_t h = 0; h < n_head; h++) {
            memcpy(q_nope_cpu + h * qk_nope,
                   q_b_cpu + h * head_dim,
                   qk_nope * sizeof(float));
        }

        // Write q_nope into a temporary GPU tensor — reuse q_pe_1xHxR buffer
        // since we've already extracted q_pe. Actually, we need a separate buffer.
        // Let's reuse v_decomp_HxP as temp (it's n_head * 128 = n_head * qk_nope)
        ds4_metal_tensor_write(mc->v_decomp_HxP, 0, q_nope_cpu,
                               (uint64_t)n_head * qk_nope * sizeof(float));

        free(q_b_cpu);
        free(q_nope_cpu);

        // Compute head_stride for wk_b: [P, C, H] → each head slice is [P, C]
        // C rows × P cols of IQ4_NL → C * (P/32) * 18 bytes
        const uint64_t head_bytes_k = (uint64_t)t_k_b_PxCxH->dim[1] *
            ((uint64_t)t_k_b_PxCxH->dim[0] / 32) * 18;

        if (!l26f_metal_batch_iq4_nl_matvec(mc->q_absorbed_HxC,
                m->map, m->size,
                t_k_b_PxCxH->abs_offset,
                t_k_b_PxCxH->dim[0],     // in_dim = P = 128
                t_k_b_PxCxH->dim[1],     // out_rows = C = 512
                n_head,
                head_bytes_k,
                mc->v_decomp_HxP))       // input (q_nope per head)
            return 0;
        mla_ckpt_sum(mc->q_absorbed_HxC, (uint64_t)n_head * kv_lora_rank * sizeof(float), "q_absorbed", layer);
    }

    // --- Step 9: KV cache update (CPU) ---
    {
        float kv_cmpr_cpu[512];
        float k_pe_cpu[64];
        ds4_metal_tensor_read(mc->kv_cmpr_1xC, 0, kv_cmpr_cpu, kv_lora_rank * sizeof(float));
        ds4_metal_tensor_read(mc->k_pe_1xR, 0, k_pe_cpu, n_rot * sizeof(float));

        float *k_new = kv_cache->data_TxCR + (uint64_t)kv_cache->n_tokens * kv_dim;
        memcpy(k_new, kv_cmpr_cpu, kv_lora_rank * sizeof(float));
        memcpy(k_new + kv_lora_rank, k_pe_cpu, n_rot * sizeof(float));
        kv_cache->n_tokens++;

        // Upload entire cache to GPU
        uint64_t cache_bytes = (uint64_t)kv_cache->n_tokens * kv_dim * sizeof(float);
        ds4_metal_tensor_write(kv_cache->gpu_cache, 0, kv_cache->data_TxCR, cache_bytes);
    }

    // --- Step 10: Attention (MQA) ---
    {
        const float attn_scale = 1.0f / sqrtf((float)kv_dim);

        if (!l26f_mla_attn(mc->attn_result_HxC,
                mc->q_absorbed_HxC, mc->q_pe_1xHxR, kv_cache->gpu_cache,
                n_head, kv_lora_rank, n_rot,
                (uint32_t)kv_cache->n_tokens, attn_scale))
            return 0;
        mla_ckpt_sum(mc->attn_result_HxC, (uint64_t)n_head * kv_lora_rank * sizeof(float), "attn_result", layer);
    }

    // --- Step 11: V decompression ---
    // wv_b layout: [C, P, H] = [512, 128, 32] IQ4_NL
    // For each head h: v_decomp[h] = wv_b[h] × attn_result[h]
    {
        const uint64_t head_bytes_v = (uint64_t)t_v_b_CxPxH->dim[1] *
            ((uint64_t)t_v_b_CxPxH->dim[0] / 32) * 18;

        if (!l26f_metal_batch_iq4_nl_matvec(mc->v_decomp_HxP,
                m->map, m->size,
                t_v_b_CxPxH->abs_offset,
                t_v_b_CxPxH->dim[0],     // in_dim = C = 512
                t_v_b_CxPxH->dim[1],     // out_rows = P = 128
                n_head,
                head_bytes_v,
                mc->attn_result_HxC))    // input (attn result per head)
            return 0;
        mla_ckpt_sum(mc->v_decomp_HxP, (uint64_t)n_head * qk_nope * sizeof(float), "v_decomp", layer);
    }

    // --- Step 12: Concat heads → out_proj_in ---
    // v_decomp layout: [n_head * qk_nope] = [32 * 128] = [4096] = n_embd
    // Already contiguous in the right order! Just copy to out_proj_in.
    // Actually, we can use v_decomp directly for the output projection.
    // But we need to make sure the layout matches: the output projection expects
    // [n_embd] input. v_decomp is [n_head * qk_nope] = [32 * 128] = [4096].
    // This is already the right layout (head outputs concatenated).

    // --- Step 13: Output projection ---
    if (!l26f_metal_matvec_quant(mc->attn_proj_1xN, mc->v_decomp_HxP,
            m->map, m->size, t_output_NxN->abs_offset,
            t_output_NxN->dim[0], t_output_NxN->dim[1], t_output_NxN->type, 1))
        return 0;
    mla_ckpt_sum(mc->attn_proj_1xN, n_embd * sizeof(float), "attn_proj", layer);

    // --- Step 14: Residual add ---
    if (!ds4_metal_add_tensor(out_1xN, hidden_1xN, mc->attn_proj_1xN, n_embd))
        return 0;

    return 1;
}
