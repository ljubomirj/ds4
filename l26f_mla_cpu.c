// l26f: CPU-side MLA (Multi-head Latent Attention) decode
// Phase A: CPU-only implementation for correctness
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "l26f.h"

// ---- Quantization constants ----

#define QK_K  256
#define QK4_NL 32

// ---- Quant block structs (must match ggml-common.h field order) ----

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qh[32];
    uint8_t  qs[128];
} cpu_block_q5_K;

typedef struct {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t  scales[16];
    uint16_t d;
} cpu_block_q6_K;

typedef struct {
    uint16_t d;
    uint8_t  qs[QK4_NL / 2];
} cpu_block_iq4_nl;

_Static_assert(sizeof(cpu_block_q5_K)  == 176, "Q5_K block must be 176 bytes");
_Static_assert(sizeof(cpu_block_q6_K)  == 210, "Q6_K block must be 210 bytes");
_Static_assert(sizeof(cpu_block_iq4_nl) == 18,  "IQ4_NL block must be 18 bytes");

static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

static inline float half_to_float(uint16_t h) {
    return (float)*(const __fp16 *)&h;
}

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

// ---- CPU dequantization: one row of a quantized weight matrix ----
// base: pointer to start of the row's quantized data
// out:  float output array, must have at least 'n_elements' entries

static void cpu_dequant_q5_K_row(const void *base, float *out, int n_elements) {
    const int nb = n_elements / QK_K;
    const cpu_block_q5_K *blocks = (const cpu_block_q5_K *)base;
    for (int i = 0; i < nb; i++) {
        const uint8_t *ql = blocks[i].qs;
        const uint8_t *qh = blocks[i].qh;
        const float d   = half_to_float(blocks[i].d);
        const float min = half_to_float(blocks[i].dmin);
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, blocks[i].scales, &sc, &m);
            const float d1 = d * sc, m1 = min * m;
            get_scale_min_k4(is + 1, blocks[i].scales, &sc, &m);
            const float d2 = d * sc, m2 = min * m;
            for (int l = 0; l < 32; ++l) *out++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *out++ = d2 * ((ql[l]  >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32; is += 2;
            u1 <<= 2; u2 <<= 2;
        }
    }
}

static void cpu_dequant_q6_K_row(const void *base, float *out, int n_elements) {
    const int nb = n_elements / QK_K;
    const cpu_block_q6_K *blocks = (const cpu_block_q6_K *)base;
    for (int i = 0; i < nb; i++) {
        const float d = half_to_float(blocks[i].d);
        const uint8_t *ql = blocks[i].ql;
        const uint8_t *qh = blocks[i].qh;
        const int8_t  *sc = blocks[i].scales;
        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                out[l +  0] = d * sc[is + 0] * q1;
                out[l + 32] = d * sc[is + 2] * q2;
                out[l + 64] = d * sc[is + 4] * q3;
                out[l + 96] = d * sc[is + 6] * q4;
            }
            out += 128; ql += 64; qh += 32; sc += 8;
        }
    }
}

static void cpu_dequant_iq4_nl_row(const void *base, float *out, int n_elements) {
    const int nb = n_elements / QK4_NL;
    const cpu_block_iq4_nl *blocks = (const cpu_block_iq4_nl *)base;
    for (int i = 0; i < nb; i++) {
        const float d = half_to_float(blocks[i].d);
        const uint8_t *qs = blocks[i].qs;
        for (int j = 0; j < QK4_NL / 2; ++j) {
            out[j + 0]        = d * kvalues_iq4nl[qs[j] & 0xf];
            out[j + QK4_NL/2] = d * kvalues_iq4nl[qs[j] >> 4];
        }
        out += QK4_NL;
    }
}

// Generic: dequant one row from quantized weight data
static void cpu_dequant_row(const void *base, float *out, uint32_t type, int n_elements) {
    switch (type) {
        case 0:  memcpy(out, base, n_elements * sizeof(float)); break;
        case 1:  for (int i=0;i<n_elements;i++) out[i] = half_to_float(((const uint16_t*)base)[i]); break;
        case 13: cpu_dequant_q5_K_row(base, out, n_elements); break;
        case 14: cpu_dequant_q6_K_row(base, out, n_elements); break;
        case 20: cpu_dequant_iq4_nl_row(base, out, n_elements); break;
        default: fprintf(stderr, "cpu_dequant_row: unsupported type %u\n", type); memset(out, 0, n_elements*sizeof(float)); break;
    }
}

// Row stride in bytes for a given type and number of columns
static uint64_t cpu_row_bytes(uint32_t type, int n_cols) {
    switch (type) {
        case 0:  return n_cols * sizeof(float);
        case 1:  return n_cols * sizeof(uint16_t);
        case 13: return (uint64_t)(n_cols / QK_K) * sizeof(cpu_block_q5_K);
        case 14: return (uint64_t)(n_cols / QK_K) * sizeof(cpu_block_q6_K);
        case 20: return (uint64_t)(n_cols / QK4_NL) * sizeof(cpu_block_iq4_nl);
        default: return 0;
    }
}

// ---- CPU matvec: dequantize + multiply ----

static void cpu_matvec(const void *weights, const float *input, float *output,
                        uint32_t type, int n_rows, int n_cols) {
    const uint64_t rbytes = cpu_row_bytes(type, n_cols);
    float *row = (float *)malloc(n_cols * sizeof(float));
    for (int r = 0; r < n_rows; r++) {
        cpu_dequant_row((const uint8_t *)weights + r * rbytes, row, type, n_cols);
        float sum = 0;
        for (int c = 0; c < n_cols; c++) sum += row[c] * input[c];
        output[r] = sum;
    }
    free(row);
}

// ---- CPU RMS norm ----

static void cpu_rms_norm(const float *x, const float *w, float *out, int n, float eps) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * ss * w[i];
}

// ---- CPU RoPE (simple rotary position embedding) ----
// Apply RoPE to a [n_dims] vector at a given position.
// Uses the standard theta = 10000.0, no dim correction.

static void cpu_rope(float *x, int n_dims, int position, float theta) {
    for (int i = 0; i < n_dims; i += 2) {
        float freq = 1.0f / powf(theta, (float)(i) / (float)n_dims);
        float angle = position * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);
        float x0 = x[i], x1 = x[i + 1];
        x[i]     = x0 * cos_a - x1 * sin_a;
        x[i + 1] = x0 * sin_a + x1 * cos_a;
    }
}

// ---- CPU softmax ----

static void cpu_softmax(float *x, int n) {
    float maxv = x[0];
    for (int i = 1; i < n; i++) if (x[i] > maxv) maxv = x[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - maxv); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

// ---- MLA KV cache ----

typedef struct {
    float *data_TxCR;
    int    max_seq_len;
    int    kv_dim;
    int    n_tokens;
} l26f_mla_kv_cache;

l26f_mla_kv_cache *l26f_mla_kv_cache_alloc(int max_seq, int kv_dim) {
    l26f_mla_kv_cache *c = (l26f_mla_kv_cache *)calloc(1, sizeof(*c));
    c->data_TxCR = (float *)calloc((uint64_t)max_seq * kv_dim, sizeof(float));
    c->max_seq_len = max_seq;
    c->kv_dim = kv_dim;
    c->n_tokens = 0;
    return c;
}

void l26f_mla_kv_cache_free(l26f_mla_kv_cache *c) {
    if (!c) return;
    free(c->data_TxCR);
    free(c);
}

// ---- MLA decode: single token ----
// Returns 1 on success, 0 on failure.
// hidden_in/hidden_out are GPU tensors; MLA runs on CPU, reads/writes via read/write.

int l26f_mla_layer_cpu(
        l26f_model *m, uint32_t layer, int position,
        l26f_mla_kv_cache *kv_cache,
        const float *hidden_cpu,    // [n_embd] input (already read from GPU)
        float *hidden_out_cpu)      // [n_embd] output (will be written to GPU)
{
    const uint32_t n_embd       = m->n_embd;
    const uint32_t n_head       = m->n_head;
    const uint32_t q_lora_rank  = m->q_lora_rank;
    const uint32_t kv_lora_rank = m->kv_lora_rank;
    const uint32_t n_rot        = m->n_rot;
    const uint32_t head_dim     = 192;  // 6144 / 32 = 192
    const uint32_t qk_nope      = head_dim - n_rot;  // 128
    const uint32_t kv_dim       = kv_lora_rank + n_rot;  // 576
    const float eps             = m->rms_norm_eps;

    // ---- Find tensors ----
    char name[128];
    #define FIND(suffix) do { \
        snprintf(name, sizeof(name), "blk.%u." suffix, layer); \
        if (!t_##suffix) t_##suffix = l26f_model_find_tensor(m, name); \
    } while(0)

    l26f_tensor *t_attn_norm_N   = NULL, *t_q_a_NxQ = NULL, *t_q_a_norm_Q = NULL;
    l26f_tensor *t_q_b_QxHxD    = NULL, *t_kv_a_mqa_NxCR = NULL, *t_kv_a_norm_C = NULL;
    l26f_tensor *t_k_b_PxCxH    = NULL, *t_v_b_CxPxH = NULL, *t_output_NxN = NULL;

    snprintf(name, sizeof(name), "blk.%u.attn_norm.weight", layer);
    t_attn_norm_N = l26f_model_find_tensor(m, name);
    snprintf(name, sizeof(name), "blk.%u.attn_q_a.weight", layer);
    t_q_a_NxQ = l26f_model_find_tensor(m, name);
    snprintf(name, sizeof(name), "blk.%u.attn_q_a_norm.weight", layer);
    t_q_a_norm_Q = l26f_model_find_tensor(m, name);
    snprintf(name, sizeof(name), "blk.%u.attn_q_b.weight", layer);
    t_q_b_QxHxD = l26f_model_find_tensor(m, name);
    snprintf(name, sizeof(name), "blk.%u.attn_kv_a_mqa.weight", layer);
    t_kv_a_mqa_NxCR = l26f_model_find_tensor(m, name);
    snprintf(name, sizeof(name), "blk.%u.attn_kv_a_norm.weight", layer);
    t_kv_a_norm_C = l26f_model_find_tensor(m, name);
    snprintf(name, sizeof(name), "blk.%u.attn_k_b.weight", layer);
    t_k_b_PxCxH = l26f_model_find_tensor(m, name);
    snprintf(name, sizeof(name), "blk.%u.attn_v_b.weight", layer);
    t_v_b_CxPxH = l26f_model_find_tensor(m, name);
    snprintf(name, sizeof(name), "blk.%u.attn_output.weight", layer);
    t_output_NxN = l26f_model_find_tensor(m, name);

    if (!t_attn_norm_N || !t_q_a_NxQ || !t_q_a_norm_Q || !t_q_b_QxHxD || !t_kv_a_mqa_NxCR ||
        !t_kv_a_norm_C || !t_k_b_PxCxH || !t_v_b_CxPxH || !t_output_NxN) {
        fprintf(stderr, "l26f: MLA layer %u missing tensors\n", layer);
        return 0;
    }

    const uint8_t *model_base = m->map;
    float *normed_1xN      = (float *)malloc(n_embd * sizeof(float));
    float *q_a_1xQ         = (float *)malloc(q_lora_rank * sizeof(float));
    float *q_a_normed_1xQ  = (float *)malloc(q_lora_rank * sizeof(float));
    float *q_b_1xHxD       = (float *)malloc((uint64_t)n_head * head_dim * sizeof(float));
    float *kv_a_1xCR      = (float *)malloc(kv_dim * sizeof(float));
    float *kv_cmpr_1xC     = (float *)malloc(kv_lora_rank * sizeof(float));
    float *q_absorbed_HxC  = (float *)malloc((uint64_t)n_head * kv_lora_rank * sizeof(float));
    float *v_decomp_HxP    = (float *)malloc((uint64_t)n_head * 128 * sizeof(float));
    float *attn_out_1xC    = (float *)malloc(kv_lora_rank * sizeof(float));
    float *scores_1xT      = (float *)malloc((kv_cache->n_tokens + 1) * sizeof(float));
    float *attn_result_HxC = (float *)malloc((uint64_t)n_head * kv_lora_rank * sizeof(float));
    float *out_proj_in_1xN = (float *)malloc(n_embd * sizeof(float));
    float *attn_proj_1xN   = (float *)malloc(n_embd * sizeof(float));

    const float *w_norm_1xN = (const float *)(model_base + t_attn_norm_N->abs_offset);
    cpu_rms_norm(hidden_cpu, w_norm_1xN, normed_1xN, n_embd, eps);

    cpu_matvec(model_base + t_q_a_NxQ->abs_offset, normed_1xN, q_a_1xQ,
               t_q_a_NxQ->type, q_lora_rank, n_embd);

    const float *w_q_a_norm_1xQ = (const float *)(model_base + t_q_a_norm_Q->abs_offset);
    cpu_rms_norm(q_a_1xQ, w_q_a_norm_1xQ, q_a_normed_1xQ, q_lora_rank, eps);

    cpu_matvec(model_base + t_q_b_QxHxD->abs_offset, q_a_normed_1xQ, q_b_1xHxD,
               t_q_b_QxHxD->type, n_head * head_dim, q_lora_rank);

    // Split Q into q_nope [n_head, qk_nope] and q_pe [n_head, n_rot]
    // q_b layout: [n_head * head_dim] = for each head, first qk_nope=128 then n_rot=64
    // So q_nope[h] = q_b[h * head_dim ... h * head_dim + qk_nope - 1]
    //    q_pe[h]   = q_b[h * head_dim + qk_nope ... (h+1) * head_dim - 1]

    // Apply RoPE to q_pe
    for (uint32_t h = 0; h < n_head; h++) {
        float *q_pe_h_R = q_b_1xHxD + h * head_dim + qk_nope;
        cpu_rope(q_pe_h_R, n_rot, position, m->rope_theta);
    }

    cpu_matvec(model_base + t_kv_a_mqa_NxCR->abs_offset, normed_1xN, kv_a_1xCR,
               t_kv_a_mqa_NxCR->type, kv_dim, n_embd);

    memcpy(kv_cmpr_1xC, kv_a_1xCR, kv_lora_rank * sizeof(float));
    float *k_pe_R = kv_a_1xCR + kv_lora_rank;

    const float *w_kv_a_norm_1xC = (const float *)(model_base + t_kv_a_norm_C->abs_offset);
    cpu_rms_norm(kv_cmpr_1xC, w_kv_a_norm_1xC, kv_cmpr_1xC, kv_lora_rank, eps);

    cpu_rope(k_pe_R, n_rot, position, m->rope_theta);

    // 4. Absorption: q_absorbed[h] = wk_b[:,:,h] × q_nope[:,h]
    // wk_b: [qk_nope, kv_lora_rank, n_head] (dim[0]=128, dim[1]=512, dim[2]=32)
    // q_nope[:,h]: [qk_nope] = [128]
    // result: [kv_lora_rank]
    {
        const uint64_t head_bytes = (uint64_t)(t_k_b_PxCxH->dim[1]) * cpu_row_bytes(t_k_b_PxCxH->type, t_k_b_PxCxH->dim[0]);
        float *row_buf = (float *)malloc(t_k_b_PxCxH->dim[0] * sizeof(float));
        for (uint32_t h = 0; h < n_head; h++) {
            const uint8_t *wk_b_h_PxC = model_base + t_k_b_PxCxH->abs_offset + h * head_bytes;
            const float *q_nope_h_P = q_b_1xHxD + h * head_dim;
            cpu_matvec(wk_b_h_PxC, q_nope_h_P, q_absorbed_HxC + h * kv_lora_rank,
                       t_k_b_PxCxH->type, t_k_b_PxCxH->dim[1], t_k_b_PxCxH->dim[0]);
        }
        free(row_buf);
    }

    {
        float *k_new_1xCR = kv_cache->data_TxCR + (uint64_t)kv_cache->n_tokens * kv_dim;
        memcpy(k_new_1xCR, kv_cmpr_1xC, kv_lora_rank * sizeof(float));
        memcpy(k_new_1xCR + kv_lora_rank, k_pe_R, n_rot * sizeof(float));
        kv_cache->n_tokens++;
    }

    {
        const int n_cached = kv_cache->n_tokens;
        const float q_scale = 1.0f / sqrtf((float)kv_dim);

        for (uint32_t h = 0; h < n_head; h++) {
            for (int t = 0; t < n_cached; t++) {
                const float *k_t_1xCR = kv_cache->data_TxCR + (uint64_t)t * kv_dim;
                float dot = 0;
                for (uint32_t d = 0; d < kv_lora_rank; d++) {
                    dot += q_absorbed_HxC[h * kv_lora_rank + d] * k_t_1xCR[d];
                }
                for (uint32_t d = 0; d < n_rot; d++) {
                    float q_val = q_b_1xHxD[h * head_dim + qk_nope + d];
                    float k_val = k_t_1xCR[kv_lora_rank + d];
                    dot += q_val * k_val;
                }
                scores_1xT[t] = dot * q_scale;
            }

            cpu_softmax(scores_1xT, n_cached);

            float *attn_h_1xC = attn_result_HxC + h * kv_lora_rank;
            memset(attn_h_1xC, 0, kv_lora_rank * sizeof(float));
            for (int t = 0; t < n_cached; t++) {
                const float *v_t_1xCR = kv_cache->data_TxCR + (uint64_t)t * kv_dim;
                float w = scores_1xT[t];
                for (uint32_t d = 0; d < kv_lora_rank; d++) {
                    attn_h_1xC[d] += w * v_t_1xCR[d];
                }
            }
        }
    }

    {
        const uint64_t head_bytes = (uint64_t)(t_v_b_CxPxH->dim[1]) * cpu_row_bytes(t_v_b_CxPxH->type, t_v_b_CxPxH->dim[0]);
        for (uint32_t h = 0; h < n_head; h++) {
            const uint8_t *wv_b_h_CxP = model_base + t_v_b_CxPxH->abs_offset + h * head_bytes;
            const float *attn_h_1xC = attn_result_HxC + h * kv_lora_rank;
            cpu_matvec(wv_b_h_CxP, attn_h_1xC, v_decomp_HxP + h * 128,
                       t_v_b_CxPxH->type, t_v_b_CxPxH->dim[1], t_v_b_CxPxH->dim[0]);
        }
    }

    for (uint32_t h = 0; h < n_head; h++) {
        memcpy(out_proj_in_1xN + h * 128, v_decomp_HxP + h * 128, 128 * sizeof(float));
    }
    cpu_matvec(model_base + t_output_NxN->abs_offset, out_proj_in_1xN, attn_proj_1xN,
               t_output_NxN->type, n_embd, n_embd);

    for (uint32_t i = 0; i < n_embd; i++) {
        hidden_out_cpu[i] = hidden_cpu[i] + attn_proj_1xN[i];
    }

    free(normed_1xN); free(q_a_1xQ); free(q_a_normed_1xQ); free(q_b_1xHxD);
    free(kv_a_1xCR); free(kv_cmpr_1xC); free(q_absorbed_HxC); free(v_decomp_HxP);
    free(attn_out_1xC); free(scores_1xT); free(attn_result_HxC);
    free(out_proj_in_1xN); free(attn_proj_1xN);

    return 1;
}
