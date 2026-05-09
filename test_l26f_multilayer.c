// l26f: inference driver — single-token decode through GLA layers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include "l26f.h"
#include "l26f_metal.h"
#include "ds4_metal.h"
#include "l26f_tokenizer.h"

// Forward declaration for MLA CPU implementation
typedef struct l26f_mla_kv_cache l26f_mla_kv_cache;
extern l26f_mla_kv_cache *l26f_mla_kv_cache_alloc(int max_seq, int kv_dim);
extern void l26f_mla_kv_cache_free(l26f_mla_kv_cache *c);
extern int l26f_mla_layer_cpu(l26f_model *m, uint32_t layer, int position,
    l26f_mla_kv_cache *kv_cache, const float *hidden_cpu, float *hidden_out_cpu);

// ---- Per-layer compute buffer set ----
// Reused across layers to avoid excessive allocation.
// We need: normed, qkv, gate, gla_out+state, proj, ffn_mid, ffn_down
// All are n_embd-sized except qkv (3*n_embd) and gla_out (n_embd + S*S*H).

typedef struct {
    ds4_metal_tensor *normed;
    ds4_metal_tensor *qkv;
    ds4_metal_tensor *gate_out;
    ds4_metal_tensor *gla_out;
    ds4_metal_tensor *attn_proj;
    ds4_metal_tensor *post_attn;
    ds4_metal_tensor *ffn_normed;
    ds4_metal_tensor *ffn_gate;
    ds4_metal_tensor *ffn_up;
    ds4_metal_tensor *ffn_mid;
    ds4_metal_tensor *ffn_down;
    ds4_metal_tensor *moe_out;      // accumulated MoE output
    ds4_metal_tensor *shexp_out;    // shared expert output
} l26f_compute;

// GLA state: one per GLA layer, persists across tokens
typedef struct {
    ds4_metal_tensor *state;
} l26f_gla_state;

typedef struct {
    l26f_model *model;
    l26f_compute comp;
    l26f_gla_state gla_states[32];
    l26f_mla_kv_cache *mla_kv[32];  // MLA KV caches (only layers 7,15,23,31)
    ds4_metal_tensor *hidden;
    ds4_metal_tensor *output_normed;
    ds4_metal_tensor *logits;
} l26f_session;

// ---- helpers ----

static float l26f_tensor_checksum(const ds4_metal_tensor *t, uint64_t bytes, int *out_nans) {
    float *data = (float *)malloc(bytes);
    if (!data) { *out_nans = -1; return 0.0f; }
    ds4_metal_tensor_read(t, 0, data, bytes);
    float sum = 0;
    int nans = 0;
    uint64_t n = bytes / sizeof(float);
    for (uint64_t i = 0; i < n; i++) {
        if (isnan(data[i])) { nans++; continue; }
        sum += data[i];
    }
    free(data);
    *out_nans = nans;
    return sum;
}

static void l26f_checksum_print(const char *label, const ds4_metal_tensor *t, uint64_t bytes) {
    int nans;
    float sum = l26f_tensor_checksum(t, bytes, &nans);
    fprintf(stderr, "    CHECKSUM %-20s  sum=%.6f  nans=%d\n", label, sum, nans);
}

static l26f_tensor *l26f_layer_tensor(const l26f_model *m, uint32_t layer, const char *suffix) {
    char name[128];
    snprintf(name, sizeof(name), "blk.%u.%s", layer, suffix);
    return l26f_model_find_tensor(m, name);
}

static int l26f_gla_layer(
        l26f_session *s, uint32_t layer,
        ds4_metal_tensor *inp, ds4_metal_tensor *out) {
    l26f_model *m = s->model;
    l26f_compute *c = &s->comp;
    const uint32_t n_embd = m->n_embd;
    const uint32_t S = 128, H = m->n_head;
    const uint64_t act_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t qkv_bytes = 3ULL * n_embd * sizeof(float);
    const uint64_t gla_state_bytes = (uint64_t)S * S * H * sizeof(float);
    const uint64_t gla_out_bytes = act_bytes + gla_state_bytes;
    float scale = 1.0f / sqrtf((float)S);

    l26f_tensor *wt_norm = l26f_layer_tensor(m, layer, "attn_norm.weight");
    l26f_tensor *wt_qkv  = l26f_layer_tensor(m, layer, "attn_qkv.weight");
    l26f_tensor *wt_gate = l26f_layer_tensor(m, layer, "attn_gate.weight");
    l26f_tensor *wt_out  = l26f_layer_tensor(m, layer, "attn_output.weight");
    if (!wt_norm || !wt_qkv || !wt_gate || !wt_out) {
        fprintf(stderr, "l26f: layer %u missing GLA tensors\n", layer);
        return 0;
    }

    // 1. RMS norm
    if (!ds4_metal_rms_norm_weight_tensor(c->normed, inp,
            m->map, m->size, wt_norm->abs_offset, n_embd, m->rms_norm_eps))
        return 0;

    // 2. QKV matvec
    if (!l26f_metal_matvec_quant(c->qkv, c->normed,
            m->map, m->size, wt_qkv->abs_offset,
            wt_qkv->dim[0], wt_qkv->dim[1], wt_qkv->type, 1))
        return 0;

    // 3. Gate matvec
    if (!l26f_metal_matvec_quant(c->gate_out, c->normed,
            m->map, m->size, wt_gate->abs_offset,
            wt_gate->dim[0], wt_gate->dim[1], wt_gate->type, 1))
        return 0;

    // 4. GLA attention
    ds4_metal_tensor *q_view = ds4_metal_tensor_view(c->qkv, 0, act_bytes);
    ds4_metal_tensor *k_view = ds4_metal_tensor_view(c->qkv, act_bytes, act_bytes);
    ds4_metal_tensor *v_view = ds4_metal_tensor_view(c->qkv, 2*act_bytes, act_bytes);

    int ok = l26f_metal_gla(c->gla_out, s->gla_states[layer].state,
                q_view, k_view, v_view, c->gate_out,
                1, 1, S, H, scale);

    ds4_metal_tensor_free(v_view);
    ds4_metal_tensor_free(k_view);
    ds4_metal_tensor_free(q_view);
    if (!ok) return 0;

    // 5. Output projection (first n_embd floats of gla_out are the attention output)
    ds4_metal_tensor *gla_act = ds4_metal_tensor_view(c->gla_out, 0, act_bytes);
    ok = l26f_metal_matvec_quant(c->attn_proj, gla_act,
            m->map, m->size, wt_out->abs_offset,
            wt_out->dim[0], wt_out->dim[1], wt_out->type, 1);
    ds4_metal_tensor_free(gla_act);
    if (!ok) return 0;

    // 6. Residual add: out = inp + attn_proj
    if (!ds4_metal_add_tensor(out, inp, c->attn_proj, n_embd))
        return 0;

    return 1;
}

static int l26f_dense_ffn(
        l26f_session *s,
        ds4_metal_tensor *inp, ds4_metal_tensor *out) {
    l26f_model *m = s->model;
    l26f_compute *c = &s->comp;
    const uint32_t n_embd = m->n_embd;
    const uint32_t n_ff = m->n_ff;

    l26f_tensor *wt_norm = l26f_layer_tensor(m, 0, "ffn_norm.weight");
    l26f_tensor *wt_gate = l26f_layer_tensor(m, 0, "ffn_gate.weight");
    l26f_tensor *wt_up   = l26f_layer_tensor(m, 0, "ffn_up.weight");
    l26f_tensor *wt_down = l26f_layer_tensor(m, 0, "ffn_down.weight");
    if (!wt_norm || !wt_gate || !wt_up || !wt_down) {
        fprintf(stderr, "l26f: layer 0 missing dense FFN tensors\n");
        return 0;
    }

    // 1. RMS norm
    if (!ds4_metal_rms_norm_weight_tensor(c->ffn_normed, inp,
            m->map, m->size, wt_norm->abs_offset, n_embd, m->rms_norm_eps))
        return 0;

    // 2. Gate projection
    if (!l26f_metal_matvec_quant(c->ffn_gate, c->ffn_normed,
            m->map, m->size, wt_gate->abs_offset,
            wt_gate->dim[0], wt_gate->dim[1], wt_gate->type, 1))
        return 0;

    // 3. Up projection
    if (!l26f_metal_matvec_quant(c->ffn_up, c->ffn_normed,
            m->map, m->size, wt_up->abs_offset,
            wt_up->dim[0], wt_up->dim[1], wt_up->type, 1))
        return 0;

    // 4. SwiGLU: silu(gate) * up
    if (!ds4_metal_swiglu_tensor(c->ffn_mid, c->ffn_gate, c->ffn_up, n_ff, 0.0f, 1.0f))
        return 0;

    // 5. Down projection
    if (!l26f_metal_matvec_quant(c->ffn_down, c->ffn_mid,
            m->map, m->size, wt_down->abs_offset,
            wt_down->dim[0], wt_down->dim[1], wt_down->type, 1))
        return 0;

    // 6. Residual add: out = inp + ffn_down
    if (!ds4_metal_add_tensor(out, inp, c->ffn_down, n_embd))
        return 0;

    return 1;
}

// ---- MoE FFN ----
//
// Routing algorithm (from llama.cpp build_moe_ffn):
// 1. logits = gate_inp × hidden  (F32 matvec, [4096]×[4096,256]→[256])
// 2. Add exp_probs_b bias
// 3. probs = softmax(logits)
// 4. Group scoring: 8 groups × 32 experts, score = sum(top-2 probs per group)
// 5. Select top-4 groups
// 6. Mask: only keep experts from top-4 groups
// 7. top-8 from masked pool
// 8. weights = probs[selected], normalize, scale by 2.5
// 9. Run 8 experts + shared expert, weighted sum

static void cpu_softmax(float *x, int n) {
    float maxv = x[0];
    for (int i = 1; i < n; i++) if (x[i] > maxv) maxv = x[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - maxv); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

static int cmp_float_desc(const void *a, const void *b) {
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    return (fa < fb) - (fa > fb);
}

static int l26f_moe_ffn(
        l26f_session *s, uint32_t layer,
        ds4_metal_tensor *inp, ds4_metal_tensor *out) {
    l26f_model *m = s->model;
    l26f_compute *c = &s->comp;
    const uint32_t n_embd = m->n_embd;
    const uint32_t n_ff_exp = 1024;  // expert hidden dim
    const uint32_t n_expert = 256;
    const uint32_t n_expert_used = 8;
    const uint32_t n_groups = 8;
    const uint32_t n_group_used = 4;
    const uint32_t n_exp_per_group = n_expert / n_groups;  // 32
    const float w_scale = 2.5f;

    // Find tensors
    l26f_tensor *wt_norm      = l26f_layer_tensor(m, layer, "ffn_norm.weight");
    l26f_tensor *wt_gate_inp  = l26f_layer_tensor(m, layer, "ffn_gate_inp.weight");
    l26f_tensor *wt_exp_b     = l26f_layer_tensor(m, layer, "exp_probs_b.bias");
    l26f_tensor *wt_gate_exps = l26f_layer_tensor(m, layer, "ffn_gate_exps.weight");
    l26f_tensor *wt_up_exps   = l26f_layer_tensor(m, layer, "ffn_up_exps.weight");
    l26f_tensor *wt_down_exps = l26f_layer_tensor(m, layer, "ffn_down_exps.weight");
    l26f_tensor *wt_gate_sh   = l26f_layer_tensor(m, layer, "ffn_gate_shexp.weight");
    l26f_tensor *wt_up_sh     = l26f_layer_tensor(m, layer, "ffn_up_shexp.weight");
    l26f_tensor *wt_down_sh   = l26f_layer_tensor(m, layer, "ffn_down_shexp.weight");
    if (!wt_norm || !wt_gate_inp || !wt_gate_exps || !wt_up_exps || !wt_down_exps ||
        !wt_gate_sh || !wt_up_sh || !wt_down_sh) {
        fprintf(stderr, "l26f: layer %u missing MoE tensors\n", layer);
        return 0;
    }

    // 1. RMS norm
    if (!ds4_metal_rms_norm_weight_tensor(c->ffn_normed, inp,
            m->map, m->size, wt_norm->abs_offset, n_embd, m->rms_norm_eps))
        return 0;

    // 2. CPU-side router: read hidden, compute logits
    float *hidden_cpu = (float *)malloc(n_embd * sizeof(float));
    ds4_metal_tensor_read(c->ffn_normed, 0, hidden_cpu, n_embd * sizeof(float));

    float logits[256];
    float *gate_inp_data = (float *)(m->map + wt_gate_inp->abs_offset);
    // ffn_gate_inp shape: [4096, 256] — row-major, so element [i, e] is at i*256 + e
    for (uint32_t e = 0; e < n_expert; e++) {
        float sum = 0;
        for (uint32_t i = 0; i < n_embd; i++) {
            sum += hidden_cpu[i] * gate_inp_data[i * n_expert + e];
        }
        logits[e] = sum;
    }

    // Add exp_probs_b bias if present
    if (wt_exp_b) {
        float *bias = (float *)(m->map + wt_exp_b->abs_offset);
        for (uint32_t e = 0; e < n_expert; e++) logits[e] += bias[e];
    }

    // 3. Softmax
    float probs[256];
    memcpy(probs, logits, sizeof(logits));
    cpu_softmax(probs, n_expert);

    // 4. Group scoring: sum of top-2 per group
    float group_scores[8];
    for (uint32_t g = 0; g < n_groups; g++) {
        float top2[2] = {0, 0};
        for (uint32_t i = 0; i < n_exp_per_group; i++) {
            float p = probs[g * n_exp_per_group + i];
            if (p > top2[0]) { top2[1] = top2[0]; top2[0] = p; }
            else if (p > top2[1]) { top2[1] = p; }
        }
        group_scores[g] = top2[0] + top2[1];
    }

    // 5. Select top-4 groups
    int selected_groups[8];
    for (int i = 0; i < 8; i++) selected_groups[i] = i;
    // Simple selection sort for top-4
    for (int i = 0; i < 4; i++) {
        int best = i;
        for (int j = i+1; j < 8; j++) {
            if (group_scores[selected_groups[j]] > group_scores[selected_groups[best]])
                best = j;
        }
        int tmp = selected_groups[i]; selected_groups[i] = selected_groups[best]; selected_groups[best] = tmp;
    }

    // 6. Mask: build masked probs (only selected groups)
    float masked_probs[256];
    memcpy(masked_probs, probs, sizeof(probs));
    for (uint32_t g = 0; g < n_groups; g++) {
        bool selected = false;
        for (int i = 0; i < 4; i++) if (selected_groups[i] == (int)g) selected = true;
        if (!selected) {
            for (uint32_t i = 0; i < n_exp_per_group; i++)
                masked_probs[g * n_exp_per_group + i] = -INFINITY;
        }
    }

    // 7. Top-8 from masked pool
    int selected_experts[8];
    float selected_weights[8];
    for (int i = 0; i < 8; i++) {
        int best_e = 0;
        float best_p = -INFINITY;
        for (uint32_t e = 0; e < n_expert; e++) {
            if (masked_probs[e] > best_p) { best_p = masked_probs[e]; best_e = (int)e; }
        }
        selected_experts[i] = best_e;
        selected_weights[i] = probs[best_e];  // use original probs for weights
        masked_probs[best_e] = -INFINITY;  // remove from pool
    }

    // 8. Normalize and scale weights
    float wsum = 0;
    for (int i = 0; i < 8; i++) wsum += selected_weights[i];
    if (wsum > 1e-6f) {
        for (int i = 0; i < 8; i++) selected_weights[i] /= wsum;
    }
    for (int i = 0; i < 8; i++) selected_weights[i] *= w_scale;

    free(hidden_cpu);

    if (layer == 4) {
        printf("    MoE layer %u: experts=[", layer);
        for (int i = 0; i < 8; i++) printf("%d ", selected_experts[i]);
        printf("] weights=[");
        for (int i = 0; i < 8; i++) printf("%.4f ", selected_weights[i]);
        printf("]\n");
    }

    // 9. Run selected experts on GPU
    // Zero the accumulator
    float zero = 0.0f;
    ds4_metal_tensor_write(c->moe_out, 0, &zero, sizeof(float));
    // Actually need to zero the whole buffer. Use a temp zero buffer.
    {
        void *z = calloc(1, n_embd * sizeof(float));
        ds4_metal_tensor_write(c->moe_out, 0, z, n_embd * sizeof(float));
        free(z);
    }

    if (layer == 1) {
        printf("    down_exps: ndim=%u dims=[%lu,%lu,%lu] type=%u abs_off=%lu\n",
               wt_down_exps->ndim,
               (unsigned long)wt_down_exps->dim[0], (unsigned long)wt_down_exps->dim[1],
               (unsigned long)wt_down_exps->dim[2],
               wt_down_exps->type, (unsigned long)wt_down_exps->abs_offset);
        printf("    gate_exps: ndim=%u dims=[%lu,%lu,%lu] type=%u\n",
               wt_gate_exps->ndim,
               (unsigned long)wt_gate_exps->dim[0], (unsigned long)wt_gate_exps->dim[1],
               (unsigned long)wt_gate_exps->dim[2],
               wt_gate_exps->type);
        printf("    up_exps:   ndim=%u dims=[%lu,%lu,%lu] type=%u\n",
               wt_up_exps->ndim,
               (unsigned long)wt_up_exps->dim[0], (unsigned long)wt_up_exps->dim[1],
               (unsigned long)wt_up_exps->dim[2],
               wt_up_exps->type);
        printf("    gate_sh:   ndim=%u dims=[%lu,%lu,%lu] type=%u\n",
               wt_gate_sh->ndim,
               (unsigned long)wt_gate_sh->dim[0], (unsigned long)wt_gate_sh->dim[1],
               (unsigned long)wt_gate_sh->dim[2],
               wt_gate_sh->type);
        printf("    down_sh:   ndim=%u dims=[%lu,%lu,%lu] type=%u\n",
               wt_down_sh->ndim,
               (unsigned long)wt_down_sh->dim[0], (unsigned long)wt_down_sh->dim[1],
               (unsigned long)wt_down_sh->dim[2],
               wt_down_sh->type);
    }

    // Expert bytes per tensor — use actual type block_size/type_size
    const uint32_t gate_bs = l26f_types[wt_gate_exps->type].block_size;
    const uint32_t gate_ts = l26f_types[wt_gate_exps->type].type_size;
    const uint32_t up_bs   = l26f_types[wt_up_exps->type].block_size;
    const uint32_t up_ts   = l26f_types[wt_up_exps->type].type_size;
    const uint32_t down_bs = l26f_types[wt_down_exps->type].block_size;
    const uint32_t down_ts = l26f_types[wt_down_exps->type].type_size;
    // per-expert rows = dim[1], cols = dim[0], stride = dim[1] * (dim[0]/bs) * ts
    const uint64_t gate_exp_bytes = (uint64_t)wt_gate_exps->dim[1] * (wt_gate_exps->dim[0] / gate_bs) * gate_ts;
    const uint64_t up_exp_bytes   = (uint64_t)wt_up_exps->dim[1]   * (wt_up_exps->dim[0]   / up_bs)   * up_ts;
    const uint64_t down_exp_bytes = (uint64_t)wt_down_exps->dim[1] * (wt_down_exps->dim[0] / down_bs) * down_ts;

    for (int i = 0; i < 8; i++) {
        int e = selected_experts[i];
        uint64_t gate_off = wt_gate_exps->abs_offset + (uint64_t)e * gate_exp_bytes;
        uint64_t up_off   = wt_up_exps->abs_offset   + (uint64_t)e * up_exp_bytes;
        uint64_t down_off = wt_down_exps->abs_offset + (uint64_t)e * down_exp_bytes;

        // Expert gate matvec: [n_embd → n_ff_exp]
        if (!l26f_metal_matvec_quant(c->ffn_gate, c->ffn_normed,
                m->map, m->size, gate_off,
                n_embd, n_ff_exp, wt_gate_exps->type, 1))
            return 0;

        // Expert up matvec: [n_embd → n_ff_exp]
        if (!l26f_metal_matvec_quant(c->ffn_up, c->ffn_normed,
                m->map, m->size, up_off,
                n_embd, n_ff_exp, wt_up_exps->type, 1))
            return 0;

        // SwiGLU
        if (!ds4_metal_swiglu_tensor(c->ffn_mid, c->ffn_gate, c->ffn_up, n_ff_exp, 0.0f, 1.0f))
            return 0;

        // Expert down matvec: [n_ff_exp → n_embd]
        if (!l26f_metal_matvec_quant(c->ffn_down, c->ffn_mid,
                m->map, m->size, down_off,
                n_ff_exp, n_embd, wt_down_exps->type, 1))
            return 0;

        // Debug: check expert output for NaN
        if (layer == 1 || layer == 4) {
            float *exp_out = (float *)malloc(n_embd * sizeof(float));
            ds4_metal_tensor_read(c->ffn_down, 0, exp_out, n_embd * sizeof(float));
            float sum = 0; int nans = 0;
            for (uint32_t j = 0; j < n_embd; j++) {
                if (isnan(exp_out[j])) nans++;
                sum += exp_out[j];
            }
            if (nans > 0 || layer == 1) {
                printf("    EXPERT %d (e=%d) DOWN: nans=%d sum=%.3f\n", i, e, nans, sum);
            }
            free(exp_out);
        }
        if (layer == 1) {
            float *tmp = (float *)malloc(n_ff_exp * sizeof(float));
            ds4_metal_tensor_read(c->ffn_gate, 0, tmp, n_ff_exp * sizeof(float));
            float sum = 0; int nans = 0;
            for (uint32_t j = 0; j < n_ff_exp; j++) { if (isnan(tmp[j])) nans++; else sum += tmp[j]; }
            if (nans > 0) printf("    EXPERT %d gate: nans=%d sum=%.3f\n", i, nans, sum);
            ds4_metal_tensor_read(c->ffn_up, 0, tmp, n_ff_exp * sizeof(float));
            sum = 0; nans = 0;
            for (uint32_t j = 0; j < n_ff_exp; j++) { if (isnan(tmp[j])) nans++; else sum += tmp[j]; }
            if (nans > 0) printf("    EXPERT %d up: nans=%d sum=%.3f\n", i, nans, sum);
            ds4_metal_tensor_read(c->ffn_mid, 0, tmp, n_ff_exp * sizeof(float));
            sum = 0; nans = 0;
            for (uint32_t j = 0; j < n_ff_exp; j++) { if (isnan(tmp[j])) nans++; else sum += tmp[j]; }
            if (nans > 0) printf("    EXPERT %d mid: nans=%d sum=%.3f\n", i, nans, sum);
            free(tmp);
        }

        // Accumulate weighted output: moe_out += weight * ffn_down
        // ds4 doesn't have a weighted-add kernel. Do on CPU for now.
        float *exp_out = (float *)malloc(n_embd * sizeof(float));
        ds4_metal_tensor_read(c->ffn_down, 0, exp_out, n_embd * sizeof(float));
        float *acc = (float *)malloc(n_embd * sizeof(float));
        ds4_metal_tensor_read(c->moe_out, 0, acc, n_embd * sizeof(float));
        float w = selected_weights[i];
        for (uint32_t j = 0; j < n_embd; j++) acc[j] += w * exp_out[j];
        ds4_metal_tensor_write(c->moe_out, 0, acc, n_embd * sizeof(float));
        free(exp_out);
        free(acc);
    }

    // 10. Shared expert (always runs)
    if (!l26f_metal_matvec_quant(c->ffn_gate, c->ffn_normed,
            m->map, m->size, wt_gate_sh->abs_offset,
            wt_gate_sh->dim[0], wt_gate_sh->dim[1], wt_gate_sh->type, 1))
        return 0;
    if (!l26f_metal_matvec_quant(c->ffn_up, c->ffn_normed,
            m->map, m->size, wt_up_sh->abs_offset,
            wt_up_sh->dim[0], wt_up_sh->dim[1], wt_up_sh->type, 1))
        return 0;
    if (!ds4_metal_swiglu_tensor(c->ffn_mid, c->ffn_gate, c->ffn_up, n_ff_exp, 0.0f, 1.0f))
        return 0;
    if (!l26f_metal_matvec_quant(c->shexp_out, c->ffn_mid,
            m->map, m->size, wt_down_sh->abs_offset,
            wt_down_sh->dim[0], wt_down_sh->dim[1], wt_down_sh->type, 1))
        return 0;

    // Add shared expert to MoE output
    {
        float *moe = (float *)malloc(n_embd * sizeof(float));
        float *sh = (float *)malloc(n_embd * sizeof(float));
        ds4_metal_tensor_read(c->moe_out, 0, moe, n_embd * sizeof(float));
        ds4_metal_tensor_read(c->shexp_out, 0, sh, n_embd * sizeof(float));
        for (uint32_t j = 0; j < n_embd; j++) moe[j] += sh[j];
        ds4_metal_tensor_write(c->moe_out, 0, moe, n_embd * sizeof(float));
        free(moe);
        free(sh);
    }

    // Debug: print moe_out magnitude before residual
    if (layer == 1 || layer == 4) {
        float *moe = (float *)malloc(n_embd * sizeof(float));
        ds4_metal_tensor_read(c->moe_out, 0, moe, n_embd * sizeof(float));
        float sum = 0, mn = moe[0], mx = moe[0]; int nans = 0;
        for (uint32_t i = 0; i < n_embd; i++) {
            if (isnan(moe[i])) { nans++; continue; }
            sum += moe[i];
            if (moe[i] < mn) mn = moe[i];
            if (moe[i] > mx) mx = moe[i];
        }
        printf("    moe_out before residual: sum=%.3f min=%.3f max=%.3f nans=%d\n", sum, mn, mx, nans);
        free(moe);
    }
    // Debug shared expert output for layer 1
    if (layer == 1) {
        float *sh = (float *)malloc(n_embd * sizeof(float));
        ds4_metal_tensor_read(c->shexp_out, 0, sh, n_embd * sizeof(float));
        float sum = 0; int nans = 0;
        for (uint32_t i = 0; i < n_embd; i++) { if (isnan(sh[i])) nans++; else sum += sh[i]; }
        printf("    shared expert down: sum=%.3f nans=%d\n", sum, nans);
        free(sh);
    }

    // 11. Residual add: out = inp + moe_out
    if (!ds4_metal_add_tensor(out, inp, c->moe_out, n_embd))
        return 0;

    return 1;
}

static int l26f_session_init(l26f_session *s, l26f_model *m) {
    memset(s, 0, sizeof(*s));
    s->model = m;

    const uint32_t n_embd = m->n_embd;
    const uint32_t n_ff = m->n_ff;
    const uint32_t S = 128, H = m->n_head;
    const uint64_t act_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t qkv_bytes = 3ULL * n_embd * sizeof(float);
    const uint64_t ffn_bytes = (uint64_t)n_ff * sizeof(float);
    const uint64_t gla_state_bytes = (uint64_t)S * S * H * sizeof(float);
    const uint64_t gla_out_bytes = act_bytes + gla_state_bytes;

    // Allocate compute buffers (zeroed — Metal shared memory is uninitialized)
    s->comp.normed      = ds4_metal_tensor_alloc(act_bytes);
    s->comp.qkv         = ds4_metal_tensor_alloc(qkv_bytes);
    s->comp.gate_out    = ds4_metal_tensor_alloc(act_bytes);
    s->comp.gla_out     = ds4_metal_tensor_alloc(gla_out_bytes);
    s->comp.attn_proj   = ds4_metal_tensor_alloc(act_bytes);
    s->comp.post_attn   = ds4_metal_tensor_alloc(act_bytes);
    s->comp.ffn_normed  = ds4_metal_tensor_alloc(act_bytes);
    s->comp.ffn_gate    = ds4_metal_tensor_alloc(ffn_bytes);
    s->comp.ffn_up      = ds4_metal_tensor_alloc(ffn_bytes);
    s->comp.ffn_mid     = ds4_metal_tensor_alloc(ffn_bytes);
    s->comp.ffn_down    = ds4_metal_tensor_alloc(act_bytes);
    s->comp.moe_out     = ds4_metal_tensor_alloc(act_bytes);
    s->comp.shexp_out   = ds4_metal_tensor_alloc(act_bytes);

    s->hidden         = ds4_metal_tensor_alloc(act_bytes);
    s->output_normed  = ds4_metal_tensor_alloc(act_bytes);
    s->logits         = ds4_metal_tensor_alloc((uint64_t)m->n_vocab * sizeof(float));

    // Zero all compute buffers to eliminate uninitialized-memory non-determinism
    ds4_metal_tensor_fill(s->comp.normed,      0.0f);
    ds4_metal_tensor_fill(s->comp.qkv,         0.0f);
    ds4_metal_tensor_fill(s->comp.gate_out,    0.0f);
    ds4_metal_tensor_fill(s->comp.gla_out,     0.0f);
    ds4_metal_tensor_fill(s->comp.attn_proj,   0.0f);
    ds4_metal_tensor_fill(s->comp.post_attn,   0.0f);
    ds4_metal_tensor_fill(s->comp.ffn_normed,  0.0f);
    ds4_metal_tensor_fill(s->comp.ffn_gate,    0.0f);
    ds4_metal_tensor_fill(s->comp.ffn_up,      0.0f);
    ds4_metal_tensor_fill(s->comp.ffn_mid,     0.0f);
    ds4_metal_tensor_fill(s->comp.ffn_down,    0.0f);
    ds4_metal_tensor_fill(s->comp.moe_out,     0.0f);
    ds4_metal_tensor_fill(s->comp.shexp_out,   0.0f);
    ds4_metal_tensor_fill(s->hidden,           0.0f);
    ds4_metal_tensor_fill(s->output_normed,    0.0f);
    ds4_metal_tensor_fill(s->logits,           0.0f);

    // GLA states for all GLA layers (28 layers: all except 7, 15, 23, 31)
    for (uint32_t i = 0; i < 32; i++) {
        if (m->is_mla[i]) continue;
        s->gla_states[i].state = ds4_metal_tensor_alloc(gla_state_bytes);
        if (!s->gla_states[i].state) { fprintf(stderr, "l26f: OOM GLA state %u\n", i); return 0; }
        void *zeros = calloc(1, gla_state_bytes);
        if (zeros) {
            ds4_metal_tensor_write(s->gla_states[i].state, 0, zeros, gla_state_bytes);
            free(zeros);
        }
    }

    // MLA KV caches for MLA layers (7, 15, 23, 31)
    const uint32_t kv_dim = m->kv_lora_rank + m->n_rot;  // 512 + 64 = 576
    for (uint32_t i = 0; i < 32; i++) {
        if (!m->is_mla[i]) continue;
        s->mla_kv[i] = l26f_mla_kv_cache_alloc(4096, kv_dim);
        if (!s->mla_kv[i]) { fprintf(stderr, "l26f: OOM MLA KV cache %u\n", i); return 0; }
    }

    if (!s->comp.normed || !s->comp.qkv || !s->comp.gate_out ||
        !s->comp.gla_out || !s->comp.attn_proj || !s->comp.post_attn ||
        !s->comp.ffn_normed || !s->comp.ffn_gate || !s->comp.ffn_up ||
        !s->comp.ffn_mid || !s->comp.ffn_down ||
        !s->hidden || !s->output_normed || !s->logits) {
        fprintf(stderr, "l26f: OOM compute buffers\n");
        return 0;
    }

    return 1;
}

static void l26f_session_free(l26f_session *s) {
    ds4_metal_tensor_free(s->comp.normed);
    ds4_metal_tensor_free(s->comp.qkv);
    ds4_metal_tensor_free(s->comp.gate_out);
    ds4_metal_tensor_free(s->comp.gla_out);
    ds4_metal_tensor_free(s->comp.attn_proj);
    ds4_metal_tensor_free(s->comp.post_attn);
    ds4_metal_tensor_free(s->comp.ffn_normed);
    ds4_metal_tensor_free(s->comp.ffn_gate);
    ds4_metal_tensor_free(s->comp.ffn_up);
    ds4_metal_tensor_free(s->comp.ffn_mid);
    ds4_metal_tensor_free(s->comp.ffn_down);
    ds4_metal_tensor_free(s->comp.moe_out);
    ds4_metal_tensor_free(s->comp.shexp_out);
    ds4_metal_tensor_free(s->hidden);
    ds4_metal_tensor_free(s->output_normed);
    ds4_metal_tensor_free(s->logits);
    for (uint32_t i = 0; i < 32; i++) {
        if (s->gla_states[i].state)
            ds4_metal_tensor_free(s->gla_states[i].state);
        if (s->mla_kv[i])
            l26f_mla_kv_cache_free(s->mla_kv[i]);
    }
}

static int l26f_dequant_q8_0_row(const uint8_t *base, uint64_t row_offset,
        float *out, uint32_t n_elements) {
    const uint32_t block_size = 32;
    const uint32_t type_size = 34;
    const uint32_t n_blocks = n_elements / block_size;
    const uint8_t *row = base + row_offset;
    for (uint32_t b = 0; b < n_blocks; b++) {
        const uint8_t *block = row + (uint64_t)b * type_size;
        uint16_t scale_bits;
        memcpy(&scale_bits, block, 2);
        float scale = (float)*(const __fp16 *)&scale_bits;
        const int8_t *qs = (const int8_t *)(block + 2);
        for (uint32_t j = 0; j < block_size; j++) {
            out[b * block_size + j] = scale * (float)qs[j];
        }
    }
    return 1;
}

static int l26f_embed_token(l26f_session *s, uint32_t token) {
    l26f_model *m = s->model;
    l26f_tensor *embd = l26f_model_find_tensor(m, "token_embd.weight");
    if (!embd) { fprintf(stderr, "l26f: no token_embd\n"); return 0; }
    if (token >= m->n_vocab) { fprintf(stderr, "l26f: token %u >= vocab %u\n", token, m->n_vocab); return 0; }

    const uint32_t n_embd = m->n_embd;
    const uint64_t row_bytes = (uint64_t)(n_embd / 32) * 34;
    const uint64_t offset = (uint64_t)token * row_bytes;
    if (embd->abs_offset + offset + row_bytes > m->size) {
        fprintf(stderr, "l26f: embedding OOB\n"); return 0;
    }

    float *data = (float *)malloc((uint64_t)n_embd * sizeof(float));
    if (!data) return 0;
    l26f_dequant_q8_0_row(m->map + embd->abs_offset, offset, data, n_embd);
    int ok = ds4_metal_tensor_write(s->hidden, 0, data, (uint64_t)n_embd * sizeof(float));
    free(data);
    return ok;
}

static int l26f_output_logits(l26f_session *s) {
    l26f_model *m = s->model;
    l26f_tensor *wt_norm = l26f_model_find_tensor(m, "output_norm.weight");
    l26f_tensor *wt_out  = l26f_model_find_tensor(m, "output.weight");
    if (!wt_norm || !wt_out) {
        fprintf(stderr, "l26f: missing output tensors\n"); return 0;
    }

    if (!ds4_metal_rms_norm_weight_tensor(s->output_normed, s->hidden,
            m->map, m->size, wt_norm->abs_offset, m->n_embd, m->rms_norm_eps))
        return 0;

    if (!l26f_metal_matvec_quant(s->logits, s->output_normed,
            m->map, m->size, wt_out->abs_offset,
            wt_out->dim[0], wt_out->dim[1], wt_out->type, 1))
        return 0;

    return 1;
}

static int32_t l26f_argmax(l26f_session *s) {
    float *logits = (float *)malloc((uint64_t)s->model->n_vocab * sizeof(float));
    if (!logits) return -1;
    ds4_metal_tensor_read(s->logits, 0, logits, (uint64_t)s->model->n_vocab * sizeof(float));

    int32_t best = 0;
    float best_v = logits[0];
    for (uint32_t i = 1; i < s->model->n_vocab; i++) {
        if (logits[i] > best_v) { best_v = logits[i]; best = (int32_t)i; }
    }
    free(logits);
    return best;
}

static int l26f_forward_pass(l26f_session *session, l26f_model *model, int position, bool verbose) {
    for (uint32_t il = 0; il < 32; il++) {
        if (model->is_mla[il]) {
            const uint64_t act_bytes = (uint64_t)model->n_embd * sizeof(float);
            float *hidden_cpu = (float *)malloc(act_bytes);
            float *hidden_out = (float *)malloc(act_bytes);
            ds4_metal_tensor_read(session->hidden, 0, hidden_cpu, act_bytes);
            if (!l26f_mla_layer_cpu(model, il, position, session->mla_kv[il],
                                     hidden_cpu, hidden_out)) {
                fprintf(stderr, "MLA layer %u failed\n", il);
                free(hidden_cpu); free(hidden_out);
                return 0;
            }
            ds4_metal_tensor *out = session->comp.post_attn;
            ds4_metal_tensor_write(out, 0, hidden_out, act_bytes);
            ds4_metal_tensor_write(session->hidden, 0, hidden_out, act_bytes);
            free(hidden_cpu); free(hidden_out);

            if (!l26f_moe_ffn(session, il, out, session->hidden)) {
                fprintf(stderr, "MoE FFN layer %u failed\n", il);
                return 0;
            }
            if (verbose) printf("  layer %u: MLA + MoE\n", il);
            continue;
        }

        ds4_metal_tensor *inp = session->hidden;
        ds4_metal_tensor *out = session->comp.post_attn;

        if (!l26f_gla_layer(session, il, inp, out)) {
            fprintf(stderr, "GLA layer %u failed\n", il);
            return 0;
        }

        if (il == 0) {
            if (!l26f_dense_ffn(session, out, session->hidden)) {
                fprintf(stderr, "Dense FFN layer 0 failed\n");
                return 0;
            }
            if (verbose) printf("  layer %u: GLA + dense FFN\n", il);
        } else {
            if (!l26f_moe_ffn(session, il, out, session->hidden)) {
                fprintf(stderr, "MoE FFN layer %u failed\n", il);
                return 0;
            }
            if (verbose && il < 8) printf("  layer %u: GLA + MoE\n", il);
        }
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [prompt_or_token] [n_gen]\n", argv[0]);
        fprintf(stderr, "  If prompt starts with a digit, treated as token ID\n");
        fprintf(stderr, "  Otherwise, tokenized as text\n");
        return 1;
    }

    printf("Loading tokenizer...\n");
    l26f_tokenizer *tok = l26f_tokenizer_open(argv[1]);
    if (!tok) { fprintf(stderr, "Tokenizer load failed\n"); return 1; }
    printf("  vocab: %u tokens, %u merges, BOS=%d EOS=%d\n",
           tok->n_tokens, tok->n_merges, tok->bos_id, tok->eos_id);

    int32_t start_token;
    bool is_text = false;
    if (argc > 2) {
        // Check if argument is all digits → token ID, else text
        const char *arg = argv[2];
        is_text = false;
        for (int i = 0; arg[i]; i++) {
            if (!isdigit((unsigned char)arg[i]) && arg[i] != '-') { is_text = true; break; }
        }
        if (is_text) {
            int32_t encoded[256];
            int n = l26f_text_encode(tok, arg, encoded, 256);
            if (n == 0) { fprintf(stderr, "Encoding failed\n"); return 1; }
            start_token = encoded[0];
            printf("  encoded '%s' → %d tokens, using first: %d\n", arg, n, start_token);
        } else {
            start_token = atoi(argv[2]);
        }
    } else {
        start_token = 1;
    }
    int n_gen = argc > 3 ? atoi(argv[3]) : 16;

    char tokbuf[256];
    l26f_token_decode(tok, start_token, tokbuf, sizeof(tokbuf));
    printf("Starting from token %d (%s), generating %d tokens\n", start_token, tokbuf, n_gen);

    int32_t *generated = (int32_t *)malloc(n_gen * sizeof(int32_t));

    printf("Loading model...\n");
    l26f_model model;
    l26f_model_open(&model, argv[1]);

    printf("Metal init...\n");
    if (!ds4_metal_init()) { fprintf(stderr, "Metal init failed\n"); return 1; }

    printf("Model map...\n");
    if (!ds4_metal_set_model_map(model.map, model.size)) {
        fprintf(stderr, "Model map failed\n"); return 1;
    }

    l26f_session session;
    if (!l26f_session_init(&session, &model)) {
        fprintf(stderr, "Session init failed\n"); return 1;
    }

    printf("Embedding token %d...\n", start_token);
    if (!l26f_embed_token(&session, (uint32_t)start_token)) {
        fprintf(stderr, "Embed failed\n"); return 1;
    }

    printf("Generating %d tokens from token %d...\n", n_gen, start_token);
    int32_t current_token = start_token;

    for (int gen = 0; gen < n_gen; gen++) {
        printf("\n--- Token %d (input=%d) ---\n", gen, current_token);

        bool verbose = (gen < 2);
        if (!l26f_forward_pass(&session, &model, gen, verbose)) {
            fprintf(stderr, "Forward pass %d failed\n", gen);
            return 1;
        }

        if (!l26f_output_logits(&session)) {
            fprintf(stderr, "Output projection failed\n"); return 1;
        }

        int32_t next_token = l26f_argmax(&session);
        l26f_token_decode(tok, next_token, tokbuf, sizeof(tokbuf));
        printf("  -> token %d: \"%s\"\n", next_token, tokbuf);
        generated[gen] = next_token;

        {
            float *h = (float *)malloc((uint64_t)model.n_embd * sizeof(float));
            ds4_metal_tensor_read(session.hidden, 0, h, (uint64_t)model.n_embd * sizeof(float));
            float sum = 0; int nans = 0;
            for (uint32_t i = 0; i < model.n_embd; i++) {
                if (isnan(h[i])) nans++;
                sum += h[i];
            }
            if (nans > 0 || verbose) printf("  hidden: sum=%.3f nans=%d\n", sum, nans);
            free(h);
        }

        current_token = next_token;

        // Embed next token for next iteration
        if (gen + 1 < n_gen) {
            if (!l26f_embed_token(&session, (uint32_t)current_token)) {
                fprintf(stderr, "Embed failed for token %d\n", current_token);
                return 1;
            }
        }
    }

    printf("\nFull text: ");
    l26f_text_decode(tok, generated, n_gen, tokbuf, sizeof(tokbuf));
    printf("%s\n", tokbuf);

    l26f_session_free(&session);
    ds4_metal_cleanup();
    l26f_model_close(&model);
    l26f_tokenizer_close(tok);
    free(generated);
    return 0;
}
