// l26f: inference driver — single-token decode through GLA layers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
#include "l26f.h"
#include "l26f_metal.h"
#include "ds4_metal.h"
#include "l26f_tokenizer.h"

// Forward declarations for MLA CPU implementation
typedef struct l26f_mla_kv_cache l26f_mla_kv_cache;
extern l26f_mla_kv_cache *l26f_mla_kv_cache_alloc(int max_seq, int kv_dim);
extern void l26f_mla_kv_cache_free(l26f_mla_kv_cache *c);
extern int l26f_mla_layer_cpu(l26f_model *m, uint32_t layer, int position,
    l26f_mla_kv_cache *kv_cache, const float *hidden_cpu, float *hidden_out_cpu);

// Forward declarations for MLA GPU implementation
typedef struct l26f_mla_kv_cache_gpu l26f_mla_kv_cache_gpu;
typedef struct l26f_mla_compute l26f_mla_compute;

extern l26f_mla_kv_cache_gpu *l26f_mla_kv_cache_gpu_alloc(int max_seq, int kv_dim);
extern void l26f_mla_kv_cache_gpu_free(l26f_mla_kv_cache_gpu *c);
extern l26f_mla_compute *l26f_mla_compute_alloc(uint32_t n_embd, uint32_t n_head,
    uint32_t q_lora_rank, uint32_t kv_lora_rank, uint32_t n_rot, uint32_t head_dim);
extern void l26f_mla_compute_free(l26f_mla_compute *mc);
extern int l26f_mla_layer_gpu(l26f_model *m, uint32_t layer, int position,
    l26f_mla_kv_cache_gpu *kv_cache, l26f_mla_compute *mc,
    ds4_metal_tensor *hidden_1xN, ds4_metal_tensor *out_1xN);

static int use_gpu_mla(void) {
    static int initialized = 0;
    static int use_gpu = 0;
    if (!initialized) {
        use_gpu = getenv("L26F_CPU_MLA") == NULL;
        initialized = 1;
    }
    return use_gpu;
}

// ---- Per-layer compute buffer set ----
// Reused across layers to avoid excessive allocation.
// We need: normed, qkv, gate, gla_out+state, proj, ffn_mid, ffn_down
// All are n_embd-sized except qkv (3*n_embd) and gla_out (n_embd + S*S*H).

typedef struct {
    ds4_metal_tensor *normed_1xN;
    ds4_metal_tensor *qkv_1x3N;
    ds4_metal_tensor *gate_out_1xN;
    ds4_metal_tensor *gla_out_1xNxSxSxH;
    ds4_metal_tensor *attn_proj_1xN;
    ds4_metal_tensor *post_attn_1xN;
    ds4_metal_tensor *ffn_normed_1xN;
    ds4_metal_tensor *ffn_gate_1xF;
    ds4_metal_tensor *ffn_up_1xF;
    ds4_metal_tensor *ffn_mid_1xF;
    ds4_metal_tensor *ffn_down_1xN;
    ds4_metal_tensor *moe_out_1xN;
    ds4_metal_tensor *shexp_out_1xN;
    ds4_metal_tensor *router_logits_1xE;
    ds4_metal_tensor *moe_sel_idx_K;
    ds4_metal_tensor *moe_sel_wt_K;
    ds4_metal_tensor *moe_expert_gate_8xM;
    ds4_metal_tensor *moe_expert_up_8xM;
    ds4_metal_tensor *moe_expert_mid_8xM;
    ds4_metal_tensor *moe_expert_down_8xN;
    ds4_metal_tensor *moe_gate_off_K;
    ds4_metal_tensor *moe_up_off_K;
    ds4_metal_tensor *moe_down_off_K;
} l26f_compute;

// GLA state: one per GLA layer, persists across tokens
typedef struct {
    ds4_metal_tensor *state;
} l26f_gla_state;

typedef struct {
    l26f_model *model;
    l26f_compute comp;
    l26f_gla_state gla_states[32];
    l26f_mla_kv_cache_gpu *mla_kv_gpu[32];
    l26f_mla_kv_cache    *mla_kv_cpu[32];
    l26f_mla_compute *mla_comp;
    ds4_metal_tensor *hidden_1xN;
    ds4_metal_tensor *output_normed_1xN;
    ds4_metal_tensor *logits_1xV;
    ds4_metal_tensor *moe_gate_all_off[32];
    ds4_metal_tensor *moe_up_all_off[32];
    ds4_metal_tensor *moe_down_all_off[32];
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
    float scale = 1.0f / sqrtf((float)S);

    l26f_tensor *wt_norm_N      = l26f_layer_tensor(m, layer, "attn_norm.weight");
    l26f_tensor *wt_qkv_Nx3N    = l26f_layer_tensor(m, layer, "attn_qkv.weight");
    l26f_tensor *wt_gate_NxN    = l26f_layer_tensor(m, layer, "attn_gate.weight");
    l26f_tensor *wt_out_NxN     = l26f_layer_tensor(m, layer, "attn_output.weight");
    if (!wt_norm_N || !wt_qkv_Nx3N || !wt_gate_NxN || !wt_out_NxN) {
        fprintf(stderr, "l26f: layer %u missing GLA tensors\n", layer);
        return 0;
    }

    // 1. RMS norm
    if (!ds4_metal_rms_norm_weight_tensor(c->normed_1xN, inp,
            m->map, m->size, wt_norm_N->abs_offset, n_embd, m->rms_norm_eps))
        return 0;

    if (!l26f_metal_matvec_quant(c->qkv_1x3N, c->normed_1xN,
            m->map, m->size, wt_qkv_Nx3N->abs_offset,
            wt_qkv_Nx3N->dim[0], wt_qkv_Nx3N->dim[1], wt_qkv_Nx3N->type, 1))
        return 0;

    if (!l26f_metal_matvec_quant(c->gate_out_1xN, c->normed_1xN,
            m->map, m->size, wt_gate_NxN->abs_offset,
            wt_gate_NxN->dim[0], wt_gate_NxN->dim[1], wt_gate_NxN->type, 1))
        return 0;

    ds4_metal_tensor *q_view_1xN = ds4_metal_tensor_view(c->qkv_1x3N, 0, act_bytes);
    ds4_metal_tensor *k_view_1xN = ds4_metal_tensor_view(c->qkv_1x3N, act_bytes, act_bytes);
    ds4_metal_tensor *v_view_1xN = ds4_metal_tensor_view(c->qkv_1x3N, 2*act_bytes, act_bytes);

    int ok = l26f_metal_gla(c->gla_out_1xNxSxSxH, s->gla_states[layer].state,
                q_view_1xN, k_view_1xN, v_view_1xN, c->gate_out_1xN,
                1, 1, S, H, scale);

    ds4_metal_tensor_free(v_view_1xN);
    ds4_metal_tensor_free(k_view_1xN);
    ds4_metal_tensor_free(q_view_1xN);
    if (!ok) return 0;

    // 5. Output projection (first n_embd floats of gla_out are the attention output)
    ds4_metal_tensor *gla_act_1xN = ds4_metal_tensor_view(c->gla_out_1xNxSxSxH, 0, act_bytes);
    ok = l26f_metal_matvec_quant(c->attn_proj_1xN, gla_act_1xN,
            m->map, m->size, wt_out_NxN->abs_offset,
            wt_out_NxN->dim[0], wt_out_NxN->dim[1], wt_out_NxN->type, 1);
    ds4_metal_tensor_free(gla_act_1xN);
    if (!ok) return 0;

    // 6. Residual add: out = inp + attn_proj
    if (!ds4_metal_add_tensor(out, inp, c->attn_proj_1xN, n_embd))
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

    l26f_tensor *wt_norm_N     = l26f_layer_tensor(m, 0, "ffn_norm.weight");
    l26f_tensor *wt_gate_NxF   = l26f_layer_tensor(m, 0, "ffn_gate.weight");
    l26f_tensor *wt_up_NxF     = l26f_layer_tensor(m, 0, "ffn_up.weight");
    l26f_tensor *wt_down_FxN   = l26f_layer_tensor(m, 0, "ffn_down.weight");
    if (!wt_norm_N || !wt_gate_NxF || !wt_up_NxF || !wt_down_FxN) {
        fprintf(stderr, "l26f: layer 0 missing dense FFN tensors\n");
        return 0;
    }

    // 1. RMS norm
    if (!ds4_metal_rms_norm_weight_tensor(c->ffn_normed_1xN, inp,
            m->map, m->size, wt_norm_N->abs_offset, n_embd, m->rms_norm_eps))
        return 0;

    if (!l26f_metal_matvec_quant(c->ffn_gate_1xF, c->ffn_normed_1xN,
            m->map, m->size, wt_gate_NxF->abs_offset,
            wt_gate_NxF->dim[0], wt_gate_NxF->dim[1], wt_gate_NxF->type, 1))
        return 0;

    if (!l26f_metal_matvec_quant(c->ffn_up_1xF, c->ffn_normed_1xN,
            m->map, m->size, wt_up_NxF->abs_offset,
            wt_up_NxF->dim[0], wt_up_NxF->dim[1], wt_up_NxF->type, 1))
        return 0;

    if (!ds4_metal_swiglu_tensor(c->ffn_mid_1xF, c->ffn_gate_1xF, c->ffn_up_1xF, n_ff, 0.0f, 1.0f))
        return 0;

    if (!l26f_metal_matvec_quant(c->ffn_down_1xN, c->ffn_mid_1xF,
            m->map, m->size, wt_down_FxN->abs_offset,
            wt_down_FxN->dim[0], wt_down_FxN->dim[1], wt_down_FxN->type, 1))
        return 0;

    if (!ds4_metal_add_tensor(out, inp, c->ffn_down_1xN, n_embd))
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
    const uint32_t n_ff_exp = 1024;
    const uint32_t n_expert = 256;
    const uint32_t n_groups = 8;
    const uint32_t n_exp_per_group = n_expert / n_groups;
    const float w_scale = 2.5f;

    l26f_tensor *wt_norm_N          = l26f_layer_tensor(m, layer, "ffn_norm.weight");
    l26f_tensor *wt_gate_inp_NxE    = l26f_layer_tensor(m, layer, "ffn_gate_inp.weight");
    l26f_tensor *wt_exp_b_1xE       = l26f_layer_tensor(m, layer, "exp_probs_b.bias");
    l26f_tensor *wt_gate_exps_NxMxE = l26f_layer_tensor(m, layer, "ffn_gate_exps.weight");
    l26f_tensor *wt_up_exps_NxMxE   = l26f_layer_tensor(m, layer, "ffn_up_exps.weight");
    l26f_tensor *wt_down_exps_MxNxE = l26f_layer_tensor(m, layer, "ffn_down_exps.weight");
    l26f_tensor *wt_gate_sh_NxM     = l26f_layer_tensor(m, layer, "ffn_gate_shexp.weight");
    l26f_tensor *wt_up_sh_NxM       = l26f_layer_tensor(m, layer, "ffn_up_shexp.weight");
    l26f_tensor *wt_down_sh_MxN     = l26f_layer_tensor(m, layer, "ffn_down_shexp.weight");
    if (!wt_norm_N || !wt_gate_inp_NxE || !wt_gate_exps_NxMxE || !wt_up_exps_NxMxE || !wt_down_exps_MxNxE ||
        !wt_gate_sh_NxM || !wt_up_sh_NxM || !wt_down_sh_MxN) {
        fprintf(stderr, "l26f: layer %u missing MoE tensors\n", layer);
        return 0;
    }

    // 1. RMS norm
    if (!ds4_metal_rms_norm_weight_tensor(c->ffn_normed_1xN, inp,
            m->map, m->size, wt_norm_N->abs_offset, n_embd, m->rms_norm_eps))
        return 0;

    // 2. GPU router matvec + routing
    if (!ds4_metal_matmul_f32_tensor(c->router_logits_1xE, m->map, m->size,
            wt_gate_inp_NxE->abs_offset, n_embd, n_expert, c->ffn_normed_1xN, 1))
        return 0;

    int32_t has_bias = (wt_exp_b_1xE != NULL) ? 1 : 0;
    uint64_t bias_off = has_bias ? wt_exp_b_1xE->abs_offset : 0;
    if (!l26f_metal_moe_route(c->router_logits_1xE, m->map, m->size,
            bias_off, has_bias,
            c->moe_sel_idx_K, c->moe_sel_wt_K,
            n_expert, n_groups, n_exp_per_group, 4, 8, w_scale))
        return 0;

    // 3. Read back 8 indices + 8 weights (64 bytes)
    int32_t selected_experts_K[8];
    float selected_weights_K[8];
    ds4_metal_end_commands();
    ds4_metal_tensor_read(c->moe_sel_idx_K, 0, selected_experts_K, 8 * sizeof(int32_t));
    ds4_metal_tensor_read(c->moe_sel_wt_K,  0, selected_weights_K, 8 * sizeof(float));
    ds4_metal_begin_commands();

    // 4. Expert byte strides
    const uint32_t gate_bs = l26f_types[wt_gate_exps_NxMxE->type].block_size;
    const uint32_t gate_ts = l26f_types[wt_gate_exps_NxMxE->type].type_size;
    const uint32_t up_bs   = l26f_types[wt_up_exps_NxMxE->type].block_size;
    const uint32_t up_ts   = l26f_types[wt_up_exps_NxMxE->type].type_size;
    const uint32_t down_bs = l26f_types[wt_down_exps_MxNxE->type].block_size;
    const uint32_t down_ts = l26f_types[wt_down_exps_MxNxE->type].type_size;
    const uint64_t gate_exp_bytes = (uint64_t)wt_gate_exps_NxMxE->dim[1] * (wt_gate_exps_NxMxE->dim[0] / gate_bs) * gate_ts;
    const uint64_t up_exp_bytes   = (uint64_t)wt_up_exps_NxMxE->dim[1]   * (wt_up_exps_NxMxE->dim[0]   / up_bs)   * up_ts;
    const uint64_t down_exp_bytes = (uint64_t)wt_down_exps_MxNxE->dim[1] * (wt_down_exps_MxNxE->dim[0] / down_bs) * down_ts;

    // 5. Run 8 selected experts
    ds4_metal_tensor_fill(c->moe_out_1xN, 0.0f);

    for (int i = 0; i < 8; i++) {
        int e = selected_experts_K[i];
        uint64_t gate_off = wt_gate_exps_NxMxE->abs_offset + (uint64_t)e * gate_exp_bytes;
        uint64_t up_off   = wt_up_exps_NxMxE->abs_offset   + (uint64_t)e * up_exp_bytes;
        uint64_t down_off = wt_down_exps_MxNxE->abs_offset + (uint64_t)e * down_exp_bytes;

        if (!l26f_metal_matvec_quant(c->ffn_gate_1xF, c->ffn_normed_1xN,
                m->map, m->size, gate_off,
                n_embd, n_ff_exp, wt_gate_exps_NxMxE->type, 1))
            return 0;

        if (!l26f_metal_matvec_quant(c->ffn_up_1xF, c->ffn_normed_1xN,
                m->map, m->size, up_off,
                n_embd, n_ff_exp, wt_up_exps_NxMxE->type, 1))
            return 0;

        if (!ds4_metal_swiglu_tensor(c->ffn_mid_1xF, c->ffn_gate_1xF, c->ffn_up_1xF, n_ff_exp, 0.0f, 1.0f))
            return 0;

        if (!l26f_metal_matvec_quant(c->ffn_down_1xN, c->ffn_mid_1xF,
                m->map, m->size, down_off,
                n_ff_exp, n_embd, wt_down_exps_MxNxE->type, 1))
            return 0;

        if (!l26f_metal_axpy(c->moe_out_1xN, c->ffn_down_1xN, selected_weights_K[i], n_embd))
            return 0;
    }

    // 6. Shared expert
    if (!l26f_metal_matvec_quant(c->ffn_gate_1xF, c->ffn_normed_1xN,
            m->map, m->size, wt_gate_sh_NxM->abs_offset,
            wt_gate_sh_NxM->dim[0], wt_gate_sh_NxM->dim[1], wt_gate_sh_NxM->type, 1))
        return 0;
    if (!l26f_metal_matvec_quant(c->ffn_up_1xF, c->ffn_normed_1xN,
            m->map, m->size, wt_up_sh_NxM->abs_offset,
            wt_up_sh_NxM->dim[0], wt_up_sh_NxM->dim[1], wt_up_sh_NxM->type, 1))
        return 0;
    if (!ds4_metal_swiglu_tensor(c->ffn_mid_1xF, c->ffn_gate_1xF, c->ffn_up_1xF, n_ff_exp, 0.0f, 1.0f))
        return 0;
    if (!l26f_metal_matvec_quant(c->shexp_out_1xN, c->ffn_mid_1xF,
            m->map, m->size, wt_down_sh_MxN->abs_offset,
            wt_down_sh_MxN->dim[0], wt_down_sh_MxN->dim[1], wt_down_sh_MxN->type, 1))
        return 0;

    if (!l26f_metal_axpy(c->moe_out_1xN, c->shexp_out_1xN, 1.0f, n_embd))
        return 0;

    // 7. Residual add
    if (!ds4_metal_add_tensor(out, inp, c->moe_out_1xN, n_embd))
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
    s->comp.normed_1xN      = ds4_metal_tensor_alloc(act_bytes);
    s->comp.qkv_1x3N         = ds4_metal_tensor_alloc(qkv_bytes);
    s->comp.gate_out_1xN    = ds4_metal_tensor_alloc(act_bytes);
    s->comp.gla_out_1xNxSxSxH     = ds4_metal_tensor_alloc(gla_out_bytes);
    s->comp.attn_proj_1xN   = ds4_metal_tensor_alloc(act_bytes);
    s->comp.post_attn_1xN   = ds4_metal_tensor_alloc(act_bytes);
    s->comp.ffn_normed_1xN  = ds4_metal_tensor_alloc(act_bytes);
    s->comp.ffn_gate_1xF    = ds4_metal_tensor_alloc(ffn_bytes);
    s->comp.ffn_up_1xF      = ds4_metal_tensor_alloc(ffn_bytes);
    s->comp.ffn_mid_1xF     = ds4_metal_tensor_alloc(ffn_bytes);
    s->comp.ffn_down_1xN    = ds4_metal_tensor_alloc(act_bytes);
    s->comp.moe_out_1xN     = ds4_metal_tensor_alloc(act_bytes);
    s->comp.shexp_out_1xN   = ds4_metal_tensor_alloc(act_bytes);
    s->comp.router_logits_1xE = ds4_metal_tensor_alloc(256 * sizeof(float));
    s->comp.moe_sel_idx_K    = ds4_metal_tensor_alloc(8 * sizeof(int32_t));
    s->comp.moe_sel_wt_K     = ds4_metal_tensor_alloc(8 * sizeof(float));
    s->comp.moe_expert_gate_8xM = ds4_metal_tensor_alloc((uint64_t)8 * n_ff * sizeof(float));
    s->comp.moe_expert_up_8xM   = ds4_metal_tensor_alloc((uint64_t)8 * n_ff * sizeof(float));
    s->comp.moe_expert_mid_8xM  = ds4_metal_tensor_alloc((uint64_t)8 * n_ff * sizeof(float));
    s->comp.moe_expert_down_8xN = ds4_metal_tensor_alloc((uint64_t)8 * n_embd * sizeof(float));
    s->comp.moe_gate_off_K  = ds4_metal_tensor_alloc(8 * sizeof(uint64_t));
    s->comp.moe_up_off_K    = ds4_metal_tensor_alloc(8 * sizeof(uint64_t));
    s->comp.moe_down_off_K  = ds4_metal_tensor_alloc(8 * sizeof(uint64_t));

    s->hidden_1xN         = ds4_metal_tensor_alloc(act_bytes);
    s->output_normed_1xN  = ds4_metal_tensor_alloc(act_bytes);
    s->logits_1xV         = ds4_metal_tensor_alloc((uint64_t)m->n_vocab * sizeof(float));

    // Zero all compute buffers to eliminate uninitialized-memory non-determinism
    ds4_metal_tensor_fill(s->comp.normed_1xN,      0.0f);
    ds4_metal_tensor_fill(s->comp.qkv_1x3N,         0.0f);
    ds4_metal_tensor_fill(s->comp.gate_out_1xN,    0.0f);
    ds4_metal_tensor_fill(s->comp.gla_out_1xNxSxSxH,     0.0f);
    ds4_metal_tensor_fill(s->comp.attn_proj_1xN,   0.0f);
    ds4_metal_tensor_fill(s->comp.post_attn_1xN,   0.0f);
    ds4_metal_tensor_fill(s->comp.ffn_normed_1xN,  0.0f);
    ds4_metal_tensor_fill(s->comp.ffn_gate_1xF,    0.0f);
    ds4_metal_tensor_fill(s->comp.ffn_up_1xF,      0.0f);
    ds4_metal_tensor_fill(s->comp.ffn_mid_1xF,     0.0f);
    ds4_metal_tensor_fill(s->comp.ffn_down_1xN,    0.0f);
    ds4_metal_tensor_fill(s->comp.moe_out_1xN,     0.0f);
    ds4_metal_tensor_fill(s->comp.shexp_out_1xN,   0.0f);
    ds4_metal_tensor_fill(s->comp.moe_sel_idx_K,   0.0f);
    ds4_metal_tensor_fill(s->comp.moe_sel_wt_K,    0.0f);
    ds4_metal_tensor_fill(s->comp.moe_expert_gate_8xM, 0.0f);
    ds4_metal_tensor_fill(s->comp.moe_expert_up_8xM,   0.0f);
    ds4_metal_tensor_fill(s->comp.moe_expert_mid_8xM,  0.0f);
    ds4_metal_tensor_fill(s->comp.moe_expert_down_8xN, 0.0f);
    ds4_metal_tensor_fill(s->comp.moe_gate_off_K,  0.0f);
    ds4_metal_tensor_fill(s->comp.moe_up_off_K,    0.0f);
    ds4_metal_tensor_fill(s->comp.moe_down_off_K,  0.0f);
    ds4_metal_tensor_fill(s->hidden_1xN,           0.0f);
    ds4_metal_tensor_fill(s->output_normed_1xN,    0.0f);
    ds4_metal_tensor_fill(s->logits_1xV,           0.0f);

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
        if (use_gpu_mla()) {
            s->mla_kv_gpu[i] = l26f_mla_kv_cache_gpu_alloc(4096, kv_dim);
            if (!s->mla_kv_gpu[i]) { fprintf(stderr, "l26f: OOM MLA GPU KV %u\n", i); return 0; }
        } else {
            s->mla_kv_cpu[i] = l26f_mla_kv_cache_alloc(4096, kv_dim);
            if (!s->mla_kv_cpu[i]) { fprintf(stderr, "l26f: OOM MLA CPU KV %u\n", i); return 0; }
        }
    }

    // MLA GPU compute buffers (shared across all MLA layers) — only if GPU path
    if (use_gpu_mla()) {
        s->mla_comp = l26f_mla_compute_alloc(n_embd, m->n_head, m->q_lora_rank,
                                               m->kv_lora_rank, m->n_rot, 192);
        if (!s->mla_comp) { fprintf(stderr, "l26f: OOM MLA compute buffers\n"); return 0; }
    }

    // Precompute per-layer expert weight offset tables (256 entries × 8 bytes each)
    for (uint32_t il = 0; il < 32; il++) {
        if (il == 0) continue;  // layer 0 uses dense FFN, not MoE

        l26f_tensor *wt_gate = l26f_layer_tensor(m, il, "ffn_gate_exps.weight");
        l26f_tensor *wt_up   = l26f_layer_tensor(m, il, "ffn_up_exps.weight");
        l26f_tensor *wt_down = l26f_layer_tensor(m, il, "ffn_down_exps.weight");
        if (!wt_gate || !wt_up || !wt_down) continue;

        const uint32_t gate_bs = l26f_types[wt_gate->type].block_size;
        const uint32_t gate_ts = l26f_types[wt_gate->type].type_size;
        const uint32_t up_bs   = l26f_types[wt_up->type].block_size;
        const uint32_t up_ts   = l26f_types[wt_up->type].type_size;
        const uint32_t down_bs = l26f_types[wt_down->type].block_size;
        const uint32_t down_ts = l26f_types[wt_down->type].type_size;
        const uint64_t gate_exp_bytes = (uint64_t)wt_gate->dim[1] * ((uint64_t)wt_gate->dim[0] / gate_bs) * gate_ts;
        const uint64_t up_exp_bytes   = (uint64_t)wt_up->dim[1]   * ((uint64_t)wt_up->dim[0]   / up_bs)   * up_ts;
        const uint64_t down_exp_bytes = (uint64_t)wt_down->dim[1] * ((uint64_t)wt_down->dim[0] / down_bs) * down_ts;

        uint64_t gate_offs[256], up_offs[256], down_offs[256];
        for (int e = 0; e < 256; e++) {
            gate_offs[e] = (uint64_t)e * gate_exp_bytes;
            up_offs[e]   = (uint64_t)e * up_exp_bytes;
            down_offs[e] = (uint64_t)e * down_exp_bytes;
        }

        s->moe_gate_all_off[il] = ds4_metal_tensor_alloc(256 * sizeof(uint64_t));
        s->moe_up_all_off[il]   = ds4_metal_tensor_alloc(256 * sizeof(uint64_t));
        s->moe_down_all_off[il] = ds4_metal_tensor_alloc(256 * sizeof(uint64_t));
        if (!s->moe_gate_all_off[il] || !s->moe_up_all_off[il] || !s->moe_down_all_off[il]) {
            fprintf(stderr, "l26f: OOM MoE offset tables\n"); return 0;
        }
        ds4_metal_tensor_write(s->moe_gate_all_off[il], 0, gate_offs, sizeof(gate_offs));
        ds4_metal_tensor_write(s->moe_up_all_off[il],   0, up_offs,   sizeof(up_offs));
        ds4_metal_tensor_write(s->moe_down_all_off[il], 0, down_offs, sizeof(down_offs));
    }

    if (!s->comp.normed_1xN || !s->comp.qkv_1x3N || !s->comp.gate_out_1xN ||
        !s->comp.gla_out_1xNxSxSxH || !s->comp.attn_proj_1xN || !s->comp.post_attn_1xN ||
        !s->comp.ffn_normed_1xN || !s->comp.ffn_gate_1xF || !s->comp.ffn_up_1xF ||
        !s->comp.ffn_mid_1xF || !s->comp.ffn_down_1xN ||
        !s->hidden_1xN || !s->output_normed_1xN || !s->logits_1xV) {
        fprintf(stderr, "l26f: OOM compute buffers\n");
        return 0;
    }

    return 1;
}

static void l26f_session_free(l26f_session *s) {
    ds4_metal_tensor_free(s->comp.normed_1xN);
    ds4_metal_tensor_free(s->comp.qkv_1x3N);
    ds4_metal_tensor_free(s->comp.gate_out_1xN);
    ds4_metal_tensor_free(s->comp.gla_out_1xNxSxSxH);
    ds4_metal_tensor_free(s->comp.attn_proj_1xN);
    ds4_metal_tensor_free(s->comp.post_attn_1xN);
    ds4_metal_tensor_free(s->comp.ffn_normed_1xN);
    ds4_metal_tensor_free(s->comp.ffn_gate_1xF);
    ds4_metal_tensor_free(s->comp.ffn_up_1xF);
    ds4_metal_tensor_free(s->comp.ffn_mid_1xF);
    ds4_metal_tensor_free(s->comp.ffn_down_1xN);
    ds4_metal_tensor_free(s->comp.moe_out_1xN);
    ds4_metal_tensor_free(s->comp.shexp_out_1xN);
    ds4_metal_tensor_free(s->comp.moe_sel_idx_K);
    ds4_metal_tensor_free(s->comp.moe_sel_wt_K);
    ds4_metal_tensor_free(s->comp.moe_expert_gate_8xM);
    ds4_metal_tensor_free(s->comp.moe_expert_up_8xM);
    ds4_metal_tensor_free(s->comp.moe_expert_mid_8xM);
    ds4_metal_tensor_free(s->comp.moe_expert_down_8xN);
    ds4_metal_tensor_free(s->comp.moe_gate_off_K);
    ds4_metal_tensor_free(s->comp.moe_up_off_K);
    ds4_metal_tensor_free(s->comp.moe_down_off_K);
    ds4_metal_tensor_free(s->comp.router_logits_1xE);
    ds4_metal_tensor_free(s->hidden_1xN);
    ds4_metal_tensor_free(s->output_normed_1xN);
    ds4_metal_tensor_free(s->logits_1xV);
    for (uint32_t i = 0; i < 32; i++) {
        if (s->gla_states[i].state)
            ds4_metal_tensor_free(s->gla_states[i].state);
        if (s->mla_kv_gpu[i])
            l26f_mla_kv_cache_gpu_free(s->mla_kv_gpu[i]);
        if (s->mla_kv_cpu[i])
            l26f_mla_kv_cache_free(s->mla_kv_cpu[i]);
    }
    if (s->mla_comp)
        l26f_mla_compute_free(s->mla_comp);
    for (uint32_t i = 0; i < 32; i++) {
        if (s->moe_gate_all_off[i]) ds4_metal_tensor_free(s->moe_gate_all_off[i]);
        if (s->moe_up_all_off[i])   ds4_metal_tensor_free(s->moe_up_all_off[i]);
        if (s->moe_down_all_off[i]) ds4_metal_tensor_free(s->moe_down_all_off[i]);
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
    int ok = ds4_metal_tensor_write(s->hidden_1xN, 0, data, (uint64_t)n_embd * sizeof(float));
    free(data);
    return ok;
}

static int l26f_output_logits(l26f_session *s) {
    l26f_model *m = s->model;
    l26f_tensor *wt_norm_N   = l26f_model_find_tensor(m, "output_norm.weight");
    l26f_tensor *wt_out_NxV  = l26f_model_find_tensor(m, "output.weight");
    if (!wt_norm_N || !wt_out_NxV) {
        fprintf(stderr, "l26f: missing output tensors\n"); return 0;
    }

    if (!ds4_metal_rms_norm_weight_tensor(s->output_normed_1xN, s->hidden_1xN,
            m->map, m->size, wt_norm_N->abs_offset, m->n_embd, m->rms_norm_eps))
        return 0;

    if (!l26f_metal_matvec_quant(s->logits_1xV, s->output_normed_1xN,
            m->map, m->size, wt_out_NxV->abs_offset,
            wt_out_NxV->dim[0], wt_out_NxV->dim[1], wt_out_NxV->type, 1))
        return 0;

    return 1;
}

typedef struct {
    float temperature;
    int32_t top_k;
    float top_p;
    uint64_t seed;
} l26f_sample_params;

static uint64_t l26f_rng_next(uint64_t *state) {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    return *state;
}

static float l26f_rng_float(uint64_t *state) {
    return (float)(l26f_rng_next(state) & 0xFFFFFF) / (float)0x1000000;
}

typedef struct {
    float val;
    int32_t idx;
} l26f_logit_entry;

static int l26f_logit_cmp_desc(const void *a, const void *b) {
    float fa = ((const l26f_logit_entry *)a)->val;
    float fb = ((const l26f_logit_entry *)b)->val;
    return (fa < fb) - (fa > fb);
}

static int32_t l26f_sample(l26f_session *s, const l26f_sample_params *p, uint64_t *rng) {
    const uint32_t V = s->model->n_vocab;
    float *logits = (float *)malloc((uint64_t)V * sizeof(float));
    if (!logits) return -1;
    ds4_metal_tensor_read(s->logits_1xV, 0, logits, (uint64_t)V * sizeof(float));

    if (p->temperature <= 0.0f) {
        int32_t best = 0;
        for (uint32_t i = 1; i < V; i++)
            if (logits[i] > logits[best]) best = (int32_t)i;
        free(logits);
        return best;
    }

    for (uint32_t i = 0; i < V; i++) logits[i] /= p->temperature;

    l26f_logit_entry *entries = (l26f_logit_entry *)malloc((uint64_t)V * sizeof(l26f_logit_entry));
    for (uint32_t i = 0; i < V; i++) {
        entries[i].val = logits[i];
        entries[i].idx = (int32_t)i;
    }

    int32_t k = V;
    if (p->top_k > 0 && p->top_k < (int32_t)V) k = p->top_k;
    int32_t partial = (k < V) ? k : V;
    if (partial < V) {
        qsort(entries, V, sizeof(l26f_logit_entry), l26f_logit_cmp_desc);
    }

    float max_v = entries[0].val;
    for (int32_t i = 0; i < partial; i++)
        entries[i].val = expf(entries[i].val - max_v);

    if (p->top_p < 1.0f) {
        float cumsum = 0.0f, total = 0.0f;
        for (int32_t i = 0; i < partial; i++) total += entries[i].val;
        for (int32_t i = 0; i < partial; i++) {
            cumsum += entries[i].val;
            if (cumsum / total > p->top_p) {
                partial = i + 1;
                break;
            }
        }
    }

    float sum = 0.0f;
    for (int32_t i = 0; i < partial; i++) sum += entries[i].val;
    float r = l26f_rng_float(rng) * sum;
    int32_t result = entries[partial - 1].idx;
    float cumsum = 0.0f;
    for (int32_t i = 0; i < partial; i++) {
        cumsum += entries[i].val;
        if (cumsum >= r) { result = entries[i].idx; break; }
    }

    free(entries);
    free(logits);
    return result;
}

static int32_t l26f_argmax(l26f_session *s) {
    l26f_sample_params p = {0, 0, 1.0f, 0};
    uint64_t rng = 0;
    return l26f_sample(s, &p, &rng);
}

static int l26f_forward_pass(l26f_session *session, l26f_model *model, int position, bool verbose) {
    const uint64_t act_bytes = (uint64_t)model->n_embd * sizeof(float);

    if (!ds4_metal_begin_commands()) {
        fprintf(stderr, "begin_commands failed\n");
        return 0;
    }

    for (uint32_t il = 0; il < 32; il++) {
        if (model->is_mla[il]) {
            if (use_gpu_mla()) {
                // GPU MLA: all ops are GPU kernels, no command buffer break needed
                ds4_metal_tensor *mla_out_1xN = session->comp.post_attn_1xN;
                if (!l26f_mla_layer_gpu(model, il, position, session->mla_kv_gpu[il],
                                         session->mla_comp, session->hidden_1xN, mla_out_1xN)) {
                    fprintf(stderr, "MLA GPU layer %u failed\n", il);
                    return 0;
                }
                if (!l26f_moe_ffn(session, il, mla_out_1xN, session->hidden_1xN)) {
                    fprintf(stderr, "MoE FFN layer %u failed\n", il);
                    return 0;
                }
            } else {
                ds4_metal_tensor *out_1xN = session->comp.post_attn_1xN;
                ds4_metal_end_commands();
                float *hidden_cpu = (float *)malloc(act_bytes);
                float *hidden_out = (float *)malloc(act_bytes);
                ds4_metal_tensor_read(session->hidden_1xN, 0, hidden_cpu, act_bytes);
                if (!l26f_mla_layer_cpu(model, il, position, session->mla_kv_cpu[il],
                                         hidden_cpu, hidden_out)) {
                    fprintf(stderr, "MLA CPU layer %u failed\n", il);
                    free(hidden_cpu); free(hidden_out);
                    return 0;
                }
                ds4_metal_tensor_write(session->hidden_1xN, 0, hidden_out, act_bytes);
                ds4_metal_tensor_write(out_1xN, 0, hidden_out, act_bytes);
                free(hidden_cpu);
                free(hidden_out);
                ds4_metal_begin_commands();
                if (!l26f_moe_ffn(session, il, out_1xN, session->hidden_1xN)) {
                    fprintf(stderr, "MoE FFN layer %u failed\n", il);
                    return 0;
                }
            }
            if (verbose) printf("  layer %u: MLA %s + MoE\n", il, use_gpu_mla() ? "GPU" : "CPU");
            continue;
        }

        ds4_metal_tensor *inp_1xN = session->hidden_1xN;
        ds4_metal_tensor *out_1xN = session->comp.post_attn_1xN;

        if (!l26f_gla_layer(session, il, inp_1xN, out_1xN)) {
            fprintf(stderr, "GLA layer %u failed\n", il);
            return 0;
        }

        if (il == 0) {
            if (!l26f_dense_ffn(session, out_1xN, session->hidden_1xN)) {
                fprintf(stderr, "Dense FFN layer 0 failed\n");
                return 0;
            }
            if (verbose) printf("  layer %u: GLA + dense FFN\n", il);
        } else {
            if (!l26f_moe_ffn(session, il, out_1xN, session->hidden_1xN)) {
                fprintf(stderr, "MoE FFN layer %u failed\n", il);
                return 0;
            }
            if (verbose && il < 8) printf("  layer %u: GLA + MoE\n", il);
        }
    }

    if (!ds4_metal_end_commands()) {
        fprintf(stderr, "end_commands failed\n");
        return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [prompt_or_token] [n_gen] [temp] [top_k] [top_p]\n", argv[0]);
        fprintf(stderr, "  If prompt starts with a digit, treated as token ID\n");
        fprintf(stderr, "  Otherwise, tokenized as text\n");
        fprintf(stderr, "  temp=0 → greedy (argmax), default=0.8\n");
        return 1;
    }

    printf("Loading model...\n");
    l26f_model model;
    l26f_model_open(&model, argv[1]);

    printf("Loading tokenizer...\n");
    l26f_tokenizer *tok = l26f_tokenizer_from_model(&model);
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

    l26f_sample_params sample_params = {
        .temperature = argc > 4 ? (float)atof(argv[4]) : 0.0f,
        .top_k       = argc > 5 ? atoi(argv[5]) : 40,
        .top_p       = argc > 6 ? (float)atof(argv[6]) : 0.95f,
        .seed        = 42
    };
    uint64_t rng = sample_params.seed;

    char tokbuf[256];
    l26f_token_decode(tok, start_token, tokbuf, sizeof(tokbuf));
    printf("Starting from token %d (%s), generating %d tokens\n", start_token, tokbuf, n_gen);

    int32_t *generated = (int32_t *)malloc(n_gen * sizeof(int32_t));

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

    printf("Generating %d tokens from token %d (temp=%.2f, top_k=%d, top_p=%.2f)...\n",
           n_gen, start_token, sample_params.temperature, sample_params.top_k, sample_params.top_p);
    int32_t current_token = start_token;
    int n_generated = 0;
    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

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

        int32_t next_token = l26f_sample(&session, &sample_params, &rng);
        l26f_token_decode(tok, next_token, tokbuf, sizeof(tokbuf));
        printf("  -> token %d: \"%s\"\n", next_token, tokbuf);
        generated[gen] = next_token;
        n_generated = gen + 1;

        if (next_token == tok->eos_id) {
            printf("  (EOS reached)\n");
            break;
        }

        current_token = next_token;

        if (gen + 1 < n_gen) {
            if (!l26f_embed_token(&session, (uint32_t)current_token)) {
                fprintf(stderr, "Embed failed for token %d\n", current_token);
                return 1;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double elapsed = (t_end.tv_sec - t_start.tv_sec) + (t_end.tv_nsec - t_start.tv_nsec) / 1e9;
    printf("\nFull text: ");
    l26f_text_decode(tok, generated, n_generated, tokbuf, sizeof(tokbuf));
    printf("%s\n", tokbuf);
    printf("\n%d tokens in %.3fs (%.1f tok/s)\n", n_generated, elapsed,
           n_generated > 0 ? (double)n_generated / elapsed : 0.0);

    l26f_session_free(&session);
    ds4_metal_cleanup();
    l26f_model_close(&model);
    l26f_tokenizer_close(tok);
    free(generated);
    return 0;
}
