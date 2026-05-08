// l26f: inference driver — single-token decode through GLA layers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "l26f.h"
#include "l26f_metal.h"
#include "ds4_metal.h"

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
} l26f_compute;

// GLA state: one per GLA layer, persists across tokens
typedef struct {
    ds4_metal_tensor *state;
} l26f_gla_state;

typedef struct {
    l26f_model *model;
    l26f_compute comp;
    l26f_gla_state gla_states[32];
    ds4_metal_tensor *hidden;
    ds4_metal_tensor *output_normed;
    ds4_metal_tensor *logits;
} l26f_session;

// ---- helpers ----

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

    // Allocate compute buffers
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

    // Hidden state (ping-pong between layers)
    s->hidden         = ds4_metal_tensor_alloc(act_bytes);
    s->output_normed  = ds4_metal_tensor_alloc(act_bytes);
    s->logits         = ds4_metal_tensor_alloc((uint64_t)m->n_vocab * sizeof(float));

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
    ds4_metal_tensor_free(s->hidden);
    ds4_metal_tensor_free(s->output_normed);
    ds4_metal_tensor_free(s->logits);
    for (uint32_t i = 0; i < 32; i++) {
        if (s->gla_states[i].state)
            ds4_metal_tensor_free(s->gla_states[i].state);
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

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [token_id]\n", argv[0]);
        return 1;
    }
    int start_token = argc > 2 ? atoi(argv[2]) : 1;

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

    printf("Running 32 transformer layers...\n");
    for (uint32_t il = 0; il < 32; il++) {
        if (model.is_mla[il]) {
            printf("  layer %u: MLA (skipped, passing through)\n", il);
            continue;
        }

        ds4_metal_tensor *inp = session.hidden;
        ds4_metal_tensor *out = session.comp.post_attn;

        if (!l26f_gla_layer(&session, il, inp, out)) {
            fprintf(stderr, "GLA layer %u failed\n", il);
            return 1;
        }

        // Dense FFN for layer 0 only
        if (il == 0) {
            printf("  layer %u: GLA + dense FFN\n", il);
            ds4_metal_tensor *ffn_in = out;
            ds4_metal_tensor *ffn_out = session.hidden;
            if (!l26f_dense_ffn(&session, ffn_in, ffn_out)) {
                fprintf(stderr, "Dense FFN layer 0 failed\n");
                return 1;
            }
        } else {
            printf("  layer %u: GLA (MoE FFN skipped)\n", il);
            ds4_metal_tensor_copy(session.hidden, 0,
                session.comp.post_attn, 0,
                (uint64_t)model.n_embd * sizeof(float));
        }

        if (il < 3 || il == 31) {
            float *h = (float *)malloc((uint64_t)model.n_embd * sizeof(float));
            ds4_metal_tensor_read(session.hidden, 0, h, (uint64_t)model.n_embd * sizeof(float));
            float sum = 0, mn = h[0], mx = h[0]; int nans = 0;
            for (uint32_t i = 0; i < model.n_embd; i++) {
                if (isnan(h[i])) { nans++; continue; }
                sum += h[i];
                if (h[i] < mn) mn = h[i];
                if (h[i] > mx) mx = h[i];
            }
            printf("    -> sum=%.3f min=%.3f max=%.3f nans=%d\n", sum, mn, mx, nans);
            free(h);
        }
    }

    printf("Output projection...\n");

    // Debug: check hidden state before output
    {
        float *h = (float *)malloc((uint64_t)model.n_embd * sizeof(float));
        ds4_metal_tensor_read(session.hidden, 0, h, (uint64_t)model.n_embd * sizeof(float));
        float sum = 0; int nans = 0;
        for (uint32_t i = 0; i < model.n_embd; i++) {
            if (isnan(h[i])) nans++;
            sum += h[i];
        }
        printf("  hidden before output: sum=%.3f nans=%d/%u\n", sum, nans, model.n_embd);
        free(h);
    }

    if (!l26f_output_logits(&session)) {
        fprintf(stderr, "Output projection failed\n"); return 1;
    }

    int32_t next_token = l26f_argmax(&session);
    printf("\nResult: token %d -> next_token %d\n", start_token, next_token);

    float *logits_sample = (float *)malloc(10 * sizeof(float));
    ds4_metal_tensor_read(session.logits, 0, logits_sample, 10 * sizeof(float));
    printf("First 10 logits: ");
    for (int i = 0; i < 10; i++) printf("%.4f ", logits_sample[i]);
    printf("\n");
    free(logits_sample);

    l26f_session_free(&session);
    ds4_metal_cleanup();
    l26f_model_close(&model);
    return 0;
}
