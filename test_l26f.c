// l26f: GLA layer end-to-end Metal test
#include "l26f.h"
#include "l26f_metal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void test_gla_layer(l26f_model *m) {
    printf("\n=== GLA Layer 0 End-to-End Test ===\n");

    const uint32_t n_embd = m->n_embd;  // 4096
    const uint32_t S = 128, H = 32;
    const uint64_t act_bytes = n_embd * sizeof(float);
    const uint64_t qkv_bytes = 3 * n_embd * sizeof(float);  // Q+K+V fused
    const uint64_t state_bytes = (uint64_t)S * S * H * sizeof(float);  // 524288 * 4 = 2 MiB
    const uint64_t out_gla_bytes = (n_embd + S * S * H) * sizeof(float);  // acts + state

    // Allocate tensors
    ds4_metal_tensor *inp    = ds4_metal_tensor_alloc(act_bytes);
    ds4_metal_tensor *normed = ds4_metal_tensor_alloc(act_bytes);
    ds4_metal_tensor *qkv    = ds4_metal_tensor_alloc(qkv_bytes);
    ds4_metal_tensor *gate   = ds4_metal_tensor_alloc(act_bytes);
    ds4_metal_tensor *gla_out = ds4_metal_tensor_alloc(out_gla_bytes);
    ds4_metal_tensor *state  = ds4_metal_tensor_alloc(state_bytes);
    ds4_metal_tensor *proj   = ds4_metal_tensor_alloc(act_bytes);
    ds4_metal_tensor *out    = ds4_metal_tensor_alloc(act_bytes);

    // Fill input with test data
    float *tmp = (float *)calloc(n_embd, sizeof(float));
    for (uint32_t i = 0; i < n_embd; i++) tmp[i] = (float)((i * 7 + 13) % 1000) / 1000.0f;
    ds4_metal_tensor_write(inp, 0, tmp, act_bytes);

    // Zero the state
    memset(tmp, 0, state_bytes);
    ds4_metal_tensor_write(state, 0, tmp, state_bytes);
    free(tmp);

    // Find weights
    l26f_tensor *wt_norm = l26f_model_find_tensor(m, "blk.0.attn_norm.weight");
    l26f_tensor *wt_qkv  = l26f_model_find_tensor(m, "blk.0.attn_qkv.weight");
    l26f_tensor *wt_gate = l26f_model_find_tensor(m, "blk.0.attn_gate.weight");
    l26f_tensor *wt_out  = l26f_model_find_tensor(m, "blk.0.attn_output.weight");

    if (!wt_norm || !wt_qkv || !wt_gate || !wt_out) {
        fprintf(stderr, "Missing layer 0 tensors\n");
        return;
    }

    printf("Weights: norm(f32), qkv(q5_k %u×%u), gate(q5_k %u×%u), out(q5_k %u×%u)\n",
           (unsigned)wt_qkv->dim[0], (unsigned)wt_qkv->dim[1],
           (unsigned)wt_gate->dim[0], (unsigned)wt_gate->dim[1],
           (unsigned)wt_out->dim[0], (unsigned)wt_out->dim[1]);

    // Step 1: RMS Norm
    printf("1. RMS norm...\n");
    if (!ds4_metal_rms_norm_weight_tensor(normed, inp, m->map, m->size,
                                          wt_norm->abs_offset,
                                          n_embd, 1e-6f)) {
        fprintf(stderr, "RMS norm failed\n"); return;
    }

    // Step 2: attn_qkv matvec
    printf("2. QKV matvec...\n");
    if (!l26f_metal_matvec_quant(qkv, normed, m->map, m->size,
                                 wt_qkv->abs_offset,
                                 wt_qkv->dim[0], wt_qkv->dim[1], wt_qkv->type, 1)) {
        fprintf(stderr, "QKV matvec failed\n"); return;
    }

    // Step 3: attn_gate matvec
    printf("3. Gate matvec...\n");
    if (!l26f_metal_matvec_quant(gate, normed, m->map, m->size,
                                 wt_gate->abs_offset,
                                 wt_gate->dim[0], wt_gate->dim[1], wt_gate->type, 1)) {
        fprintf(stderr, "Gate matvec failed\n"); return;
    }

    // Step 4: GLA attention
    // Q=[4096]=[S,H], K=[4096]=[S,H], V=[4096]=[S,H], G=[4096]=[S,H]
    // But GLA kernel expects Q, K, V, G as separate buffers, each [4096]
    // qkv buffer already has Q(0-4095), K(4096-8191), V(8192-12287)
    // We need to pass views into the qkv buffer
    printf("4. GLA attention...\n");
    ds4_metal_tensor *q_view = ds4_metal_tensor_view(qkv, 0, act_bytes);
    ds4_metal_tensor *k_view = ds4_metal_tensor_view(qkv, act_bytes, act_bytes);
    ds4_metal_tensor *v_view = ds4_metal_tensor_view(qkv, 2*act_bytes, act_bytes);

    if (!l26f_metal_gla(gla_out, state, q_view, k_view, v_view, gate,
                        1, 1, S, H, 1.0f / sqrtf((float)S))) {
        fprintf(stderr, "GLA failed\n"); return;
    }
    ds4_metal_tensor_free(v_view);
    ds4_metal_tensor_free(k_view);
    ds4_metal_tensor_free(q_view);

    // Step 5: output matvec (use first n_embd elements of gla_out as activation)
    printf("5. Output matvec...\n");
    ds4_metal_tensor *gla_act = ds4_metal_tensor_view(gla_out, 0, act_bytes);
    if (!l26f_metal_matvec_quant(proj, gla_act, m->map, m->size,
                                 wt_out->abs_offset,
                                 wt_out->dim[0], wt_out->dim[1], wt_out->type, 1)) {
        fprintf(stderr, "Output matvec failed\n"); return;
    }
    ds4_metal_tensor_free(gla_act);

    // Step 6: Residual add
    printf("6. Residual add...\n");
    if (!ds4_metal_add_tensor(out, inp, proj, n_embd)) {
        fprintf(stderr, "Add failed\n"); return;
    }

    // Read back
    float *result = (float *)malloc(act_bytes);
    ds4_metal_tensor_read(out, 0, result, act_bytes);

    float min_v = result[0], max_v = result[0], sum = 0;
    for (uint32_t i = 0; i < n_embd; i++) {
        float v = result[i];
        sum += v;
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
    }
    printf("\nResult: sum=%.6f min=%.6f max=%.6f avg=%.6f\n", sum, min_v, max_v, sum/n_embd);
    printf("First 10 values: ");
    for (int i = 0; i < 10; i++) printf("%.4f ", result[i]);
    printf("\n");

    // Verify state was updated
    float *state_data = (float *)malloc(state_bytes);
    ds4_metal_tensor_read(state, 0, state_data, state_bytes);
    float state_sum = 0;
    for (uint64_t i = 0; i < S*S*H; i++) state_sum += state_data[i];
    printf("State: sum=%.6f (should be non-zero after attention)\n", state_sum);
    free(state_data);
    free(result);

    ds4_metal_tensor_free(out);
    ds4_metal_tensor_free(proj);
    ds4_metal_tensor_free(state);
    ds4_metal_tensor_free(gla_out);
    ds4_metal_tensor_free(gate);
    ds4_metal_tensor_free(qkv);
    ds4_metal_tensor_free(normed);
    ds4_metal_tensor_free(inp);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]); return 1; }

    printf("Loading model...\n");
    l26f_model model;
    l26f_model_open(&model, argv[1]);

    printf("Metal init...\n");
    if (!ds4_metal_init()) { fprintf(stderr, "Metal init failed\n"); return 1; }

    printf("Model map...\n");
    if (!ds4_metal_set_model_map(model.map, model.size)) {
        fprintf(stderr, "Model map failed\n"); return 1;
    }

    test_gla_layer(&model);

    ds4_metal_cleanup();
    l26f_model_close(&model);
    return 0;
}
