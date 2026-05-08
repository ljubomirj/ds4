// l26f: Metal pipeline test — init + model map + RMS norm
#include "l26f.h"
#include "l26f_metal.h"
#include <stdio.h>
#include <stdlib.h>

static void metal_test(const l26f_model *model) {
    printf("\n=== Metal Pipeline Test ===\n");

    if (!ds4_metal_init()) {
        fprintf(stderr, "Metal init failed\n");
        return;
    }

    // Set up model map (zero-copy)
    ds4_metal_set_model_map(model->map, model->size);
    printf("Metal: model map set (%.2f GiB)\n", (double)model->size / (1<<30));

    // Allocate tensors for a single-token test
    const uint32_t n_embd = model->n_embd;
    const uint64_t act_bytes = n_embd * sizeof(float);

    ds4_metal_tensor *inp  = ds4_metal_tensor_alloc(act_bytes);
    ds4_metal_tensor *norm = ds4_metal_tensor_alloc(act_bytes);

    // Fill input with random-ish data (non-zero)
    float *inp_data = (float *)malloc(act_bytes);
    for (uint32_t i = 0; i < n_embd; i++) inp_data[i] = (float)(i % 1000) / 1000.0f;
    ds4_metal_tensor_write(inp, 0, inp_data, act_bytes);
    free(inp_data);
    printf("Input tensor: %lu bytes filled\n", act_bytes);

    // Find the first GLA layer's attn_norm weight
    l26f_tensor *wt_norm = l26f_model_find_tensor(model, "blk.0.attn_norm.weight");
    if (!wt_norm) {
        fprintf(stderr, "blk.0.attn_norm.weight not found\n");
        return;
    }
    printf("attn_norm at offset %llu, type=%s\n",
           (unsigned long long)wt_norm->abs_offset, l26f_type_name(wt_norm->type));

    // RMS norm: out = norm_weight * (inp / rms(inp))
    if (!ds4_metal_rms_norm_weight_tensor(norm, inp, model->map, model->size,
                                          wt_norm->abs_offset - model->tensor_data_pos,
                                          n_embd, 1e-6f)) {
        fprintf(stderr, "RMS norm failed\n");
        return;
    }
    printf("RMS norm dispatched\n");

    ds4_metal_synchronize();
    printf("Metal sync done\n");

    // Read back output and verify it's non-zero
    float *norm_data = (float *)malloc(act_bytes);
    ds4_metal_tensor_read(norm, 0, norm_data, act_bytes);
    float sum = 0;
    for (uint32_t i = 0; i < n_embd; i++) sum += norm_data[i];
    printf("RMS norm output sum: %f\n", sum);
    free(norm_data);

    ds4_metal_tensor_free(norm);
    ds4_metal_tensor_free(inp);
    ds4_metal_cleanup();
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    l26f_model model;
    l26f_model_open(&model, argv[1]);
    metal_test(&model);
    l26f_model_close(&model);
    return 0;
}
