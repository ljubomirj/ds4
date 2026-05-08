// l26f: inference driver — minimal test
#include "l26f.h"
#include <stdio.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    l26f_model m;
    l26f_model_open(&m, argv[1]);

    // Print tensor list summary
    int n_f32 = 0, n_q5k = 0, n_q6k = 0, n_iq4nl = 0, n_q8_0 = 0, n_other = 0;
    for (uint64_t i = 0; i < m.n_tensors; i++) {
        switch (m.tensors[i].type) {
        case 0:  n_f32++; break;
        case 13: n_q5k++; break;
        case 14: n_q6k++; break;
        case 20: n_iq4nl++; break;
        case 8:  n_q8_0++; break;
        default: n_other++; break;
        }
    }
    printf("\nTensor types: F32=%d, Q5_K=%d, Q6_K=%d, IQ4_NL=%d, Q8_0=%d, other=%d\n",
           n_f32, n_q5k, n_q6k, n_iq4nl, n_q8_0, n_other);

    // Verify key layer tensors
    printf("\n--- Layer 0 (GLA) ---\n");
    const char *gla_tensors[] = {
        "blk.0.attn_norm.weight",
        "blk.0.attn_qkv.weight",
        "blk.0.attn_q_norm.weight",
        "blk.0.attn_k_norm.weight",
        "blk.0.attn_gate.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.ffn_gate_inp.weight",
        "blk.0.exp_probs_b.bias",
        "blk.0.ffn_gate_shexp.weight",
        "blk.0.ffn_up_shexp.weight",
        "blk.0.ffn_down_shexp.weight",
        "blk.0.layer_output_norm.weight",
    };
    for (int i = 0; i < (int)(sizeof(gla_tensors)/sizeof(gla_tensors[0])); i++) {
        l26f_tensor *t = l26f_model_find_tensor(&m, gla_tensors[i]);
        if (t) {
            printf("  %-40s type=%-6s dims=[", gla_tensors[i], l26f_type_name(t->type));
            for (uint32_t j = 0; j < t->ndim; j++) {
                if (j) printf(", ");
                printf("%" PRIu64, t->dim[j]);
            }
            printf("]\n");
        }
    }

    printf("\n--- Layer 7 (MLA) ---\n");
    const char *mla_tensors[] = {
        "blk.7.attn_norm.weight",
        "blk.7.attn_q_a.weight", "blk.7.attn_q_a_norm.weight", "blk.7.attn_q_b.weight",
        "blk.7.attn_kv_a_mqa.weight", "blk.7.attn_kv_a_norm.weight",
        "blk.7.attn_k_b.weight", "blk.7.attn_v_b.weight",
        "blk.7.attn_output.weight",
    };
    for (int i = 0; i < (int)(sizeof(mla_tensors)/sizeof(mla_tensors[0])); i++) {
        l26f_tensor *t = l26f_model_find_tensor(&m, mla_tensors[i]);
        if (t) {
            printf("  %-40s type=%-6s dims=[", mla_tensors[i], l26f_type_name(t->type));
            for (uint32_t j = 0; j < t->ndim; j++) {
                if (j) printf(", ");
                printf("%" PRIu64, t->dim[j]);
            }
            printf("]\n");
        }
    }

    l26f_model_close(&m);
    return 0;
}
