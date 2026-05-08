// l26f: test GGUF loader
#include "l26f.h"
#include <stdio.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    l26f_model m;
    l26f_model_open(&m, argv[1]);

    printf("\n=== Model Info ===\n");
    printf("layers: %u (MLA at: ", m.n_layer);
    bool first = true;
    for (uint32_t i = 0; i < m.n_layer && i < L26F_MAX_LAYERS; i++) {
        if (m.is_mla[i]) {
            if (!first) printf(", ");
            printf("%u", i);
            first = false;
        }
    }
    printf(")\n");
    printf("embd: %u, n_head: %u\n", m.n_embd, m.n_head);
    printf("vocab: %u, max_seq: %u\n", m.n_vocab, m.max_seq_len);
    printf("expert_count: %u, expert_used: %u\n", m.n_expert, m.n_expert_used);
    printf("q_lora_rank: %u, kv_lora_rank: %u, n_rot: %u\n",
           m.q_lora_rank, m.kv_lora_rank, m.n_rot);
    printf("nextn_predict_layers: %u\n", m.nextn_predict_layers);

    // Verify a few known tensors
    const char *check_tensors[] = {
        "token_embd.weight",
        "blk.0.attn_qkv.weight",
        "blk.7.attn_kv_a_mqa.weight",
        "blk.31.attn_q_a.weight",
        "output.weight",
        "blk.32.attn_norm.weight",
    };
    printf("\n=== Tensor Check ===\n");
    for (int i = 0; i < (int)(sizeof(check_tensors)/sizeof(check_tensors[0])); i++) {
        l26f_tensor *t = l26f_model_find_tensor(&m, check_tensors[i]);
        if (t) {
            printf("  %-40s type=%-8s shape=[", check_tensors[i], l26f_type_name(t->type));
            for (uint32_t j = 0; j < t->ndim; j++) {
                if (j > 0) printf(", ");
                printf("%" PRIu64, t->dim[j]);
            }
            printf("] bytes=%" PRIu64 "\n", t->bytes);
        } else {
            printf("  %-40s NOT FOUND\n", check_tensors[i]);
        }
    }

    l26f_model_close(&m);
    return 0;
}
