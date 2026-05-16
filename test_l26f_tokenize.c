#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include "l26f.h"
#include "l26f_tokenizer.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    l26f_model model;
    l26f_model_open(&model, argv[1]);

    l26f_tokenizer *tok = l26f_tokenizer_from_model(&model);
    if (!tok) {
        fprintf(stderr, "failed to load tokenizer\n");
        return 1;
    }

    fprintf(stderr, "tokenizer: %u tokens, bos=%d, eos=%d\n",
            tok->n_tokens, tok->bos_id, tok->eos_id);

    // Find special tokens by searching for them
    const char *special[] = {
        "<role>", "</role>", "<|role_end|>",
        "<role>SYSTEM</role>", "<role>USER</role>", "<role>ASSISTANT</role>",
        "<|begin_of_text|>", "<|end_of_text|>",
        "SYSTEM", "USER", "ASSISTANT",
        NULL
    };
    fprintf(stderr, "\n=== Special token lookup ===\n");
    for (int i = 0; special[i]; i++) {
        int32_t id = l26f_token_lookup(tok, special[i], strlen(special[i]));
        fprintf(stderr, "  '%s' -> id=%d\n", special[i], id);
    }

    // Encode a sample prompt
    const char *prompt = "<role>SYSTEM</role>detailed thinking off<|role_end|>"
                         "<role>USER</role>Hello<|role_end|>"
                         "<role>ASSISTANT</role>";
    fprintf(stderr, "\n=== Encoding prompt ===\n");
    fprintf(stderr, "prompt: '%s'\n", prompt);

    int32_t tokens[4096];
    int n = l26f_text_encode(tok, prompt, tokens, 4096);
    fprintf(stderr, "encoded %d tokens:\n", n);
    for (int i = 0; i < n; i++) {
        char decoded[256];
        int dlen = l26f_token_decode(tok, tokens[i], decoded, sizeof(decoded));
        fprintf(stderr, "  [%d] id=%d decoded='%.*s' (len=%d)\n",
                i, tokens[i], dlen, decoded, dlen);
    }

    l26f_tokenizer_close(tok);
    l26f_model_close(&model);
    return 0;
}
