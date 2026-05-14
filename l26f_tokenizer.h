#ifndef L26F_TOKENIZER_H
#define L26F_TOKENIZER_H

// Required includes (must appear before this header in the .c file):
//   #include <stdint.h>    // uint32_t, int32_t, uint64_t
//   #include <stdbool.h>   // bool

struct l26f_model;

typedef struct {
    char *text;
    uint32_t len;
} l26f_token_entry;

typedef struct {
    char *text;
    int32_t id;
} l26f_token_map_entry;

typedef struct {
    char *left_text;
    char *right_text;
    uint32_t rank;
} l26f_merge;

typedef struct {
    l26f_token_entry *tokens;
    uint32_t n_tokens;

    l26f_merge *merges;
    uint32_t n_merges;

    l26f_token_map_entry *token_map;
    uint32_t token_map_cap;

    int32_t bos_id;
    int32_t eos_id;
    int32_t unk_id;

    uint64_t tokens_data_pos;
    uint64_t merges_data_pos;
    bool loaded;
} l26f_tokenizer;

l26f_tokenizer *l26f_tokenizer_open(const char *path);
l26f_tokenizer *l26f_tokenizer_from_model(const struct l26f_model *m);
void l26f_tokenizer_close(l26f_tokenizer *t);

int l26f_token_decode(const l26f_tokenizer *t, int32_t token_id, char *buf, int buf_size);
int l26f_text_encode(const l26f_tokenizer *t, const char *text, int32_t *tokens, int max_tokens);
int l26f_text_decode(const l26f_tokenizer *t, const int32_t *tokens, int n_tokens, char *buf, int buf_size);

#endif
