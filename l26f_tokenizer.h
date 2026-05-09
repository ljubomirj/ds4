#ifndef L26F_TOKENIZER_H
#define L26F_TOKENIZER_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
    char *text;
    uint32_t len;
} l26f_token_entry;

typedef struct {
    char *text;
    int32_t id;
} l26f_token_map_entry;

typedef struct {
    char *left_text;    // points into allocated string (owns the allocation)
    char *right_text;   // points into same string after the space
    uint32_t rank;
} l26f_merge;

typedef struct {
    int fd;
    const uint8_t *map;
    uint64_t map_size;

    l26f_token_entry *tokens;
    uint32_t n_tokens;

    l26f_merge *merges;
    uint32_t n_merges;

    l26f_token_map_entry *token_map;
    uint32_t token_map_cap;

    int32_t bos_id;
    int32_t eos_id;
    int32_t unk_id;
} l26f_tokenizer;

l26f_tokenizer *l26f_tokenizer_open(const char *path);
void l26f_tokenizer_close(l26f_tokenizer *t);

int l26f_token_decode(const l26f_tokenizer *t, int32_t token_id, char *buf, int buf_size);
int l26f_text_encode(const l26f_tokenizer *t, const char *text, int32_t *tokens, int max_tokens);
int l26f_text_decode(const l26f_tokenizer *t, const int32_t *tokens, int n_tokens, char *buf, int buf_size);

#endif
