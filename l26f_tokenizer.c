#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include "l26f_tokenizer.h"
#include "l26f.h"

#ifndef NDEBUG
#define XLOG(fmt, ...) fprintf(stderr, fmt "\n", ##__VA_ARGS__)
#else
#define XLOG(fmt, ...) ((void)0)
#endif

#define XLOG_EVERY(n, i, total, fmt, ...) \
    do { if ((i) % (n) == 0) XLOG(fmt, ##__VA_ARGS__); } while(0)

static uint8_t gpt2_bytes[256];
static int gpt2_unicode_to_byte[256];

static void gpt2_init(void) {
    static bool done = false;
    if (done) return;
    done = true;
    int n = 0;
    for (int i = 33; i <= 126; i++) gpt2_bytes[n++] = (uint8_t)i;
    for (int i = 161; i <= 172; i++) gpt2_bytes[n++] = (uint8_t)i;
    for (int i = 174; i <= 255; i++) gpt2_bytes[n++] = (uint8_t)i;
    for (int i = 0; i < 256; i++) {
        if (i >= 33 && i <= 126) continue;
        if (i >= 161 && i <= 172) continue;
        if (i >= 174 && i <= 255) continue;
        gpt2_bytes[n++] = (uint8_t)i;
    }
    for (int i = 0; i < 256; i++) gpt2_unicode_to_byte[gpt2_bytes[i]] = i;
}

typedef struct {
    const uint8_t *data;
    uint64_t size;
    uint64_t pos;
} tk_reader;

static bool tk_read_u64(tk_reader *r, uint64_t *v) {
    if (r->pos + 8 > r->size) return false;
    memcpy(v, r->data + r->pos, 8);
    r->pos += 8;
    return true;
}

static bool tk_read_string(tk_reader *r, const char **out, uint64_t *out_len) {
    uint64_t len;
    if (!tk_read_u64(r, &len)) return false;
    if (r->pos + len > r->size) return false;
    *out = (const char *)(r->data + r->pos);
    *out_len = len;
    r->pos += len;
    return true;
}

l26f_tokenizer *l26f_tokenizer_open(const char *path) {
    (void)path;
    return NULL;
}

l26f_tokenizer *l26f_tokenizer_from_model(const l26f_model *m) {
    XLOG("TOK: entering, tok_found=%d count=%llu pos=%llu",
            m->tok_found, (unsigned long long)m->tok_tokens_count,
            (unsigned long long)m->tok_tokens_pos);
    if (!m->tok_found) {
        fprintf(stderr, "l26f_tokenizer: no tokenizer data in GGUF\n");
        return NULL;
    }

    l26f_tokenizer *t = (l26f_tokenizer *)calloc(1, sizeof(*t));
    XLOG("TOK: calloc t done");
    t->bos_id = m->tok_bos_id;
    t->eos_id = m->tok_eos_id;

    tk_reader tr = { .data = m->map, .size = m->size, .pos = m->tok_tokens_pos };
    t->n_tokens = (uint32_t)m->tok_tokens_count;
    XLOG("TOK: allocating %u token entries (%llu bytes)",
            t->n_tokens, (unsigned long long)(t->n_tokens * sizeof(l26f_token_entry)));
    t->tokens = (l26f_token_entry *)calloc(t->n_tokens, sizeof(l26f_token_entry));
    XLOG("TOK: calloc tokens done, reading...");
    for (uint32_t i = 0; i < t->n_tokens; i++) {
        if (i % 10000 == 0) XLOG("TOK: reading token %u / %u", i, t->n_tokens);
        const char *s; uint64_t slen;
        if (!tk_read_string(&tr, &s, &slen)) {
            fprintf(stderr, "l26f_tokenizer: truncated token table at token %u", i);
            l26f_tokenizer_close(t);
            return NULL;
        }
        t->tokens[i].text = (char *)malloc(slen + 1);
        memcpy(t->tokens[i].text, s, slen);
        t->tokens[i].text[slen] = 0;
        t->tokens[i].len = (uint32_t)slen;
    }

    XLOG("TOK: %u tokens read, hash map deferred (skip 5MB calloc)", t->n_tokens);
    t->token_map_cap = 0;
    t->token_map = NULL;

    XLOG("TOK: hash map skipped");
    if (m->tok_merges_count > 0) {
        XLOG("TOK: loading %llu merges...",
                (unsigned long long)m->tok_merges_count);
        t->n_merges = (uint32_t)m->tok_merges_count;
        t->merges = (l26f_merge *)calloc(t->n_merges, sizeof(l26f_merge));
        tk_reader mr = { .data = m->map, .size = m->size, .pos = m->tok_merges_pos };
        for (uint32_t i = 0; i < t->n_merges; i++) {
            const char *s; uint64_t slen;
            if (!tk_read_string(&mr, &s, &slen)) break;
            t->merges[i].left_text = (char *)malloc(slen + 1);
            memcpy(t->merges[i].left_text, s, slen);
            t->merges[i].left_text[slen] = 0;
            char *sp = strchr(t->merges[i].left_text, ' ');
            if (sp) {
                *sp = 0;
                t->merges[i].right_text = sp + 1;
            } else {
                t->merges[i].right_text = t->merges[i].left_text + slen;
            }
            t->merges[i].rank = i;
        }
    }
    XLOG("TOK: merges done");

    gpt2_init();
    t->loaded = true;
    XLOG("TOK: returning t=%p", (void *)t);
    return t;
}

void l26f_tokenizer_close(l26f_tokenizer *t) {
    if (!t) return;
    for (uint32_t i = 0; i < t->n_tokens; i++) free(t->tokens[i].text);
    free(t->tokens);
    if (t->merges) {
        for (uint32_t i = 0; i < t->n_merges; i++) free(t->merges[i].left_text);
        free(t->merges);
    }
    free(t->token_map);
    free(t);
}

int l26f_token_decode(const l26f_tokenizer *t, int32_t token_id, char *buf, int buf_size) {
    if (token_id < 0 || token_id >= (int32_t)t->n_tokens) {
        buf[0] = 0;
        return 0;
    }
    const char *raw = t->tokens[token_id].text;
    uint32_t len = t->tokens[token_id].len;

    if (len == 6 && raw[0] == '<' && raw[1] == '0' && raw[2] == 'x' && raw[5] == '>') {
        char hex[3] = { raw[3], raw[4], 0 };
        uint8_t byte = (uint8_t)strtol(hex, NULL, 16);
        if (buf_size >= 2) { buf[0] = (char)byte; buf[1] = 0; }
        return 1;
    }

    if (raw[0] == '<' && len > 1) {
        int n = len < buf_size ? (int)len : buf_size - 1;
        memcpy(buf, raw, n);
        buf[n] = 0;
        return n;
    }

    int out = 0;
    for (uint32_t i = 0; i < len && out < buf_size - 1; i++) {
        uint8_t b = (uint8_t)raw[i];
        if (b < 128) {
            buf[out++] = (char)b;
        } else {
            buf[out++] = (char)gpt2_unicode_to_byte[b % 256];
        }
    }
    buf[out] = 0;
    return out;
}

static int32_t l26f_token_lookup(const l26f_tokenizer *t, const char *text, uint32_t len) {
    if (!t->token_map) return -1;  // hash map not built yet
    uint32_t h = 5381;
    for (uint32_t j = 0; j < len; j++)
        h = ((h << 5) + h) + (uint32_t)(uint8_t)text[j];
    uint32_t idx = h % t->token_map_cap;
    while (t->token_map[idx].id >= 0) {
        if (strlen(t->token_map[idx].text) == len &&
            memcmp(t->token_map[idx].text, text, len) == 0) {
            return t->token_map[idx].id;
        }
        idx = (idx + 1) % t->token_map_cap;
    }
    return -1;
}

int l26f_text_encode(const l26f_tokenizer *t, const char *text, int32_t *tokens, int max_tokens) {
    int n_tokens = 0;
    int text_len = (int)strlen(text);
    int pos = 0;

    while (pos < text_len && n_tokens < max_tokens) {
        int best_len = 0;
        int32_t best_id = -1;
        for (int len = text_len - pos; len >= 1; len--) {
            int32_t id = l26f_token_lookup(t, text + pos, len);
            if (id >= 0) { best_len = len; best_id = id; break; }
        }
        if (best_len > 0) {
            tokens[n_tokens++] = best_id;
            pos += best_len;
        } else {
            uint8_t byte = (uint8_t)text[pos];
            char byte_tok[8];
            snprintf(byte_tok, sizeof(byte_tok), "<0x%02X>", byte);
            int32_t id = l26f_token_lookup(t, byte_tok, 6);
            if (id >= 0) {
                tokens[n_tokens++] = id;
            }
            pos++;
        }
    }
    return n_tokens;
}

int l26f_text_decode(const l26f_tokenizer *t, const int32_t *tokens, int n_tokens, char *buf, int buf_size) {
    int pos = 0;
    for (int i = 0; i < n_tokens && pos < buf_size - 1; i++) {
        pos += l26f_token_decode(t, tokens[i], buf + pos, buf_size - pos);
    }
    buf[pos < buf_size ? pos : buf_size - 1] = 0;
    return pos;
}
