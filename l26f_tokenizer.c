#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <limits.h>
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
static int gpt2_unicode_to_byte[768];

static void gpt2_init(void) {
    static bool done = false;
    if (done) return;
    done = true;
    int n = 0;
    for (int i = 33; i <= 126; i++) gpt2_bytes[n++] = (uint8_t)i;
    for (int i = 161; i <= 172; i++) gpt2_bytes[n++] = (uint8_t)i;
    for (int i = 174; i <= 255; i++) gpt2_bytes[n++] = (uint8_t)i;
    int n_direct = n;
    for (int i = 0; i < 256; i++) {
        if (i >= 33 && i <= 126) continue;
        if (i >= 161 && i <= 172) continue;
        if (i >= 174 && i <= 255) continue;
        gpt2_bytes[n++] = (uint8_t)i;
    }
    memset(gpt2_unicode_to_byte, -1, sizeof(gpt2_unicode_to_byte));
    for (int i = 0; i < 256; i++) {
        int cp = (i < n_direct) ? (int)gpt2_bytes[i] : (256 + (i - n_direct));
        if (cp < 768) gpt2_unicode_to_byte[cp] = (int)gpt2_bytes[i];
    }
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

    XLOG("TOK: %u tokens read, building hash map (cap=%u)...",
            t->n_tokens, t->n_tokens * 2 + 1);
    t->token_map_cap = t->n_tokens * 2 + 1;
    t->token_map = (l26f_token_map_entry *)malloc(t->token_map_cap * sizeof(l26f_token_map_entry));
    XLOG("TOK: malloc done, memset...");
    memset(t->token_map, 0, t->token_map_cap * sizeof(l26f_token_map_entry));
    XLOG("TOK: memset done, inserting %u entries...", t->n_tokens);
    for (uint32_t i = 0; i < t->n_tokens; i++) {
        if (i % 10000 == 0) XLOG("TOK: hash insert %u / %u", i, t->n_tokens);
        uint32_t h = 5381;
        for (uint32_t j = 0; j < t->tokens[i].len; j++)
            h = ((h << 5) + h) + (uint32_t)(uint8_t)t->tokens[i].text[j];
        uint32_t idx = h % t->token_map_cap;
        while (t->token_map[idx].text != NULL)
            idx = (idx + 1) % t->token_map_cap;
        t->token_map[idx].text = t->tokens[i].text;
        t->token_map[idx].id = (int32_t)i;
    }

    XLOG("TOK: hash map built");
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
    for (uint32_t i = 0; i < len && out < buf_size - 1; ) {
        unsigned char c = (unsigned char)raw[i];
        int cp;
        if (c < 0x80) {
            cp = c;
            i += 1;
        } else if (c >= 0xC2 && c <= 0xDF && i + 1 < len) {
            cp = ((c & 0x1F) << 6) | ((unsigned char)raw[i+1] & 0x3F);
            i += 2;
        } else if (c >= 0xE0 && c <= 0xEF && i + 2 < len) {
            cp = ((c & 0x0F) << 12) | (((unsigned char)raw[i+1] & 0x3F) << 6)
                | ((unsigned char)raw[i+2] & 0x3F);
            i += 3;
        } else if (c >= 0xF0 && c <= 0xF7 && i + 3 < len) {
            cp = ((c & 0x07) << 18) | (((unsigned char)raw[i+1] & 0x3F) << 12)
                | (((unsigned char)raw[i+2] & 0x3F) << 6) | ((unsigned char)raw[i+3] & 0x3F);
            i += 4;
        } else {
            i += 1;
            continue;
        }
        int byte_val = (cp >= 0 && cp < 768) ? gpt2_unicode_to_byte[cp] : -1;
        if (byte_val >= 0 && byte_val < 256) {
            buf[out++] = (char)(uint8_t)byte_val;
        } else if (cp >= 0 && cp < 128) {
            buf[out++] = (char)cp;
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
    while (t->token_map[idx].text != NULL) {
        if (strlen(t->token_map[idx].text) == len &&
            memcmp(t->token_map[idx].text, text, len) == 0) {
            return t->token_map[idx].id;
        }
        idx = (idx + 1) % t->token_map_cap;
    }
    return -1;
}

static int gpt2_byte_to_cp[256];
static int gpt2_maps_init = 0;

static void gpt2_encode_text_init(void) {
    if (gpt2_maps_init) return;
    gpt2_maps_init = 1;
    memset(gpt2_byte_to_cp, 0, sizeof(gpt2_byte_to_cp));
    for (int i = 33; i <= 126; i++) gpt2_byte_to_cp[i] = i;
    for (int i = 161; i <= 172; i++) gpt2_byte_to_cp[i] = i;
    for (int i = 174; i <= 255; i++) gpt2_byte_to_cp[i] = i;
    int cp = 256;
    for (int i = 0; i < 256; i++) {
        if (i >= 33 && i <= 126) continue;
        if (i >= 161 && i <= 172) continue;
        if (i >= 174 && i <= 255) continue;
        gpt2_byte_to_cp[i] = cp++;
    }
}

static int gpt2_encode_text(const char *input, int input_len, char *out, int out_cap) {
    int pos = 0;
    for (int i = 0; i < input_len && pos < out_cap - 4; i++) {
        int cp = gpt2_byte_to_cp[(uint8_t)input[i]];
        if (cp < 0x80) {
            out[pos++] = (char)cp;
        } else if (cp < 0x800) {
            out[pos++] = (char)(0xC0 | (cp >> 6));
            out[pos++] = (char)(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out[pos++] = (char)(0xE0 | (cp >> 12));
            out[pos++] = (char)(0x80 | ((cp >> 6) & 0x3F));
            out[pos++] = (char)(0x80 | (cp & 0x3F));
        } else {
            out[pos++] = (char)(0xF0 | (cp >> 18));
            out[pos++] = (char)(0x80 | ((cp >> 12) & 0x3F));
            out[pos++] = (char)(0x80 | ((cp >> 6) & 0x3F));
            out[pos++] = (char)(0x80 | (cp & 0x3F));
        }
    }
    out[pos] = 0;
    return pos;
}

#define L26F_BPE_MAX_PIECES 4096

typedef struct {
    int32_t id;
    int start;
    int len;
} l26f_bpe_piece;

static int l26f_bpe_find_merge(const l26f_tokenizer *t,
        const char *gpt2, const l26f_bpe_piece *pieces, int n_pieces,
        int pi) {
    if (pi + 1 >= n_pieces) return -1;
    const char *left = gpt2 + pieces[pi].start;
    int left_len = pieces[pi].len;
    const char *right = gpt2 + pieces[pi + 1].start;
    int right_len = pieces[pi + 1].len;
    for (uint32_t m = 0; m < t->n_merges; m++) {
        int ml = (int)strlen(t->merges[m].left_text);
        int mr = (int)strlen(t->merges[m].right_text);
        if (ml == left_len && mr == right_len
            && memcmp(t->merges[m].left_text, left, ml) == 0
            && memcmp(t->merges[m].right_text, right, mr) == 0) {
            return (int)m;
        }
    }
    return -1;
}

static int l26f_bpe_apply(const l26f_tokenizer *t,
        const char *gpt2, int frag_start, int frag_len,
        int32_t *tokens, int max_tokens) {
    if (frag_len <= 0) return 0;

    l26f_bpe_piece pieces[L26F_BPE_MAX_PIECES];
    int n_pieces = 0;
    int p = frag_start;
    while (p < frag_start + frag_len && n_pieces < L26F_BPE_MAX_PIECES) {
        uint8_t c = (uint8_t)gpt2[p];
        int clen = 1;
        if ((c & 0xE0) == 0xC0) clen = 2;
        else if ((c & 0xF0) == 0xE0) clen = 3;
        else if ((c & 0xF8) == 0xF0) clen = 4;
        if (p + clen > frag_start + frag_len) clen = 1;
        int32_t id = l26f_token_lookup(t, gpt2 + p, clen);
        if (id < 0) { p += clen; continue; }
        pieces[n_pieces].id = id;
        pieces[n_pieces].start = p;
        pieces[n_pieces].len = clen;
        n_pieces++;
        p += clen;
    }

    for (;;) {
        int best_merge = -1;
        int best_rank = INT32_MAX;
        int best_idx = -1;
        for (int i = 0; i < n_pieces - 1; i++) {
            int m = l26f_bpe_find_merge(t, gpt2, pieces, n_pieces, i);
            if (m >= 0 && t->merges[m].rank < best_rank) {
                best_rank = t->merges[m].rank;
                best_merge = m;
                best_idx = i;
            }
        }
        if (best_merge < 0) break;

        int merged_len = pieces[best_idx].len + pieces[best_idx + 1].len;
        int32_t merged_id = l26f_token_lookup(t, gpt2 + pieces[best_idx].start, merged_len);
        if (merged_id < 0) break;

        pieces[best_idx].id = merged_id;
        pieces[best_idx].len = merged_len;
        for (int i = best_idx + 1; i < n_pieces - 1; i++) {
            pieces[i] = pieces[i + 1];
        }
        n_pieces--;
    }

    int n_tokens = 0;
    for (int i = 0; i < n_pieces && n_tokens < max_tokens; i++) {
        tokens[n_tokens++] = pieces[i].id;
    }
    return n_tokens;
}

int l26f_text_encode(const l26f_tokenizer *t, const char *text, int32_t *tokens, int max_tokens) {
    gpt2_encode_text_init();

    static const struct { const char *text; int len; } specials[] = {
        {"<|role_end|>", 12},
        {"</role>", 7},
        {"<role>", 6},
    };
    static int32_t special_ids[3] = {0, 0, 0};
    static int ids_init = 0;
    if (!ids_init) {
        for (int i = 0; i < 3; i++)
            special_ids[i] = l26f_token_lookup(t, specials[i].text, specials[i].len);
        ids_init = 1;
    }

    int text_len = (int)strlen(text);
    char *gpt2 = (char *)malloc(text_len * 4 + 1);
    int gpt2_len = gpt2_encode_text(text, text_len, gpt2, text_len * 4 + 1);

    int n_tokens = 0;
    int pos = 0;

    while (pos < gpt2_len && n_tokens < max_tokens) {
        int sp_match = -1;
        for (int i = 0; i < 3; i++) {
            if (special_ids[i] >= 0 && pos + specials[i].len <= gpt2_len
                && memcmp(gpt2 + pos, specials[i].text, specials[i].len) == 0) {
                sp_match = i;
                break;
            }
        }
        if (sp_match >= 0) {
            tokens[n_tokens++] = special_ids[sp_match];
            pos += specials[sp_match].len;
            continue;
        }

        int end = gpt2_len;
        for (int i = 0; i < 3; i++) {
            if (special_ids[i] < 0) continue;
            const char *found = memmem(gpt2 + pos, end - pos, specials[i].text, specials[i].len);
            if (found) {
                int off = (int)(found - gpt2);
                if (off < end) end = off;
            }
        }

        int frag_len = end - pos;
        int n = l26f_bpe_apply(t, gpt2, pos, frag_len, tokens + n_tokens, max_tokens - n_tokens);
        n_tokens += n;
        pos += frag_len;
    }
    free(gpt2);
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
