// l26f: Minimal BPE tokenizer for Bailing/Ling models
// Reads tokenizer metadata from GGUF, supports encode (text→tokens) and decode (tokens→text)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <ctype.h>
#include "l26f_tokenizer.h"

// ---- GPT-2 byte encoding table ----

static uint8_t gpt2_bytes[256];
static int gpt2_unicode_to_byte[256];

static void gpt2_init(void) {
    static bool done = false;
    if (done) return;
    done = true;
    int n = 0;
    for (int i = 33; i <= 126; i++) gpt2_bytes[n++] = (uint8_t)i;     // 94 chars
    for (int i = 161; i <= 172; i++) gpt2_bytes[n++] = (uint8_t)i;    // 12 chars
    for (int i = 174; i <= 255; i++) gpt2_bytes[n++] = (uint8_t)i;    // 82 chars
    // n = 94+12+82 = 188, remaining 68 are 0-32,127-160,173
    for (int i = 0; i < 256; i++) {
        if (i >= 33 && i <= 126) continue;
        if (i >= 161 && i <= 172) continue;
        if (i >= 174 && i <= 255) continue;
        gpt2_bytes[n++] = (uint8_t)i;
    }
    for (int i = 0; i < 256; i++) gpt2_unicode_to_byte[gpt2_bytes[i]] = i;
}

// ---- GGUF reader helpers ----

typedef struct {
    const uint8_t *data;
    uint64_t size;
    uint64_t pos;
} tk_reader;

static bool tk_read(tk_reader *r, void *dst, uint64_t n) {
    if (r->pos + n > r->size) return false;
    memcpy(dst, r->data + r->pos, n);
    r->pos += n;
    return true;
}

static bool tk_skip(tk_reader *r, uint64_t n) {
    if (r->pos + n > r->size) return false;
    r->pos += n;
    return true;
}

static bool tk_read_u32(tk_reader *r, uint32_t *v) { return tk_read(r, v, 4); }
static bool tk_read_u64(tk_reader *r, uint64_t *v) { return tk_read(r, v, 8); }

static bool tk_read_string(tk_reader *r, char **out, uint64_t *out_len) {
    uint64_t len;
    if (!tk_read_u64(r, &len)) return false;
    if (r->pos + len > r->size) return false;
    *out = (char *)(r->data + r->pos);
    *out_len = len;
    r->pos += len;
    return true;
}

static bool tk_skip_value(tk_reader *r, uint32_t type, int depth) {
    if (depth > 8) return false;
    if (type == 0 || type == 1) return tk_skip(r, 1);
    if (type == 2 || type == 3) return tk_skip(r, 2);
    if (type == 4 || type == 5 || type == 6 || type == 7) return tk_skip(r, 4);
    if (type == 8) { char *s; uint64_t l; return tk_read_string(r, &s, &l); }
    if (type == 10 || type == 11 || type == 12) return tk_skip(r, 8);
    if (type == 9) {
        uint32_t it; uint64_t al;
        if (!tk_read_u32(r, &it)) return false;
        if (!tk_read_u64(r, &al)) return false;
        uint64_t ss = 0;
        if (it == 0 || it == 1) ss = 1;
        else if (it == 2 || it == 3) ss = 2;
        else if (it >= 4 && it <= 7) ss = 4;
        else if (it >= 10 && it <= 12) ss = 8;
        if (ss) return tk_skip(r, al * ss);
        for (uint64_t i = 0; i < al; i++) if (!tk_skip_value(r, it, depth+1)) return false;
        return true;
    }
    return false;
}

// ---- Public API ----

l26f_tokenizer *l26f_tokenizer_open(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    struct stat st;
    fstat(fd, &st);
    const uint8_t *map = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) return NULL;

    l26f_tokenizer *t = (l26f_tokenizer *)calloc(1, sizeof(*t));
    t->fd = -1;

    tk_reader r = { .data = map, .size = st.st_size, .pos = 0 };

    uint32_t magic;
    tk_read_u32(&r, &magic);
    uint32_t version;
    tk_read_u32(&r, &version);
    uint64_t n_tensors, n_kv;
    tk_read_u64(&r, &n_tensors);
    tk_read_u64(&r, &n_kv);

    // First pass: count tokens and merges, find special token IDs
    uint64_t tokens_count = 0;
    uint64_t merges_count = 0;
    uint64_t tokens_start = 0;
    uint64_t merges_start = 0;
    uint32_t tokens_item_type = 0;

    for (uint64_t i = 0; i < n_kv; i++) {
        char *key; uint64_t klen;
        tk_read_string(&r, &key, &klen);
        uint32_t vtype;
        tk_read_u32(&r, &vtype);

        if (klen == 23 && memcmp(key, "tokenizer.ggml.tokens", 23) == 0 && vtype == 9) {
            uint32_t it; uint64_t al;
            tk_read_u32(&r, &it); tk_read_u64(&r, &al);
            tokens_item_type = it;
            tokens_count = al;
            tokens_start = r.pos;
            tk_skip_value(&r, 9, 0);
        } else if (klen == 23 && memcmp(key, "tokenizer.ggml.merges", 23) == 0 && vtype == 9) {
            uint32_t it; uint64_t al;
            tk_read_u32(&r, &it); tk_read_u64(&r, &al);
            merges_count = al;
            merges_start = r.pos;
            tk_skip_value(&r, 9, 0);
        } else if (klen == 25 && memcmp(key, "tokenizer.ggml.bos_token_id", 25) == 0 && vtype == 4) {
            tk_read_u32(&r, &t->bos_id);
        } else if (klen == 25 && memcmp(key, "tokenizer.ggml.eos_token_id", 25) == 0 && vtype == 4) {
            tk_read_u32(&r, &t->eos_id);
        } else if (klen == 32 && memcmp(key, "tokenizer.ggml.unknown_token_id", 32) == 0 && vtype == 4) {
            tk_read_u32(&r, &t->unk_id);
        } else {
            tk_skip_value(&r, vtype, 0);
        }
    }

    if (tokens_count == 0) {
        fprintf(stderr, "l26f_tokenizer: no tokens found in GGUF\n");
        munmap((void *)map, st.st_size);
        free(t);
        return NULL;
    }

    t->n_tokens = (uint32_t)tokens_count;
    t->n_merges = (uint32_t)merges_count;
    t->map = map;
    t->map_size = st.st_size;

    // Allocate and read tokens
    t->tokens = (l26f_token_entry *)calloc(tokens_count, sizeof(l26f_token_entry));
    {
        tk_reader tr = { .data = map, .size = st.st_size, .pos = tokens_start };
        for (uint64_t i = 0; i < tokens_count; i++) {
            char *s; uint64_t slen;
            tk_read_string(&tr, &s, &slen);
            t->tokens[i].text = (char *)malloc(slen + 1);
            memcpy(t->tokens[i].text, s, slen);
            t->tokens[i].text[slen] = 0;
            t->tokens[i].len = (uint32_t)slen;
        }
    }

    // Build token-to-id hash map
    t->token_map_cap = tokens_count * 2 + 1;
    t->token_map = (l26f_token_map_entry *)calloc(t->token_map_cap, sizeof(l26f_token_map_entry));
    for (uint32_t i = 0; i < tokens_count; i++) {
        uint32_t h = 5381;
        for (uint32_t j = 0; j < t->tokens[i].len; j++)
            h = ((h << 5) + h) + (uint32_t)(uint8_t)t->tokens[i].text[j];
        uint32_t idx = h % t->token_map_cap;
        while (t->token_map[idx].id >= 0) {
            idx = (idx + 1) % t->token_map_cap;
        }
        t->token_map[idx].text = t->tokens[i].text;
        t->token_map[idx].id = (int32_t)i;
    }

    // Read merges
    if (merges_count > 0 && merges_start > 0) {
        t->merges = (l26f_merge *)calloc(merges_count, sizeof(l26f_merge));
        tk_reader mr = { .data = map, .size = st.st_size, .pos = merges_start };
        for (uint64_t i = 0; i < merges_count; i++) {
            char *s; uint64_t slen;
            tk_read_string(&mr, &s, &slen);
            // Merge format: "left right" — find the space
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
            t->merges[i].rank = (uint32_t)i;
        }
    }

    gpt2_init();
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
    if (t->map) munmap((void *)t->map, t->map_size);
    free(t);
}

// ---- Decode: token ID → text ----

int l26f_token_decode(const l26f_tokenizer *t, int32_t token_id, char *buf, int buf_size) {
    if (token_id < 0 || token_id >= (int32_t)t->n_tokens) {
        buf[0] = 0;
        return 0;
    }
    const char *raw = t->tokens[token_id].text;
    uint32_t len = t->tokens[token_id].len;

    // Check if it's a byte token like <0x0A>
    if (len == 6 && raw[0] == '<' && raw[1] == '0' && raw[2] == 'x' && raw[5] == '>') {
        char hex[3] = { raw[3], raw[4], 0 };
        uint8_t byte = (uint8_t)strtol(hex, NULL, 16);
        if (buf_size >= 2) { buf[0] = (char)byte; buf[1] = 0; }
        return 1;
    }

    // Special tokens (BOS, EOS, etc.) — output as-is
    if (raw[0] == '<' && len > 1) {
        int n = len < buf_size ? (int)len : buf_size - 1;
        memcpy(buf, raw, n);
        buf[n] = 0;
        return n;
    }

    // Normal BPE token: convert from GPT-2 byte encoding
    int out = 0;
    for (uint32_t i = 0; i < len && out < buf_size - 1; i++) {
        uint8_t b = (uint8_t)raw[i];
        if (b < 128) {
            buf[out++] = (char)b;
        } else {
            // GPT-2 maps bytes 128-255 to unicode chars via the bytes_to_unicode table
            // For decode, we need the inverse: find the byte value
            buf[out++] = (char)gpt2_unicode_to_byte[b % 256];
        }
    }
    buf[out] = 0;
    return out;
}

// ---- Encode: text → token IDs (simple, no BPE merges) ----
// For now, just do character-level encoding with GPT-2 byte encoding.
// This won't produce optimal tokenization but will work for basic text.

static int32_t l26f_token_lookup(const l26f_tokenizer *t, const char *text, uint32_t len) {
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
        // Try longest match first (greedy)
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
            // Encode as byte token
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
