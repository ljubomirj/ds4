// l26f: GGUF model loader
// Adapted from ds4.c's GGUF parser

#include "l26f.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

// ---- Cursor operations ----

typedef struct {
    const l26f_model *model;
    const uint8_t *ptr;
    uint64_t remaining;
    const char *error;
} l26f_cursor;

static l26f_cursor l26f_cursor_at(const l26f_model *m, uint64_t offset) {
    return (l26f_cursor){
        .model = m,
        .ptr = m->map + offset,
        .remaining = m->size > offset ? m->size - offset : 0,
    };
}

static void l26f_cursor_error(l26f_cursor *c, const char *msg) {
    if (!c->error) c->error = msg;
}

static bool l26f_cursor_has(l26f_cursor *c, uint64_t n) {
    if (c->remaining < n) { l26f_cursor_error(c, "truncated GGUF"); return false; }
    return true;
}

static bool l26f_cursor_read(l26f_cursor *c, void *dst, uint64_t n) {
    if (!l26f_cursor_has(c, n)) return false;
    memcpy(dst, c->ptr, n);
    c->ptr += n;
    c->remaining -= n;
    return true;
}

static bool l26f_cursor_skip(l26f_cursor *c, uint64_t n) {
    if (!l26f_cursor_has(c, n)) return false;
    c->ptr += n;
    c->remaining -= n;
    return true;
}

static bool l26f_cursor_u32(l26f_cursor *c, uint32_t *v) { return l26f_cursor_read(c, v, 4); }
static bool l26f_cursor_u64(l26f_cursor *c, uint64_t *v) { return l26f_cursor_read(c, v, 8); }
static bool l26f_cursor_f32(l26f_cursor *c, float *v)    { return l26f_cursor_read(c, v, 4); }

static bool l26f_cursor_string(l26f_cursor *c, l26f_str *s) {
    uint64_t len;
    if (!l26f_cursor_u64(c, &len)) return false;
    if (!l26f_cursor_has(c, len)) return false;
    s->ptr = (const char *)c->ptr;
    s->len = len;
    return l26f_cursor_skip(c, len);
}

static uint64_t l26f_scalar_size(uint32_t type) {
    switch (type) {
    case L26F_GGUF_UINT8:  case L26F_GGUF_INT8:  case L26F_GGUF_BOOL:    return 1;
    case L26F_GGUF_UINT16: case L26F_GGUF_INT16:                          return 2;
    case L26F_GGUF_UINT32: case L26F_GGUF_INT32: case L26F_GGUF_FLOAT32: return 4;
    case L26F_GGUF_UINT64: case L26F_GGUF_INT64: case L26F_GGUF_FLOAT64: return 8;
    default: return 0;
    }
}

static bool l26f_skip_value(l26f_cursor *c, uint32_t type, int depth) {
    if (depth > 8) { l26f_cursor_error(c, "metadata too deep"); return false; }
    uint64_t scalar = l26f_scalar_size(type);
    if (scalar != 0) return l26f_cursor_skip(c, scalar);
    if (type == L26F_GGUF_STRING) { l26f_str s; return l26f_cursor_string(c, &s); }
    if (type == L26F_GGUF_ARRAY) {
        uint32_t item_type; uint64_t len;
        if (!l26f_cursor_u32(c, &item_type)) return false;
        if (!l26f_cursor_u64(c, &len)) return false;
        uint64_t item_size = l26f_scalar_size(item_type);
        if (item_size != 0) return l26f_cursor_skip(c, len * item_size);
        for (uint64_t i = 0; i < len; i++)
            if (!l26f_skip_value(c, item_type, depth + 1)) return false;
        return true;
    }
    l26f_cursor_error(c, "unknown metadata type");
    return false;
}

static void l26f_parse_metadata(l26f_model *m, l26f_cursor *c) {
    for (uint64_t i = 0; i < m->n_kv; i++) {
        l26f_str key;
        uint32_t type;
        if (!l26f_cursor_string(c, &key)) return;
        if (!l26f_cursor_u32(c, &type)) return;

        // Extract key model parameters
        if (l26f_str_eq(key, "bailing_hybrid.block_count")) {
            uint32_t v = 0; l26f_cursor_u32(c, &v); m->n_layer = v;
        } else if (l26f_str_eq(key, "bailing_hybrid.embedding_length")) {
            uint32_t v = 0; l26f_cursor_u32(c, &v); m->n_embd = v;
        } else if (l26f_str_eq(key, "bailing_hybrid.attention.head_count")) {
            uint32_t v = 0; l26f_cursor_u32(c, &v); m->n_head = v;
        } else if (l26f_str_eq(key, "bailing_hybrid.attention.head_count_kv")) {
            // Array of per-layer KV head counts
            uint32_t item_type; uint64_t len;
            if (!l26f_cursor_u32(c, &item_type)) return;
            if (!l26f_cursor_u64(c, &len)) return;
            for (uint64_t j = 0; j < len && j < L26F_MAX_LAYERS; j++) {
                int32_t hkv = 0;
                if (!l26f_cursor_read(c, &hkv, 4)) return;
                m->is_mla[j] = (hkv > 0); // MLA layers have head_count_kv > 0
            }
        } else if (l26f_str_eq(key, "bailing_hybrid.feed_forward_length")) {
            uint32_t v = 0; l26f_cursor_u32(c, &v); m->n_ff = v;
        } else if (l26f_str_eq(key, "bailing_hybrid.expert_count")) {
            uint32_t v = 0; l26f_cursor_u32(c, &v); m->n_expert = v;
        } else if (l26f_str_eq(key, "bailing_hybrid.expert_used_count")) {
            uint32_t v = 0; l26f_cursor_u32(c, &v); m->n_expert_used = v;
        } else if (l26f_str_eq(key, "bailing_hybrid.expert_group_count")) {
            uint32_t v = 0; l26f_cursor_u32(c, &v); m->n_expert_groups = v;
        } else if (l26f_str_eq(key, "bailing_hybrid.expert_group_used_count")) {
            uint32_t v = 0; l26f_cursor_u32(c, &v); m->n_group_used = v;
        } else if (l26f_str_eq(key, "bailing_hybrid.vocab_size")) {
            uint32_t v = 0; l26f_cursor_u32(c, &v); m->n_vocab = v;
        } else if (l26f_str_eq(key, "bailing_hybrid.context_length")) {
            uint32_t v = 0; l26f_cursor_u32(c, &v); m->max_seq_len = v;
        } else if (l26f_str_eq(key, "bailing_hybrid.attention.q_lora_rank")) {
            uint32_t v = 0; l26f_cursor_u32(c, &v); m->q_lora_rank = v;
        } else if (l26f_str_eq(key, "bailing_hybrid.attention.kv_lora_rank")) {
            uint32_t v = 0; l26f_cursor_u32(c, &v); m->kv_lora_rank = v;
        } else if (l26f_str_eq(key, "bailing_hybrid.rope.dimension_count")) {
            uint32_t v = 0; l26f_cursor_u32(c, &v); m->n_rot = v;
        } else if (l26f_str_eq(key, "bailing_hybrid.rope.freq_base")) {
            float v = 0; l26f_cursor_f32(c, &v); m->rope_theta = v;
        } else if (l26f_str_eq(key, "bailing_hybrid.attention.layer_norm_rms_epsilon")) {
            float v = 0; l26f_cursor_f32(c, &v); m->rms_norm_eps = v;
        } else if (l26f_str_eq(key, "tokenizer.ggml.tokens") && type == L26F_GGUF_ARRAY) {
            uint32_t it; uint64_t al;
            if (!l26f_cursor_u32(c, &it)) return;
            if (!l26f_cursor_u64(c, &al)) return;
            m->tok_tokens_pos = (uint64_t)(c->ptr - m->map);
            m->tok_tokens_count = al;
            m->tok_found = true;
            uint64_t ss = l26f_scalar_size(it);
            if (ss) { if (!l26f_cursor_skip(c, al * ss)) return; }
            else { for (uint64_t j = 0; j < al; j++) if (!l26f_skip_value(c, it, 1)) return; }
        } else if (l26f_str_eq(key, "tokenizer.ggml.merges") && type == L26F_GGUF_ARRAY) {
            uint32_t it; uint64_t al;
            if (!l26f_cursor_u32(c, &it)) return;
            if (!l26f_cursor_u64(c, &al)) return;
            m->tok_merges_pos = (uint64_t)(c->ptr - m->map);
            m->tok_merges_count = al;
            uint64_t ss = l26f_scalar_size(it);
            if (ss) { if (!l26f_cursor_skip(c, al * ss)) return; }
            else { for (uint64_t j = 0; j < al; j++) if (!l26f_skip_value(c, it, 1)) return; }
        } else if (l26f_str_eq(key, "tokenizer.ggml.bos_token_id") && type == L26F_GGUF_UINT32) {
            l26f_cursor_u32(c, (uint32_t *)&m->tok_bos_id);
        } else if (l26f_str_eq(key, "tokenizer.ggml.eos_token_id") && type == L26F_GGUF_UINT32) {
            l26f_cursor_u32(c, (uint32_t *)&m->tok_eos_id);
        } else if (l26f_str_eq(key, "tokenizer.ggml.unknown_token_id") && type == L26F_GGUF_UINT32) {
            int32_t v = 0; l26f_cursor_read(c, &v, 4);
        } else if (l26f_str_eq(key, "bailing_hybrid.nextn_predict_layers")) {
            uint32_t v = 0; l26f_cursor_u32(c, &v); m->nextn_predict_layers = v;
        } else if (l26f_str_eq(key, "bailing_hybrid.expert_shared_count")) {
            uint32_t v = 0; l26f_cursor_u32(c, &v); m->expert_shared_count = v;
        } else {
            if (!l26f_skip_value(c, type, 0)) return;
        }
    }
}

static void l26f_parse_tensors(l26f_model *m, l26f_cursor *c) {
    m->tensors = (l26f_tensor *)calloc(m->n_tensors, sizeof(l26f_tensor));
    if (!m->tensors) { l26f_cursor_error(c, "out of memory for tensors"); return; }

    m->alignment = 32;
    uint64_t max_alignment = m->alignment - 1;

    for (uint64_t i = 0; i < m->n_tensors; i++) {
        l26f_tensor *t = &m->tensors[i];
        if (!l26f_cursor_string(c, &t->name)) return;
        if (!l26f_cursor_u32(c, &t->ndim)) return;
        if (t->ndim > L26F_MAX_DIMS) { l26f_cursor_error(c, "tensor has too many dims"); return; }
        t->elements = 1;
        uint32_t j;
        for (j = 0; j < t->ndim; j++) {
            if (!l26f_cursor_u64(c, &t->dim[j])) return;
            t->elements *= t->dim[j];
        }
        if (!l26f_cursor_u32(c, &t->type)) return;
        uint64_t offset;
        if (!l26f_cursor_u64(c, &offset)) return;

        t->bytes = t->elements;
        if (t->type < sizeof(l26f_types)/sizeof(l26f_types[0]) && l26f_types[t->type].name) {
            uint32_t bs = l26f_types[t->type].block_size;
            uint32_t ts = l26f_types[t->type].type_size;
            if (bs > 1) {
                t->bytes = (t->elements / bs) * ts;
            }
        }

        // Store the GGUF offset; abs_offset will be computed after all entries
        t->abs_offset = offset;
    }

    // Tensor data starts after all entries, aligned to 32 bytes
    uint64_t pos = (uint64_t)(c->ptr - m->map);
    uint64_t aligned = (pos + max_alignment) & ~max_alignment;
    m->tensor_data_pos = aligned;

    // Now apply the proper abs_offset for each tensor
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        m->tensors[i].abs_offset += m->tensor_data_pos;
    }
}

// ---- Public API ----

void l26f_die(const char *msg) {
    fprintf(stderr, "l26f: %s\n", msg);
    exit(1);
}

static void l26f_die_errno(const char *msg, const char *path) {
    fprintf(stderr, "l26f: %s: %s: %s\n", msg, path, strerror(errno));
    exit(1);
}

void l26f_model_open(l26f_model *m, const char *path) {
    memset(m, 0, sizeof(*m));
    m->fd = -1;

    int fd = open(path, O_RDONLY);
    if (fd == -1) l26f_die_errno("cannot open model", path);

    struct stat st;
    if (fstat(fd, &st) == -1) l26f_die_errno("cannot stat model", path);
    if (st.st_size < 32) l26f_die("model file too small");

    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) l26f_die_errno("cannot mmap model", path);

    m->fd = fd;
    m->map = (const uint8_t *)map;
    m->size = (uint64_t)st.st_size;

    l26f_cursor c = l26f_cursor_at(m, 0);
    uint32_t magic;
    if (!l26f_cursor_u32(&c, &magic)) l26f_die(c.error);
    if (magic != L26F_GGUF_MAGIC) l26f_die("not a GGUF file");
    if (!l26f_cursor_u32(&c, &m->version)) l26f_die(c.error);
    if (!l26f_cursor_u64(&c, &m->n_tensors)) l26f_die(c.error);
    if (!l26f_cursor_u64(&c, &m->n_kv)) l26f_die(c.error);

    if (m->version != 3) l26f_die("only GGUF v3 supported");

    l26f_parse_metadata(m, &c);
    l26f_parse_tensors(m, &c);

    printf("l26f: loaded %s\n", path);
    printf("l26f: arch=bailing_hybrid, layers=%u, embd=%u, vocab=%u\n",
           m->n_layer, m->n_embd, m->n_vocab);
    printf("l26f: %" PRIu64 " tensors, model size = ", m->n_tensors);
    double gib = (double)m->size / (1024.0 * 1024.0 * 1024.0);
    printf("%.2f GiB\n", gib);
}

void l26f_model_close(l26f_model *m) {
    free(m->tensors);
    if (m->map) munmap((void *)m->map, (size_t)m->size);
    if (m->fd >= 0) close(m->fd);
}

l26f_tensor *l26f_model_find_tensor(const l26f_model *m, const char *name) {
    size_t len = strlen(name);
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        l26f_tensor *t = &m->tensors[i];
        if (t->name.len == len && memcmp(t->name.ptr, name, len) == 0)
            return t;
    }
    return NULL;
}

const void *l26f_tensor_data(const l26f_model *m, const l26f_tensor *t) {
    return m->map + t->abs_offset;
}
