// l26f: Ling-2.6-flash Narrow-Metal Inference Engine
// GGUF model loader — adapted from ds4.c

#ifndef L26F_H
#define L26F_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#define L26F_MAX_DIMS  8
#define L26F_MAX_LAYERS 33

// ---- String view ----
typedef struct {
    const char *ptr;
    uint64_t len;
} l26f_str;

static inline bool l26f_str_eq(l26f_str a, const char *b) {
    size_t blen = strlen(b);
    return a.len == blen && memcmp(a.ptr, b, blen) == 0;
}

// ---- GGUF types ----
typedef struct {
    const char *name;
    uint32_t block_size;
    uint32_t type_size;
} l26f_type_info;

static const l26f_type_info l26f_types[] = {
    [0]  = {"f32",     1,   4},
    [1]  = {"f16",     1,   2},
    [8]  = {"q8_0",   32,  34},
    [10] = {"q2_k",  256,  84},
    [12] = {"q4_k",  256, 144},
    [13] = {"q5_k",  256, 176},
    [14] = {"q6_k",  256, 210},
    [16] = {"iq2_xxs",256, 66},
    [20] = {"iq4_nl", 32,  18},  // QK4_NL=32, sizeof(block_iq4_nl)=18
    [25] = {"i16",     1,   2},
    [26] = {"i32",     1,   4},
};

static const char *l26f_type_name(uint32_t type) {
    if (type < sizeof(l26f_types)/sizeof(l26f_types[0]) && l26f_types[type].name)
        return l26f_types[type].name;
    return "?";
}

// ---- GGUF tensor descriptor ----
typedef struct {
    l26f_str name;
    uint32_t ndim;
    uint64_t dim[L26F_MAX_DIMS];
    uint32_t type;       // GGUF tensor type
    uint64_t abs_offset; // absolute offset in mmap
    uint64_t elements;
    uint64_t bytes;
} l26f_tensor;

// ---- GGUF model ----
typedef struct {
    int fd;
    const uint8_t *map;
    uint64_t size;

    uint32_t version;
    uint64_t n_kv;
    uint64_t n_tensors;
    uint64_t alignment;
    uint64_t tensor_data_pos;

    l26f_tensor *tensors;

    // Model metadata (hardcoded after parsing)
    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    uint32_t n_ff;
    uint32_t n_expert;
    uint32_t n_expert_used;
    uint32_t n_expert_groups;
    uint32_t n_group_used;
    uint32_t n_vocab;
    uint32_t max_seq_len;
    uint32_t q_lora_rank;
    uint32_t kv_lora_rank;
    uint32_t n_rot;
    float    rope_theta;
    float    rms_norm_eps;
    bool     is_mla[L26F_MAX_LAYERS];
    uint32_t nextn_predict_layers;
    uint32_t expert_shared_count;
} l26f_model;

// ---- GGUF file format constants ----
#define L26F_GGUF_MAGIC 0x46554747u

enum {
    L26F_GGUF_UINT8   = 0,
    L26F_GGUF_INT8    = 1,
    L26F_GGUF_UINT16  = 2,
    L26F_GGUF_INT16   = 3,
    L26F_GGUF_UINT32  = 4,
    L26F_GGUF_INT32   = 5,
    L26F_GGUF_FLOAT32 = 6,
    L26F_GGUF_BOOL    = 7,
    L26F_GGUF_STRING  = 8,
    L26F_GGUF_ARRAY   = 9,
    L26F_GGUF_UINT64  = 10,
    L26F_GGUF_INT64   = 11,
    L26F_GGUF_FLOAT64 = 12,
};

// ---- Public API ----
void l26f_model_open(l26f_model *m, const char *path);
void l26f_model_close(l26f_model *m);
l26f_tensor *l26f_model_find_tensor(const l26f_model *m, const char *name);
const void *l26f_tensor_data(const l26f_model *m, const l26f_tensor *t);

#endif
