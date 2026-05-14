// l26f: inference driver — single-token decode through GLA layers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
#include <inttypes.h>
#include "xcommon.h"
#include "l26f.h"
#include "ds4_metal.h"
#include "l26f_metal.h"
#include "l26f_tokenizer.h"

// Forward declarations for MLA CPU implementation
typedef struct l26f_mla_kv_cache l26f_mla_kv_cache;
extern l26f_mla_kv_cache *l26f_mla_kv_cache_alloc(int max_seq, int kv_dim);
extern void l26f_mla_kv_cache_free(l26f_mla_kv_cache *c);
extern int l26f_mla_layer_cpu(l26f_model *m, uint32_t layer, int position,
    l26f_mla_kv_cache *kv_cache, const float *hidden_cpu, float *hidden_out_cpu);

// Forward declarations for MLA GPU implementation
typedef struct l26f_mla_kv_cache_gpu l26f_mla_kv_cache_gpu;
typedef struct l26f_mla_compute l26f_mla_compute;

extern l26f_mla_kv_cache_gpu *l26f_mla_kv_cache_gpu_alloc(int max_seq, int kv_dim);
extern void l26f_mla_kv_cache_gpu_free(l26f_mla_kv_cache_gpu *c);
extern l26f_mla_compute *l26f_mla_compute_alloc(uint32_t n_embd, uint32_t n_head,
    uint32_t q_lora_rank, uint32_t kv_lora_rank, uint32_t n_rot, uint32_t head_dim);
extern void l26f_mla_compute_free(l26f_mla_compute *mc);
extern int l26f_mla_layer_gpu(l26f_model *m, uint32_t layer, int position,
    l26f_mla_kv_cache_gpu *kv_cache, l26f_mla_compute *mc,
    ds4_metal_tensor *hidden_1xN, ds4_metal_tensor *out_1xN);

static int use_gpu_mla(void) {
    static int initialized = 0;
    static int use_gpu = 0;
    if (!initialized) {
        use_gpu = getenv("L26F_CPU_MLA") == NULL;
        initialized = 1;
    }
    return use_gpu;
}

// ---- Per-layer compute buffer set ----
// Reused across layers to avoid excessive allocation.
// We need: normed, qkv, gate, gla_out+state, proj, ffn_mid, ffn_down
// All are n_embd-sized except qkv (3*n_embd) and gla_out (n_embd + S*S*H).

typedef struct {
    ds4_metal_tensor *normed_1xN;
    ds4_metal_tensor *qkv_1x3N;
    ds4_metal_tensor *gate_out_1xN;
    ds4_metal_tensor *q_rope_1xN;
    ds4_metal_tensor *k_rope_1xN;
    ds4_metal_tensor *gla_out_1xNxSxSxH;
    ds4_metal_tensor *gated_gla_1xN;
    ds4_metal_tensor *attn_proj_1xN;
    ds4_metal_tensor *post_attn_1xN;
    ds4_metal_tensor *ffn_normed_1xN;
    ds4_metal_tensor *ffn_gate_1xF;
    ds4_metal_tensor *ffn_up_1xF;
    ds4_metal_tensor *ffn_mid_1xF;
    ds4_metal_tensor *ffn_down_1xN;
    ds4_metal_tensor *moe_out_1xN;
    ds4_metal_tensor *shexp_out_1xN;
    ds4_metal_tensor *router_logits_1xE;
    ds4_metal_tensor *moe_sel_idx_K;
    ds4_metal_tensor *moe_sel_wt_K;
    ds4_metal_tensor *moe_expert_gate_8xM;
    ds4_metal_tensor *moe_expert_up_8xM;
    ds4_metal_tensor *moe_expert_mid_8xM;
    ds4_metal_tensor *moe_expert_down_8xN;
    ds4_metal_tensor *moe_gate_off_K;
    ds4_metal_tensor *moe_up_off_K;
    ds4_metal_tensor *moe_down_off_K;
    ds4_metal_tensor *moe_gate_cache_8xMxN;
    ds4_metal_tensor *moe_up_cache_8xMxN;
    ds4_metal_tensor *moe_down_cache_8xNxM;
} l26f_compute;

#define L26F_PREFILL_MAX_T 128
typedef struct l26f_prefill_compute l26f_prefill_compute;

// GLA state: one per GLA layer, persists across tokens
typedef struct {
    ds4_metal_tensor *state;
} l26f_gla_state;

static int l26f_write_gla_slopes_1xN(
        l26f_model *m,
        uint32_t layer,
        ds4_metal_tensor *g_slope_1xN) {
    const uint32_t S = 128;
    const uint32_t H = m->n_head;
    const uint32_t n_embd = m->n_embd;
    float *slopes_1xN = (float *)malloc((uint64_t)n_embd * sizeof(float));
    if (!slopes_1xN) return 0;

    const float rate = powf(2.0f, -(log2f((float)H) - 3.0f));
    const uint32_t denom_layers = m->n_layer > 1 ? m->n_layer - 1 : 1;
    const float layer_factor = 1.0f - ((float)layer / (float)denom_layers) + 1.0e-5f;
    for (uint32_t h = 0; h < H; h++) {
        const float exp_val = exp2f(-rate * (float)(h + 1));
        const float slope = expf(-layer_factor * exp_val);
        for (uint32_t d = 0; d < S; d++) {
            slopes_1xN[h * S + d] = slope;
        }
    }

    const int ok = ds4_metal_tensor_write(g_slope_1xN, 0, slopes_1xN,
                                          (uint64_t)n_embd * sizeof(float));
    free(slopes_1xN);
    return ok;
}

typedef struct {
    l26f_model *model;
    l26f_compute comp;
    l26f_gla_state gla_states[32];
    l26f_mla_kv_cache_gpu *mla_kv_gpu[32];
    l26f_mla_kv_cache    *mla_kv_cpu[32];
    l26f_mla_compute *mla_comp;
    ds4_metal_tensor *hidden_1xN;
    ds4_metal_tensor *output_normed_1xN;
    ds4_metal_tensor *logits_1xV;
    ds4_metal_tensor *sample_idx_1xI;
    ds4_metal_tensor *moe_gate_all_off_1xE[32];
    ds4_metal_tensor *moe_up_all_off_1xE[32];
    ds4_metal_tensor *moe_down_all_off_1xE[32];
    ds4_metal_tensor *moe_gate_cache_off_K[32];
    ds4_metal_tensor *moe_up_cache_off_K[32];
    ds4_metal_tensor *moe_down_cache_off_K[32];
    ds4_metal_tensor *gla_slopes_1xN[32];
    l26f_prefill_compute *prefill;
} l26f_session;

typedef struct l26f_prefill_compute {
    uint32_t max_tokens;
    ds4_metal_tensor *hidden_TxN;
    ds4_metal_tensor *post_attn_TxN;
    ds4_metal_tensor *normed_TxN;
    ds4_metal_tensor *router_logits_TxE;
    ds4_metal_tensor *sel_idx_TxK;
    ds4_metal_tensor *sel_wt_TxK;
    ds4_metal_tensor *tokens_per_expert_E;
    ds4_metal_tensor *ids_buffer_ExT;
    ds4_metal_tensor *gate_out_TxKxM;
    ds4_metal_tensor *up_out_TxKxM;
    ds4_metal_tensor *mid_TxKxM;
    ds4_metal_tensor *down_out_TxKxN;
    ds4_metal_tensor *moe_out_TxN;
    ds4_metal_tensor *shexp_gate_1xF;
    ds4_metal_tensor *shexp_up_1xF;
    ds4_metal_tensor *shexp_mid_1xF;
    ds4_metal_tensor *shexp_out_1xN;
};

// ---- helpers ----

static float l26f_tensor_checksum(const ds4_metal_tensor *t, uint64_t bytes, int *out_nans) {
    float *data = (float *)malloc(bytes);
    if (!data) { *out_nans = -1; return 0.0f; }
    ds4_metal_tensor_read(t, 0, data, bytes);
    float sum = 0;
    int nans = 0;
    uint64_t n = bytes / sizeof(float);
    for (uint64_t i = 0; i < n; i++) {
        if (isnan(data[i])) { nans++; continue; }
        sum += data[i];
    }
    free(data);
    *out_nans = nans;
    return sum;
}

static void l26f_checksum_print(const char *label, const ds4_metal_tensor *t, uint64_t bytes) {
    int nans;
    float sum = l26f_tensor_checksum(t, bytes, &nans);
    fprintf(stderr, "    CHECKSUM %-20s  sum=%.6f  nans=%d\n", label, sum, nans);
}

#ifndef L26F_XLOG_LEVEL
#define L26F_XLOG_LEVEL 1
#endif

#if L26F_XLOG_LEVEL >= 1
static int XREG_ENABLED(void) {
    static int initialized = 0;
    static int enabled = 0;
    if (!initialized) {
        enabled = getenv("L26F_REGRESSION") != NULL;
        initialized = 1;
    }
    return enabled;
}

static int XREG_LAYERS_ENABLED(void) {
    static int initialized = 0;
    static int enabled = 0;
    if (!initialized) {
        enabled = getenv("L26F_REGRESSION_LAYERS") != NULL;
        initialized = 1;
    }
    return enabled;
}

#define XLOG_REG_FPRINTF(...) fprintf(stderr, __VA_ARGS__)
#define XLOG_REG(args) do { \
    if (XREG_ENABLED()) { \
        fprintf(stderr, "REG "); \
        XLOG_REG_FPRINTF args; \
        fputc('\n', stderr); \
    } \
} while (0)
#define XLOG_REG_LAYER(args) do { \
    if (XREG_LAYERS_ENABLED()) { \
        fprintf(stderr, "REG "); \
        XLOG_REG_FPRINTF args; \
        fputc('\n', stderr); \
    } \
} while (0)
#define XLOG_MODEL(args) do { \
    fprintf(stderr, "XMODEL "); \
    XLOG_REG_FPRINTF args; \
    fputc('\n', stderr); \
} while (0)
#define XLOG_PROF(args) do { \
    if (XPROF_ENABLED()) { \
        fprintf(stderr, "XPROF "); \
        XLOG_REG_FPRINTF args; \
        fputc('\n', stderr); \
    } \
} while (0)
typedef struct {
    float sum;
    float min;
    float max;
    int   nans;
} XREG_TENSOR_SUMMARY_1xU;

static XREG_TENSOR_SUMMARY_1xU XREG_TENSOR_SUMMARIZE_1xU(const ds4_metal_tensor *t_1xU, uint64_t bytes) {
    XREG_TENSOR_SUMMARY_1xU s_1xU = {0.0f, 0.0f, 0.0f, -1};
    float *data_1xU = (float *)malloc(bytes);
    if (!data_1xU) return s_1xU;
    if (!ds4_metal_tensor_read(t_1xU, 0, data_1xU, bytes)) {
        free(data_1xU);
        return s_1xU;
    }

    const uint64_t n = bytes / sizeof(float);
    int have_value = 0;
    s_1xU.nans = 0;
    for (uint64_t i = 0; i < n; i++) {
        float v = data_1xU[i];
        if (isnan(v)) {
            s_1xU.nans++;
            continue;
        }
        s_1xU.sum += v;
        if (!have_value) {
            s_1xU.min = v;
            s_1xU.max = v;
            have_value = 1;
        } else {
            if (v < s_1xU.min) s_1xU.min = v;
            if (v > s_1xU.max) s_1xU.max = v;
        }
    }
    free(data_1xU);
    return s_1xU;
}

#else
#define XLOG_REG_FPRINTF(...) ((void)0)
#define XLOG_REG(args)       ((void)0)
#define XLOG_REG_LAYER(args) ((void)0)
#define XLOG_MODEL(args)     ((void)0)
#define XLOG_PROF(args)      ((void)0)
#endif

typedef enum {
    XPROF_GLA = 0,
    XPROF_GLA_NORM,
    XPROF_GLA_QKV,
    XPROF_GLA_GATE,
    XPROF_GLA_ATTN,
    XPROF_GLA_PROJ,
    XPROF_MLA,
    XPROF_MOE,
    XPROF_MOE_NORM,
    XPROF_MOE_ROUTER,
    XPROF_MOE_ROUTE,
    XPROF_MOE_OFFSETS,
    XPROF_MOE_GATE_UP,
    XPROF_MOE_DOWN,
    XPROF_MOE_SHARED,
    XPROF_MOE_ACCUM,
    XPROF_MOE_RESIDUAL,
    XPROF_DENSE,
    XPROF_LOGITS,
    XPROF_LOGITS_NORM,
    XPROF_LOGITS_HEAD,
    XPROF_SAMPLE,
    XPROF_COUNT
} XPROF_STAGE;

typedef struct {
    double ms;
    uint64_t count;
} XPROF_ACCUM;

static XPROF_ACCUM XPROF_TOTALS[XPROF_COUNT];

static __attribute__((unused)) const char *XPROF_STAGE_NAME(XPROF_STAGE stage) {
    switch (stage) {
    case XPROF_GLA:    return "gla";
    case XPROF_GLA_NORM: return "gla_norm";
    case XPROF_GLA_QKV:  return "gla_qkv";
    case XPROF_GLA_GATE: return "gla_gate";
    case XPROF_GLA_ATTN: return "gla_attn";
    case XPROF_GLA_PROJ: return "gla_proj";
    case XPROF_MLA:    return "mla";
    case XPROF_MOE:    return "moe";
    case XPROF_MOE_NORM:     return "moe_norm";
    case XPROF_MOE_ROUTER:   return "moe_router";
    case XPROF_MOE_ROUTE:    return "moe_route";
    case XPROF_MOE_OFFSETS:  return "moe_offsets";
    case XPROF_MOE_GATE_UP:  return "moe_gate_up";
    case XPROF_MOE_DOWN:     return "moe_down";
    case XPROF_MOE_SHARED:   return "moe_shared";
    case XPROF_MOE_ACCUM:    return "moe_accum";
    case XPROF_MOE_RESIDUAL: return "moe_residual";
    case XPROF_DENSE:  return "dense";
    case XPROF_LOGITS: return "logits";
    case XPROF_LOGITS_NORM: return "logits_norm";
    case XPROF_LOGITS_HEAD: return "logits_head";
    case XPROF_SAMPLE: return "sample";
    default:           return "unknown";
    }
}

static int XPROF_ENABLED(void) {
    static int initialized = 0;
    static int enabled = 0;
    if (!initialized) {
        enabled = getenv("L26F_PROFILE") != NULL;
        initialized = 1;
    }
    return enabled;
}

static int XPROF_SYNC_ENABLED(void) {
    static int initialized = 0;
    static int enabled = 0;
    if (!initialized) {
        enabled = getenv("L26F_PROFILE_SYNC") != NULL;
        initialized = 1;
    }
    return enabled;
}

static int XPROF_GLA_DEEP_ENABLED(void) {
    static int initialized = 0;
    static int enabled = 0;
    if (!initialized) {
        enabled = getenv("L26F_PROFILE_GLA") != NULL;
        initialized = 1;
    }
    return enabled;
}

static int XPROF_MOE_DEEP_ENABLED(void) {
    static int initialized = 0;
    static int enabled = 0;
    if (!initialized) {
        enabled = getenv("L26F_PROFILE_MOE") != NULL;
        initialized = 1;
    }
    return enabled;
}

static double XPROF_NOW_MS(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static int XPROF_STAGE_BEGIN(double *t_ms) {
    if (!XPROF_ENABLED()) {
        if (t_ms) *t_ms = 0.0;
        return 1;
    }
    if (XPROF_SYNC_ENABLED()) {
        if (!ds4_metal_synchronize()) return 0;
        if (!ds4_metal_begin_commands()) return 0;
    }
    if (t_ms) *t_ms = XPROF_NOW_MS();
    return 1;
}

static int XPROF_STAGE_END(XPROF_STAGE stage, uint32_t layer, double t_ms) {
    (void)layer;
    if (!XPROF_ENABLED()) return 1;
    if (XPROF_SYNC_ENABLED()) {
        if (!ds4_metal_end_commands()) return 0;
    }
    const double elapsed_ms = XPROF_NOW_MS() - t_ms;
    XPROF_TOTALS[stage].ms += elapsed_ms;
    XPROF_TOTALS[stage].count++;
    XLOG_PROF(("stage=%s layer=%u ms=%.3f", XPROF_STAGE_NAME(stage), layer, elapsed_ms));
    if (XPROF_SYNC_ENABLED()) {
        if (!ds4_metal_begin_commands()) return 0;
    }
    return 1;
}

static int XPROF_MOE_STAGE_BEGIN(double *t_ms) {
    if (!XPROF_MOE_DEEP_ENABLED()) {
        if (t_ms) *t_ms = 0.0;
        return 1;
    }
    return XPROF_STAGE_BEGIN(t_ms);
}

static int XPROF_MOE_STAGE_END(XPROF_STAGE stage, uint32_t layer, double t_ms) {
    if (!XPROF_MOE_DEEP_ENABLED()) return 1;
    return XPROF_STAGE_END(stage, layer, t_ms);
}

static void XPROF_DIRECT_ADD(XPROF_STAGE stage, double elapsed_ms) {
    if (!XPROF_ENABLED()) return;
    XPROF_TOTALS[stage].ms += elapsed_ms;
    XPROF_TOTALS[stage].count++;
    XLOG_PROF(("stage=%s layer=999 ms=%.3f", XPROF_STAGE_NAME(stage), elapsed_ms));
}

static int XPROF_STANDALONE_BEGIN(double *t_ms) {
    if (!XPROF_ENABLED()) {
        if (t_ms) *t_ms = 0.0;
        return 1;
    }
    if (XPROF_SYNC_ENABLED()) {
        if (!ds4_metal_synchronize()) return 0;
    }
    if (t_ms) *t_ms = XPROF_NOW_MS();
    return 1;
}

static int XPROF_STANDALONE_END(XPROF_STAGE stage, uint32_t layer, double t_ms) {
    if (!XPROF_ENABLED()) return 1;
    if (XPROF_SYNC_ENABLED()) {
        if (!ds4_metal_synchronize()) return 0;
    }
    const double elapsed_ms = XPROF_NOW_MS() - t_ms;
    XPROF_TOTALS[stage].ms += elapsed_ms;
    XPROF_TOTALS[stage].count++;
    XLOG_PROF(("stage=%s layer=%u ms=%.3f", XPROF_STAGE_NAME(stage), layer, elapsed_ms));
    return 1;
}

static void XPROF_PRINT_TOTALS(void) {
    if (!XPROF_ENABLED()) return;
    for (int i = 0; i < XPROF_COUNT; i++) {
        if (XPROF_TOTALS[i].count == 0) continue;
        XLOG_PROF(("summary stage=%s count=%llu total_ms=%.3f avg_ms=%.3f",
                   XPROF_STAGE_NAME((XPROF_STAGE)i),
                   (unsigned long long)XPROF_TOTALS[i].count,
                   XPROF_TOTALS[i].ms,
                   XPROF_TOTALS[i].ms / (double)XPROF_TOTALS[i].count));
    }
}

static void XREG_PRINT_TENSOR_1xU(const char *label, const ds4_metal_tensor *t_1xU, uint64_t bytes) {
#if L26F_XLOG_LEVEL >= 1
    if (!XREG_ENABLED()) return;
    XREG_TENSOR_SUMMARY_1xU s_1xU = XREG_TENSOR_SUMMARIZE_1xU(t_1xU, bytes);
    XLOG_REG(("tensor=%s sum=%.9g min=%.9g max=%.9g nans=%d",
              label, s_1xU.sum, s_1xU.min, s_1xU.max, s_1xU.nans));
#else
    (void)label; (void)t_1xU; (void)bytes;
#endif
}

static void XREG_PRINT_LAYER_HIDDEN_1xN(uint32_t layer, const ds4_metal_tensor *hidden_1xN, uint64_t bytes) {
#if L26F_XLOG_LEVEL >= 1
    if (!XREG_LAYERS_ENABLED()) return;
    XREG_TENSOR_SUMMARY_1xU s_1xN = XREG_TENSOR_SUMMARIZE_1xU(hidden_1xN, bytes);
    XLOG_REG_LAYER(("layer=%u hidden_1xN sum=%.9g min=%.9g max=%.9g nans=%d",
                    layer, s_1xN.sum, s_1xN.min, s_1xN.max, s_1xN.nans));
#else
    (void)layer; (void)hidden_1xN; (void)bytes;
#endif
}

static l26f_tensor *l26f_layer_tensor(const l26f_model *m, uint32_t layer, const char *suffix) {
    char name[128];
    snprintf(name, sizeof(name), "blk.%u.%s", layer, suffix);
    return l26f_model_find_tensor(m, name);
}

static int XMODEL_TENSOR_RANGE(const l26f_model *m, uint64_t *range_off, uint64_t *range_bytes) {
    if (!m || !range_off || !range_bytes || m->n_tensors == 0) return 0;

    uint64_t min_off = UINT64_MAX;
    uint64_t max_end = 0;
    uint64_t max_tensor_bytes = 0;
    l26f_str max_tensor_name = {0};

    for (uint64_t i = 0; i < m->n_tensors; i++) {
        const l26f_tensor *t = &m->tensors[i];
        if (t->bytes == 0) continue;
        if (t->abs_offset > m->size || t->bytes > m->size - t->abs_offset) return 0;

        const uint64_t end = t->abs_offset + t->bytes;
        if (t->abs_offset < min_off) min_off = t->abs_offset;
        if (end > max_end) max_end = end;
        if (t->bytes > max_tensor_bytes) {
            max_tensor_bytes = t->bytes;
            max_tensor_name = t->name;
        }
    }

    if (min_off == UINT64_MAX || max_end <= min_off) return 0;
    *range_off = min_off;
    *range_bytes = max_end - min_off;
#if L26F_XLOG_LEVEL < 1
    (void)max_tensor_bytes;
    (void)max_tensor_name;
#endif
    XLOG_MODEL(("tensor_range off=%.2f MiB bytes=%.2f MiB max_tensor=%.2f MiB %.*s",
                min_off / 1024.0 / 1024.0,
                (max_end - min_off) / 1024.0 / 1024.0,
                max_tensor_bytes / 1024.0 / 1024.0,
                (int)(max_tensor_name.len < 48 ? max_tensor_name.len : 48),
                max_tensor_name.ptr ? max_tensor_name.ptr : ""));
    return 1;
}

// =========================================================================
// L26F_DBG: Debug checkpoint infrastructure for fused MoE NaN investigation
// =========================================================================
// Enable with: -DL26F_DBG_FUSED
// Compare mode: -DL26F_DBG_COMPARE (implies L26F_DBG_FUSED)
// Control which layer to debug: env L26F_DBG_LAYER=N (default: layer 1)
// Control per-stage breaks: env L26F_DBG_BREAKS=1 (default: 1, set 0 for no-break test)

#ifdef L26F_DBG_COMPARE
#define L26F_DBG_FUSED
#endif

#ifdef L26F_DBG_FUSED

typedef enum {
    L26F_DBG_VAL_OK,
    L26F_DBG_VAL_LARGE,
    L26F_DBG_VAL_POS_INF,
    L26F_DBG_VAL_NEG_INF,
    L26F_DBG_VAL_NAN,
    L26F_DBG_VAL_ZERO
} l26f_dbg_val_class;

static l26f_dbg_val_class L26F_DBG_CLASSIFY(float v, float threshold) {
    if (isnan(v))                    return L26F_DBG_VAL_NAN;
    if (isinf(v) && v > 0)           return L26F_DBG_VAL_POS_INF;
    if (isinf(v) && v < 0)           return L26F_DBG_VAL_NEG_INF;
    if (v == 0.0f)                    return L26F_DBG_VAL_ZERO;
    if (fabsf(v) > threshold)         return L26F_DBG_VAL_LARGE;
    return L26F_DBG_VAL_OK;
}

static void L26F_DBG_CHECKPOINT_IMPL(
        const char *label,
        const ds4_metal_tensor *t, uint64_t nfloats,
        const char *file, int line)
{
    float insane_threshold = 1e4f;
    uint64_t bytes = nfloats * sizeof(float);
    float *data = (float *)malloc(bytes);
    if (!data) { fprintf(stderr, "DBG OOM %s\n", label); return; }

    ds4_metal_end_commands();
    ds4_metal_tensor_read((ds4_metal_tensor *)t, 0, data, bytes);

    int n_ok = 0, n_nan = 0, n_pinf = 0, n_ninf = 0, n_large = 0, n_zero = 0;
    float sum = 0.0f, max_abs = 0.0f, min_val = 1e30f, max_val = -1e30f;
    int first_bad_idx = -1;
    l26f_dbg_val_class first_bad_class = L26F_DBG_VAL_OK;

    for (uint64_t i = 0; i < nfloats; i++) {
        l26f_dbg_val_class c = L26F_DBG_CLASSIFY(data[i], insane_threshold);
        switch (c) {
            case L26F_DBG_VAL_OK:      n_ok++;    sum += data[i]; break;
            case L26F_DBG_VAL_NAN:     n_nan++;   break;
            case L26F_DBG_VAL_POS_INF: n_pinf++;  break;
            case L26F_DBG_VAL_NEG_INF: n_ninf++;  break;
            case L26F_DBG_VAL_LARGE:   n_large++; sum += data[i]; break;
            case L26F_DBG_VAL_ZERO:    n_zero++;  break;
        }
        float a = fabsf(data[i]);
        if (a > max_abs) max_abs = a;
        if (data[i] == data[i]) { // not NaN
            if (data[i] < min_val) min_val = data[i];
            if (data[i] > max_val) max_val = data[i];
        }
        if (first_bad_idx < 0 && c != L26F_DBG_VAL_OK && c != L26F_DBG_VAL_ZERO) {
            first_bad_idx = (int)i;
            first_bad_class = c;
        }
    }

    fprintf(stderr, "DBG %s:%d %-30s n=%llu ok=%d nan=%d +inf=%d -inf=%d large=%d zero=%d sum=%.4f max_abs=%.4f range=[%.4f,%.4f]",
            file, line, label, (unsigned long long)nfloats,
            n_ok, n_nan, n_pinf, n_ninf, n_large, n_zero, sum, max_abs, min_val, max_val);
    if (first_bad_idx >= 0) {
        fprintf(stderr, " FIRST_BAD=[%d]=%f(cls=%d)", first_bad_idx,
                data[first_bad_idx], (int)first_bad_class);
    }
    fprintf(stderr, "\n");
    free(data);
    ds4_metal_begin_commands();
}

#define L26F_DBG_CHECKPOINT(LABEL, TENSOR, NFLOATS) \
    L26F_DBG_CHECKPOINT_IMPL((LABEL), (TENSOR), (NFLOATS), __FILE__, __LINE__)

static void L26F_DBG_ASSERT_EQ_IMPL(
        const char *label,
        const float *a, const float *b, uint64_t n,
        const char *file, int line)
{
    float max_diff = 0.0f;
    float tol = 1e-3f;
    int first_diff_idx = -1;
    int n_diff = 0;
    for (uint64_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > max_diff) max_diff = d;
        if (d > tol) {
            n_diff++;
            if (first_diff_idx < 0) first_diff_idx = (int)i;
        }
    }
    if (max_diff > tol) {
        fprintf(stderr, "DBG DIFF %s:%d %-30s MAX_DIFF=%.6f n_diff=%d first=[%d] a=%.6f b=%.6f\n",
                file, line, label, max_diff, n_diff, first_diff_idx,
                first_diff_idx >= 0 ? a[first_diff_idx] : 0.0f,
                first_diff_idx >= 0 ? b[first_diff_idx] : 0.0f);
    } else {
        fprintf(stderr, "DBG MATCH %s:%d %-30s max_diff=%.6f OK\n",
                file, line, label, max_diff);
    }
}

#define L26F_DBG_ASSERT_EQ(LABEL, A, B, N) \
    L26F_DBG_ASSERT_EQ_IMPL((LABEL), (A), (B), (N), __FILE__, __LINE__)

static int l26f_dbg_get_layer(void) {
    static int initialized = 0;
    static int layer = 1;
    if (!initialized) {
        const char *env = getenv("L26F_DBG_LAYER");
        if (env) layer = atoi(env);
        if (layer < 1) layer = 1;
        if (layer > 31) layer = 31;
        initialized = 1;
    }
    return layer;
}

static int l26f_dbg_get_breaks(void) {
    static int initialized = 0;
    static int breaks = 1;
    if (!initialized) {
        const char *env = getenv("L26F_DBG_BREAKS");
        if (env) breaks = atoi(env);
        initialized = 1;
    }
    return breaks;
}

static void L26F_DBG_READ(const ds4_metal_tensor *t, float *dst, uint64_t nfloats) {
    ds4_metal_end_commands();
    ds4_metal_tensor_read((ds4_metal_tensor *)t, 0, dst, nfloats * sizeof(float));
    ds4_metal_begin_commands();
}

#endif // L26F_DBG_FUSED

static int l26f_gla_layer(
        l26f_session *s, uint32_t layer, int position,
        ds4_metal_tensor *inp, ds4_metal_tensor *out) {
    l26f_model *m = s->model;
    l26f_compute *c = &s->comp;
    const uint32_t n_embd = m->n_embd;
    const uint32_t S = 128, H = m->n_head;
    const uint64_t act_bytes = (uint64_t)n_embd * sizeof(float);
    float scale = 1.0f / sqrtf((float)S);

    l26f_tensor *wt_norm_N      = l26f_layer_tensor(m, layer, "attn_norm.weight");
    l26f_tensor *wt_qkv_Nx3N    = l26f_layer_tensor(m, layer, "attn_qkv.weight");
    l26f_tensor *wt_q_norm_S    = l26f_layer_tensor(m, layer, "attn_q_norm.weight");
    l26f_tensor *wt_k_norm_S    = l26f_layer_tensor(m, layer, "attn_k_norm.weight");
    l26f_tensor *wt_gate_NxN    = l26f_layer_tensor(m, layer, "attn_gate.weight");
    l26f_tensor *wt_layer_out_N = l26f_layer_tensor(m, layer, "layer_output_norm.weight");
    l26f_tensor *wt_out_NxN     = l26f_layer_tensor(m, layer, "attn_output.weight");
    if (!wt_norm_N || !wt_qkv_Nx3N || !wt_q_norm_S || !wt_k_norm_S ||
        !wt_gate_NxN || !wt_layer_out_N || !wt_out_NxN) {
        fprintf(stderr, "l26f: layer %u missing GLA tensors\n", layer);
        return 0;
    }

    // 1. RMS norm
    double XPROF_T0 = 0.0;
    if (XPROF_GLA_DEEP_ENABLED() && !XPROF_STAGE_BEGIN(&XPROF_T0)) return 0;
    if (!ds4_metal_rms_norm_weight_tensor(c->normed_1xN, inp,
            m->map, m->size, wt_norm_N->abs_offset, n_embd, m->rms_norm_eps))
        return 0;
    if (XPROF_GLA_DEEP_ENABLED() && !XPROF_STAGE_END(XPROF_GLA_NORM, layer, XPROF_T0)) return 0;

    if (XPROF_GLA_DEEP_ENABLED() && !XPROF_STAGE_BEGIN(&XPROF_T0)) return 0;
    if (!l26f_metal_matvec_quant(c->qkv_1x3N, c->normed_1xN,
            m->map, m->size, wt_qkv_Nx3N->abs_offset,
            wt_qkv_Nx3N->dim[0], wt_qkv_Nx3N->dim[1], wt_qkv_Nx3N->type, 1))
        return 0;
    if (XPROF_GLA_DEEP_ENABLED() && !XPROF_STAGE_END(XPROF_GLA_QKV, layer, XPROF_T0)) return 0;

    if (XPROF_GLA_DEEP_ENABLED() && !XPROF_STAGE_BEGIN(&XPROF_T0)) return 0;
    if (!l26f_metal_matvec_quant(c->gate_out_1xN, c->normed_1xN,
            m->map, m->size, wt_gate_NxN->abs_offset,
            wt_gate_NxN->dim[0], wt_gate_NxN->dim[1], wt_gate_NxN->type, 1))
        return 0;
    if (XPROF_GLA_DEEP_ENABLED() && !XPROF_STAGE_END(XPROF_GLA_GATE, layer, XPROF_T0)) return 0;

    ds4_metal_tensor *q_view_1xN = ds4_metal_tensor_view(c->qkv_1x3N, 0, act_bytes);
    ds4_metal_tensor *k_view_1xN = ds4_metal_tensor_view(c->qkv_1x3N, act_bytes, act_bytes);
    ds4_metal_tensor *v_view_1xN = ds4_metal_tensor_view(c->qkv_1x3N, 2*act_bytes, act_bytes);
    if (!q_view_1xN || !k_view_1xN || !v_view_1xN) {
        ds4_metal_tensor_free(v_view_1xN);
        ds4_metal_tensor_free(k_view_1xN);
        ds4_metal_tensor_free(q_view_1xN);
        return 0;
    }

    int ok = l26f_metal_gla_qk_norm_rope(c->q_rope_1xN, c->k_rope_1xN,
                    q_view_1xN, k_view_1xN,
                    m->map, m->size, wt_q_norm_S->abs_offset, wt_k_norm_S->abs_offset,
                    S, H, position, m->rope_theta, m->rms_norm_eps);
    if (ok && !s->gla_slopes_1xN[layer]) ok = 0;
    if (!ok) {
        ds4_metal_tensor_free(v_view_1xN);
        ds4_metal_tensor_free(k_view_1xN);
        ds4_metal_tensor_free(q_view_1xN);
        return 0;
    }

    if (XPROF_GLA_DEEP_ENABLED() && !XPROF_STAGE_BEGIN(&XPROF_T0)) return 0;
    ok = l26f_metal_gla(c->gla_out_1xNxSxSxH, s->gla_states[layer].state,
                c->k_rope_1xN, v_view_1xN, c->q_rope_1xN, s->gla_slopes_1xN[layer],
                1, 1, S, H, scale);
    if (ok && XPROF_GLA_DEEP_ENABLED() && !XPROF_STAGE_END(XPROF_GLA_ATTN, layer, XPROF_T0)) ok = 0;

    ds4_metal_tensor_free(v_view_1xN);
    ds4_metal_tensor_free(k_view_1xN);
    ds4_metal_tensor_free(q_view_1xN);
    if (!ok) return 0;

    // 5. llama.cpp GLA epilogue: group RMS norm, layer output norm, sigmoid gate
    //    and output projection. First n_embd floats of gla_out are the activations.
    ds4_metal_tensor *gla_act_1xN = ds4_metal_tensor_view(c->gla_out_1xNxSxSxH, 0, act_bytes);
    if (!gla_act_1xN) return 0;
    ok = l26f_metal_gla_epilogue(c->gated_gla_1xN, gla_act_1xN, c->gate_out_1xN,
            m->map, m->size, wt_layer_out_N->abs_offset, n_embd, 4, m->rms_norm_eps);
    if (!ok) {
        ds4_metal_tensor_free(gla_act_1xN);
        return 0;
    }

    if (XPROF_GLA_DEEP_ENABLED() && !XPROF_STAGE_BEGIN(&XPROF_T0)) return 0;
    if (wt_out_NxN->type == 20) {
        ok = l26f_metal_matvec_iq4_nl_residual(out, c->gated_gla_1xN, inp,
                m->map, m->size, wt_out_NxN->abs_offset,
                wt_out_NxN->dim[0], wt_out_NxN->dim[1]);
    } else {
        ok = l26f_metal_matvec_quant(c->attn_proj_1xN, c->gated_gla_1xN,
                m->map, m->size, wt_out_NxN->abs_offset,
                wt_out_NxN->dim[0], wt_out_NxN->dim[1], wt_out_NxN->type, 1);
    }
    if (ok && XPROF_GLA_DEEP_ENABLED() && !XPROF_STAGE_END(XPROF_GLA_PROJ, layer, XPROF_T0)) ok = 0;
    ds4_metal_tensor_free(gla_act_1xN);
    if (!ok) return 0;

    // 6. Residual add: out = inp + attn_proj
    if (wt_out_NxN->type != 20) {
        if (!ds4_metal_add_tensor(out, inp, c->attn_proj_1xN, n_embd))
            return 0;
    }

    return 1;
}

static int l26f_dense_ffn(
        l26f_session *s,
        ds4_metal_tensor *inp, ds4_metal_tensor *out) {
    l26f_model *m = s->model;
    l26f_compute *c = &s->comp;
    const uint32_t n_embd = m->n_embd;
    const uint32_t n_ff = m->n_ff;

    l26f_tensor *wt_norm_N     = l26f_layer_tensor(m, 0, "ffn_norm.weight");
    l26f_tensor *wt_gate_NxF   = l26f_layer_tensor(m, 0, "ffn_gate.weight");
    l26f_tensor *wt_up_NxF     = l26f_layer_tensor(m, 0, "ffn_up.weight");
    l26f_tensor *wt_down_FxN   = l26f_layer_tensor(m, 0, "ffn_down.weight");
    if (!wt_norm_N || !wt_gate_NxF || !wt_up_NxF || !wt_down_FxN) {
        fprintf(stderr, "l26f: layer 0 missing dense FFN tensors\n");
        return 0;
    }

    // 1. RMS norm
    if (!ds4_metal_rms_norm_weight_tensor(c->ffn_normed_1xN, inp,
            m->map, m->size, wt_norm_N->abs_offset, n_embd, m->rms_norm_eps))
        return 0;

    if (!l26f_metal_matvec_quant(c->ffn_gate_1xF, c->ffn_normed_1xN,
            m->map, m->size, wt_gate_NxF->abs_offset,
            wt_gate_NxF->dim[0], wt_gate_NxF->dim[1], wt_gate_NxF->type, 1))
        return 0;

    if (!l26f_metal_matvec_quant(c->ffn_up_1xF, c->ffn_normed_1xN,
            m->map, m->size, wt_up_NxF->abs_offset,
            wt_up_NxF->dim[0], wt_up_NxF->dim[1], wt_up_NxF->type, 1))
        return 0;

    if (!ds4_metal_swiglu_tensor(c->ffn_mid_1xF, c->ffn_gate_1xF, c->ffn_up_1xF, n_ff, 0.0f, 1.0f))
        return 0;

    if (!l26f_metal_matvec_quant(c->ffn_down_1xN, c->ffn_mid_1xF,
            m->map, m->size, wt_down_FxN->abs_offset,
            wt_down_FxN->dim[0], wt_down_FxN->dim[1], wt_down_FxN->type, 1))
        return 0;

    if (!ds4_metal_add_tensor(out, inp, c->ffn_down_1xN, n_embd))
        return 0;

    return 1;
}

// ---- MoE FFN ----
//
// Routing algorithm (from llama.cpp build_moe_ffn):
// 1. logits = gate_inp × hidden  (F32 matvec, [4096]×[4096,256]→[256])
// 2. Add exp_probs_b bias
// 3. probs = softmax(logits)
// 4. Group scoring: 8 groups × 32 experts, score = sum(top-2 probs per group)
// 5. Select top-4 groups
// 6. Mask: only keep experts from top-4 groups
// 7. top-8 from masked pool
// 8. weights = probs[selected], normalize, scale by 2.5
// 9. Run 8 experts + shared expert, weighted sum

static void cpu_softmax(float *x, int n) {
    float maxv = x[0];
    for (int i = 1; i < n; i++) if (x[i] > maxv) maxv = x[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - maxv); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

static int cmp_float_desc(const void *a, const void *b) {
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    return (fa < fb) - (fa > fb);
}

static int l26f_moe_ffn(
        l26f_session *s, uint32_t layer,
        ds4_metal_tensor *inp, ds4_metal_tensor *out) {
    l26f_model *m = s->model;
    l26f_compute *c = &s->comp;
    const uint32_t n_embd = m->n_embd;
    const uint32_t n_ff_exp = 1024;
    const uint32_t n_expert = 256;
    const uint32_t n_groups = 8;
    const uint32_t n_exp_per_group = n_expert / n_groups;
    const float w_scale = 2.5f;

    l26f_tensor *wt_norm_N          = l26f_layer_tensor(m, layer, "ffn_norm.weight");
    l26f_tensor *wt_gate_inp_NxE    = l26f_layer_tensor(m, layer, "ffn_gate_inp.weight");
    l26f_tensor *wt_exp_b_1xE       = l26f_layer_tensor(m, layer, "exp_probs_b.bias");
    l26f_tensor *wt_gate_exps_NxMxE = l26f_layer_tensor(m, layer, "ffn_gate_exps.weight");
    l26f_tensor *wt_up_exps_NxMxE   = l26f_layer_tensor(m, layer, "ffn_up_exps.weight");
    l26f_tensor *wt_down_exps_MxNxE = l26f_layer_tensor(m, layer, "ffn_down_exps.weight");
    l26f_tensor *wt_gate_sh_NxM     = l26f_layer_tensor(m, layer, "ffn_gate_shexp.weight");
    l26f_tensor *wt_up_sh_NxM       = l26f_layer_tensor(m, layer, "ffn_up_shexp.weight");
    l26f_tensor *wt_down_sh_MxN     = l26f_layer_tensor(m, layer, "ffn_down_shexp.weight");
    if (!wt_norm_N || !wt_gate_inp_NxE || !wt_gate_exps_NxMxE || !wt_up_exps_NxMxE ||
        !wt_down_exps_MxNxE || !wt_gate_sh_NxM || !wt_up_sh_NxM || !wt_down_sh_MxN) {
        fprintf(stderr, "l26f: layer %u missing MoE tensors\n", layer);
        return 0;
    }

    // 1. RMS norm
    if (!ds4_metal_rms_norm_weight_tensor(c->ffn_normed_1xN, inp,
            m->map, m->size, wt_norm_N->abs_offset, n_embd, m->rms_norm_eps))
        return 0;

    // 2. GPU router matvec + routing
    if (!ds4_metal_matmul_f32_tensor(c->router_logits_1xE, m->map, m->size,
            wt_gate_inp_NxE->abs_offset, n_embd, n_expert, c->ffn_normed_1xN, 1))
        return 0;

    int32_t has_bias = (wt_exp_b_1xE != NULL) ? 1 : 0;
    uint64_t bias_off = has_bias ? wt_exp_b_1xE->abs_offset : 0;
    if (!l26f_metal_moe_route(c->router_logits_1xE, m->map, m->size,
            bias_off, has_bias,
            c->moe_sel_idx_K, c->moe_sel_wt_K,
            n_expert, n_groups, n_exp_per_group, 4, 8, w_scale))
        return 0;

    // 3. Read back 8 indices + 8 weights (64 bytes)
    int32_t selected_experts_K[8];
    float selected_weights_K[8];
    ds4_metal_end_commands();
    ds4_metal_tensor_read(c->moe_sel_idx_K, 0, selected_experts_K, 8 * sizeof(int32_t));
    ds4_metal_tensor_read(c->moe_sel_wt_K,  0, selected_weights_K, 8 * sizeof(float));
    ds4_metal_begin_commands();

    // 4. Expert byte strides
    const uint32_t gate_bs = l26f_types[wt_gate_exps_NxMxE->type].block_size;
    const uint32_t gate_ts = l26f_types[wt_gate_exps_NxMxE->type].type_size;
    const uint32_t up_bs   = l26f_types[wt_up_exps_NxMxE->type].block_size;
    const uint32_t up_ts   = l26f_types[wt_up_exps_NxMxE->type].type_size;
    const uint32_t down_bs = l26f_types[wt_down_exps_MxNxE->type].block_size;
    const uint32_t down_ts = l26f_types[wt_down_exps_MxNxE->type].type_size;
    const uint64_t gate_exp_bytes = (uint64_t)wt_gate_exps_NxMxE->dim[1] * (wt_gate_exps_NxMxE->dim[0] / gate_bs) * gate_ts;
    const uint64_t up_exp_bytes   = (uint64_t)wt_up_exps_NxMxE->dim[1]   * (wt_up_exps_NxMxE->dim[0]   / up_bs)   * up_ts;
    const uint64_t down_exp_bytes = (uint64_t)wt_down_exps_MxNxE->dim[1] * (wt_down_exps_MxNxE->dim[0] / down_bs) * down_ts;

    // 5. Run 8 selected experts
    ds4_metal_tensor_fill(c->moe_out_1xN, 0.0f);

    for (int i = 0; i < 8; i++) {
        int e = selected_experts_K[i];
        uint64_t gate_off = wt_gate_exps_NxMxE->abs_offset + (uint64_t)e * gate_exp_bytes;
        uint64_t up_off   = wt_up_exps_NxMxE->abs_offset   + (uint64_t)e * up_exp_bytes;
        uint64_t down_off = wt_down_exps_MxNxE->abs_offset + (uint64_t)e * down_exp_bytes;

        if (!l26f_metal_matvec_quant(c->ffn_gate_1xF, c->ffn_normed_1xN,
                m->map, m->size, gate_off,
                n_embd, n_ff_exp, wt_gate_exps_NxMxE->type, 1))
            return 0;

        if (!l26f_metal_matvec_quant(c->ffn_up_1xF, c->ffn_normed_1xN,
                m->map, m->size, up_off,
                n_embd, n_ff_exp, wt_up_exps_NxMxE->type, 1))
            return 0;

        if (!ds4_metal_swiglu_tensor(c->ffn_mid_1xF, c->ffn_gate_1xF, c->ffn_up_1xF, n_ff_exp, 0.0f, 1.0f))
            return 0;

        if (!l26f_metal_matvec_quant(c->ffn_down_1xN, c->ffn_mid_1xF,
                m->map, m->size, down_off,
                n_ff_exp, n_embd, wt_down_exps_MxNxE->type, 1))
            return 0;

        if (!l26f_metal_axpy(c->moe_out_1xN, c->ffn_down_1xN, selected_weights_K[i], n_embd))
            return 0;
    }

    // 6. Shared expert
    if (!l26f_metal_matvec_quant(c->ffn_gate_1xF, c->ffn_normed_1xN,
            m->map, m->size, wt_gate_sh_NxM->abs_offset,
            wt_gate_sh_NxM->dim[0], wt_gate_sh_NxM->dim[1], wt_gate_sh_NxM->type, 1))
        return 0;
    if (!l26f_metal_matvec_quant(c->ffn_up_1xF, c->ffn_normed_1xN,
            m->map, m->size, wt_up_sh_NxM->abs_offset,
            wt_up_sh_NxM->dim[0], wt_up_sh_NxM->dim[1], wt_up_sh_NxM->type, 1))
        return 0;
    if (!ds4_metal_swiglu_tensor(c->ffn_mid_1xF, c->ffn_gate_1xF, c->ffn_up_1xF, n_ff_exp, 0.0f, 1.0f))
        return 0;
    if (!l26f_metal_matvec_quant(c->shexp_out_1xN, c->ffn_mid_1xF,
            m->map, m->size, wt_down_sh_MxN->abs_offset,
            wt_down_sh_MxN->dim[0], wt_down_sh_MxN->dim[1], wt_down_sh_MxN->type, 1))
        return 0;

    if (!l26f_metal_axpy(c->moe_out_1xN, c->shexp_out_1xN, 1.0f, n_embd))
        return 0;

    // 7. Residual add
    if (!ds4_metal_add_tensor(out, inp, c->moe_out_1xN, n_embd))
        return 0;

    return 1;
}

// ---- Production Fused MoE (no breaks, no CPU readback) ----
//
// Runs the entire MoE on GPU: routing → gather offsets → fused matvecs →
// swiglu → accumulate + shared expert. Zero command buffer breaks.
// Enable with: env L26F_FUSED_MOE=1

static int l26f_use_fused_moe(void) {
    static int initialized = 0;
    static int use_fused = 0;
    if (!initialized) {
        use_fused = getenv("L26F_FUSED_MOE") != NULL;
        initialized = 1;
    }
    return use_fused;
}

static int l26f_use_hybrid_moe(void) {
    static int initialized = 0;
    static int use_hybrid = 0;
    if (!initialized) {
        use_hybrid = getenv("L26F_HYBRID_MOE") != NULL;
        initialized = 1;
    }
    return use_hybrid;
}

static int l26f_moe_ffn_hybrid(
        l26f_session *s, uint32_t layer,
        ds4_metal_tensor *inp, ds4_metal_tensor *out) {
    l26f_model *m = s->model;
    l26f_compute *c = &s->comp;
    const uint32_t n_embd = m->n_embd;
    const uint32_t n_ff_exp = 1024;
    const uint32_t n_expert = 256;
    const uint32_t n_groups = 8;
    const uint32_t n_exp_per_group = n_expert / n_groups;
    const float w_scale = 2.5f;
    const int K = 8;

    l26f_tensor *wt_norm_N          = l26f_layer_tensor(m, layer, "ffn_norm.weight");
    l26f_tensor *wt_gate_inp_NxE    = l26f_layer_tensor(m, layer, "ffn_gate_inp.weight");
    l26f_tensor *wt_exp_b_1xE       = l26f_layer_tensor(m, layer, "exp_probs_b.bias");
    l26f_tensor *wt_gate_exps_NxMxE = l26f_layer_tensor(m, layer, "ffn_gate_exps.weight");
    l26f_tensor *wt_up_exps_NxMxE   = l26f_layer_tensor(m, layer, "ffn_up_exps.weight");
    l26f_tensor *wt_down_exps_MxNxE = l26f_layer_tensor(m, layer, "ffn_down_exps.weight");
    l26f_tensor *wt_gate_sh_NxM     = l26f_layer_tensor(m, layer, "ffn_gate_shexp.weight");
    l26f_tensor *wt_up_sh_NxM       = l26f_layer_tensor(m, layer, "ffn_up_shexp.weight");
    l26f_tensor *wt_down_sh_MxN     = l26f_layer_tensor(m, layer, "ffn_down_shexp.weight");
    if (!wt_norm_N || !wt_gate_inp_NxE || !wt_gate_exps_NxMxE || !wt_up_exps_NxMxE ||
        !wt_down_exps_MxNxE || !wt_gate_sh_NxM || !wt_up_sh_NxM || !wt_down_sh_MxN) {
        fprintf(stderr, "l26f: layer %u missing MoE tensors\n", layer);
        return 0;
    }

    if (!ds4_metal_rms_norm_weight_tensor(c->ffn_normed_1xN, inp,
            m->map, m->size, wt_norm_N->abs_offset, n_embd, m->rms_norm_eps))
        return 0;

    if (!ds4_metal_matmul_f32_tensor(c->router_logits_1xE, m->map, m->size,
            wt_gate_inp_NxE->abs_offset, n_embd, n_expert, c->ffn_normed_1xN, 1))
        return 0;

    int32_t has_bias = (wt_exp_b_1xE != NULL) ? 1 : 0;
    uint64_t bias_off = has_bias ? wt_exp_b_1xE->abs_offset : 0;
    if (!l26f_metal_moe_route(c->router_logits_1xE, m->map, m->size,
            bias_off, has_bias,
            c->moe_sel_idx_K, c->moe_sel_wt_K,
            n_expert, n_groups, n_exp_per_group, 4, K, w_scale))
        return 0;

    int32_t selected_experts_K[8];
    float selected_weights_K[8];
    ds4_metal_end_commands();
    ds4_metal_tensor_read(c->moe_sel_idx_K, 0, selected_experts_K, K * sizeof(int32_t));
    ds4_metal_tensor_read(c->moe_sel_wt_K,  0, selected_weights_K, K * sizeof(float));
    ds4_metal_begin_commands();

    const uint32_t gate_bs = l26f_types[wt_gate_exps_NxMxE->type].block_size;
    const uint32_t gate_ts = l26f_types[wt_gate_exps_NxMxE->type].type_size;
    const uint64_t gate_exp_bytes = (uint64_t)wt_gate_exps_NxMxE->dim[1] * ((uint64_t)wt_gate_exps_NxMxE->dim[0] / gate_bs) * gate_ts;
    const uint64_t gate_total = (uint64_t)n_expert * gate_exp_bytes;

    const uint32_t up_bs = l26f_types[wt_up_exps_NxMxE->type].block_size;
    const uint32_t up_ts = l26f_types[wt_up_exps_NxMxE->type].type_size;
    const uint64_t up_exp_bytes = (uint64_t)wt_up_exps_NxMxE->dim[1] * ((uint64_t)wt_up_exps_NxMxE->dim[0] / up_bs) * up_ts;
    const uint64_t up_total = (uint64_t)n_expert * up_exp_bytes;

    const uint32_t down_bs = l26f_types[wt_down_exps_MxNxE->type].block_size;
    const uint32_t down_ts = l26f_types[wt_down_exps_MxNxE->type].type_size;
    const uint64_t down_exp_bytes = (uint64_t)wt_down_exps_MxNxE->dim[1] * ((uint64_t)wt_down_exps_MxNxE->dim[0] / down_bs) * down_ts;

    uint64_t gate_off_K[8], up_off_K[8];
    for (int i = 0; i < K; i++) {
        const uint64_t e = (uint64_t)selected_experts_K[i];
        gate_off_K[i] = e * gate_exp_bytes;
        up_off_K[i] = e * up_exp_bytes;
    }
    ds4_metal_tensor_write(c->moe_gate_off_K, 0, gate_off_K, sizeof(gate_off_K));
    ds4_metal_tensor_write(c->moe_up_off_K, 0, up_off_K, sizeof(up_off_K));

    if (!l26f_metal_fused_moe_iq4nl_gate_up(
            c->moe_expert_gate_8xM, c->moe_expert_up_8xM,
            m->map, m->size,
            c->moe_gate_off_K, c->moe_up_off_K,
            wt_gate_exps_NxMxE->abs_offset, gate_total,
            wt_up_exps_NxMxE->abs_offset, up_total,
            K, n_embd, n_ff_exp, c->ffn_normed_1xN))
        return 0;

    if (!l26f_metal_fused_swiglu(c->moe_expert_mid_8xM,
            c->moe_expert_gate_8xM, c->moe_expert_up_8xM,
            K, n_ff_exp))
        return 0;

    ds4_metal_tensor_fill(c->moe_out_1xN, 0.0f);
    const uint64_t ffn_bytes = (uint64_t)n_ff_exp * sizeof(float);
    for (int i = 0; i < K; i++) {
        const int e = selected_experts_K[i];
        const uint64_t down_off = wt_down_exps_MxNxE->abs_offset + (uint64_t)e * down_exp_bytes;
        ds4_metal_tensor *mid_i_1xM = ds4_metal_tensor_view(c->moe_expert_mid_8xM, (uint64_t)i * ffn_bytes, ffn_bytes);
        if (!mid_i_1xM) return 0;

        int ok = l26f_metal_matvec_quant(c->ffn_down_1xN, mid_i_1xM,
                m->map, m->size, down_off,
                n_ff_exp, n_embd, wt_down_exps_MxNxE->type, 1);
        ds4_metal_tensor_free(mid_i_1xM);
        if (!ok) return 0;

        if (!l26f_metal_axpy(c->moe_out_1xN, c->ffn_down_1xN, selected_weights_K[i], n_embd))
            return 0;
    }

    if (!l26f_metal_matvec_quant(c->ffn_gate_1xF, c->ffn_normed_1xN,
            m->map, m->size, wt_gate_sh_NxM->abs_offset,
            wt_gate_sh_NxM->dim[0], wt_gate_sh_NxM->dim[1], wt_gate_sh_NxM->type, 1))
        return 0;
    if (!l26f_metal_matvec_quant(c->ffn_up_1xF, c->ffn_normed_1xN,
            m->map, m->size, wt_up_sh_NxM->abs_offset,
            wt_up_sh_NxM->dim[0], wt_up_sh_NxM->dim[1], wt_up_sh_NxM->type, 1))
        return 0;
    if (!ds4_metal_swiglu_tensor(c->ffn_mid_1xF, c->ffn_gate_1xF, c->ffn_up_1xF, n_ff_exp, 0.0f, 1.0f))
        return 0;
    if (!l26f_metal_matvec_quant(c->shexp_out_1xN, c->ffn_mid_1xF,
            m->map, m->size, wt_down_sh_MxN->abs_offset,
            wt_down_sh_MxN->dim[0], wt_down_sh_MxN->dim[1], wt_down_sh_MxN->type, 1))
        return 0;
    if (!l26f_metal_axpy(c->moe_out_1xN, c->shexp_out_1xN, 1.0f, n_embd))
        return 0;

    if (!ds4_metal_add_tensor(out, inp, c->moe_out_1xN, n_embd))
        return 0;

    return 1;
}

static int l26f_moe_ffn_fused_prod(
        l26f_session *s, uint32_t layer,
        ds4_metal_tensor *inp, ds4_metal_tensor *out) {
    l26f_model *m = s->model;
    l26f_compute *c = &s->comp;
    const uint32_t n_embd = m->n_embd;
    const uint32_t n_ff_exp = 1024;
    const uint32_t n_expert = 256;
    const uint32_t n_groups = 8;
    const uint32_t n_exp_per_group = n_expert / n_groups;
    const float w_scale = 2.5f;
    const int K = 8;

    l26f_tensor *wt_norm_N          = l26f_layer_tensor(m, layer, "ffn_norm.weight");
    l26f_tensor *wt_gate_inp_NxE    = l26f_layer_tensor(m, layer, "ffn_gate_inp.weight");
    l26f_tensor *wt_exp_b_1xE       = l26f_layer_tensor(m, layer, "exp_probs_b.bias");
    l26f_tensor *wt_gate_exps_NxMxE = l26f_layer_tensor(m, layer, "ffn_gate_exps.weight");
    l26f_tensor *wt_up_exps_NxMxE   = l26f_layer_tensor(m, layer, "ffn_up_exps.weight");
    l26f_tensor *wt_down_exps_MxNxE = l26f_layer_tensor(m, layer, "ffn_down_exps.weight");
    l26f_tensor *wt_gate_sh_NxM     = l26f_layer_tensor(m, layer, "ffn_gate_shexp.weight");
    l26f_tensor *wt_up_sh_NxM       = l26f_layer_tensor(m, layer, "ffn_up_shexp.weight");
    l26f_tensor *wt_down_sh_MxN     = l26f_layer_tensor(m, layer, "ffn_down_shexp.weight");
    if (!wt_norm_N || !wt_gate_inp_NxE || !wt_gate_exps_NxMxE || !wt_up_exps_NxMxE ||
        !wt_down_exps_MxNxE || !wt_gate_sh_NxM || !wt_up_sh_NxM || !wt_down_sh_MxN) {
        fprintf(stderr, "l26f: layer %u missing MoE tensors\n", layer);
        return 0;
    }

    double XPROF_MOE_T0 = 0.0;
    if (!XPROF_MOE_STAGE_BEGIN(&XPROF_MOE_T0)) return 0;
    if (!ds4_metal_rms_norm_weight_tensor(c->ffn_normed_1xN, inp,
            m->map, m->size, wt_norm_N->abs_offset, n_embd, m->rms_norm_eps))
        return 0;
    if (!XPROF_MOE_STAGE_END(XPROF_MOE_NORM, layer, XPROF_MOE_T0)) return 0;

    if (!XPROF_MOE_STAGE_BEGIN(&XPROF_MOE_T0)) return 0;
    if (!ds4_metal_matmul_f32_tensor(c->router_logits_1xE, m->map, m->size,
            wt_gate_inp_NxE->abs_offset, n_embd, n_expert, c->ffn_normed_1xN, 1))
        return 0;
    if (!XPROF_MOE_STAGE_END(XPROF_MOE_ROUTER, layer, XPROF_MOE_T0)) return 0;

    int32_t has_bias = (wt_exp_b_1xE != NULL) ? 1 : 0;
    uint64_t bias_off = has_bias ? wt_exp_b_1xE->abs_offset : 0;
    if (!XPROF_MOE_STAGE_BEGIN(&XPROF_MOE_T0)) return 0;
    if (!l26f_metal_moe_route_offsets(c->router_logits_1xE, m->map, m->size,
            bias_off, has_bias,
            c->moe_sel_idx_K, c->moe_sel_wt_K,
            s->moe_gate_all_off_1xE[layer], s->moe_up_all_off_1xE[layer], s->moe_down_all_off_1xE[layer],
            c->moe_gate_off_K, c->moe_up_off_K, c->moe_down_off_K,
            n_expert, n_groups, n_exp_per_group, 4, K, w_scale))
        return 0;
    if (!XPROF_MOE_STAGE_END(XPROF_MOE_ROUTE, layer, XPROF_MOE_T0)) return 0;

    const uint32_t gate_bs = l26f_types[wt_gate_exps_NxMxE->type].block_size;
    const uint32_t gate_ts = l26f_types[wt_gate_exps_NxMxE->type].type_size;
    const uint64_t gate_exp_bytes = (uint64_t)wt_gate_exps_NxMxE->dim[1] * ((uint64_t)wt_gate_exps_NxMxE->dim[0] / gate_bs) * gate_ts;
    const uint64_t gate_total = (uint64_t)n_expert * gate_exp_bytes;

    const uint32_t up_bs = l26f_types[wt_up_exps_NxMxE->type].block_size;
    const uint32_t up_ts = l26f_types[wt_up_exps_NxMxE->type].type_size;
    const uint64_t up_exp_bytes = (uint64_t)wt_up_exps_NxMxE->dim[1] * ((uint64_t)wt_up_exps_NxMxE->dim[0] / up_bs) * up_ts;
    const uint64_t up_total = (uint64_t)n_expert * up_exp_bytes;

    const uint32_t down_bs = l26f_types[wt_down_exps_MxNxE->type].block_size;
    const uint32_t down_ts = l26f_types[wt_down_exps_MxNxE->type].type_size;
    const uint64_t down_exp_bytes = (uint64_t)wt_down_exps_MxNxE->dim[1] * ((uint64_t)wt_down_exps_MxNxE->dim[0] / down_bs) * down_ts;
    const uint64_t down_total = (uint64_t)n_expert * down_exp_bytes;

    // Gather 8 selected experts into contiguous cache for better locality
    static int use_cache = -1;
    if (use_cache < 0) use_cache = getenv("L26F_EXPERT_CACHE") != NULL;

    if (use_cache) {
        if (!XPROF_MOE_STAGE_BEGIN(&XPROF_MOE_T0)) return 0;
        if (!l26f_metal_gather_experts(c->moe_gate_cache_8xMxN, m->map, m->size,
                wt_gate_exps_NxMxE->abs_offset, gate_total,
                gate_exp_bytes, gate_exp_bytes,
                c->moe_sel_idx_K, K))
            return 0;
        if (!l26f_metal_gather_experts(c->moe_up_cache_8xMxN, m->map, m->size,
                wt_up_exps_NxMxE->abs_offset, up_total,
                up_exp_bytes, up_exp_bytes,
                c->moe_sel_idx_K, K))
            return 0;
        if (!l26f_metal_gather_experts(c->moe_down_cache_8xNxM, m->map, m->size,
                wt_down_exps_MxNxE->abs_offset, down_total,
                down_exp_bytes, down_exp_bytes,
                c->moe_sel_idx_K, K))
            return 0;
        if (!XPROF_MOE_STAGE_END(XPROF_MOE_OFFSETS, layer, XPROF_MOE_T0)) return 0;

        // Offsets into the contiguous K-expert cache are fixed and initialized
        // once in l26f_session_init. Do not write CPU data during the batch.

        if (!XPROF_MOE_STAGE_BEGIN(&XPROF_MOE_T0)) return 0;
        if (!l26f_metal_fused_moe_iq4nl_gate_up_swiglu_cached(
                c->moe_expert_mid_8xM,
                c->moe_gate_cache_8xMxN, c->moe_up_cache_8xMxN,
                s->moe_gate_cache_off_K[layer], s->moe_up_cache_off_K[layer],
                K, n_embd, n_ff_exp, c->ffn_normed_1xN))
            return 0;
        if (!XPROF_MOE_STAGE_END(XPROF_MOE_GATE_UP, layer, XPROF_MOE_T0)) return 0;

        if (!XPROF_MOE_STAGE_BEGIN(&XPROF_MOE_T0)) return 0;
        if (wt_down_exps_MxNxE->type == 13) {
            if (!l26f_metal_fused_moe_q5k_cached(c->moe_expert_down_8xN,
                    c->moe_down_cache_8xNxM, s->moe_down_cache_off_K[layer],
                    K, n_ff_exp, n_embd, c->moe_expert_mid_8xM))
                return 0;
        } else {
            if (!l26f_metal_fused_moe_iq4nl_cached(c->moe_expert_down_8xN,
                    c->moe_down_cache_8xNxM, s->moe_down_cache_off_K[layer],
                    K, n_ff_exp, n_embd, c->moe_expert_mid_8xM, 1))
                return 0;
        }
        if (!XPROF_MOE_STAGE_END(XPROF_MOE_DOWN, layer, XPROF_MOE_T0)) return 0;
    } else {
        if (!XPROF_MOE_STAGE_BEGIN(&XPROF_MOE_T0)) return 0;
        if (!l26f_metal_fused_moe_iq4nl_gate_up_swiglu(
                c->moe_expert_mid_8xM,
                m->map, m->size,
                c->moe_gate_off_K, c->moe_up_off_K,
                wt_gate_exps_NxMxE->abs_offset, gate_total,
                wt_up_exps_NxMxE->abs_offset, up_total,
                K, n_embd, n_ff_exp, c->ffn_normed_1xN))
            return 0;
        if (!XPROF_MOE_STAGE_END(XPROF_MOE_GATE_UP, layer, XPROF_MOE_T0)) return 0;

        if (!XPROF_MOE_STAGE_BEGIN(&XPROF_MOE_T0)) return 0;
        if (wt_down_exps_MxNxE->type == 13) {
            if (!l26f_metal_fused_moe_q5k(c->moe_expert_down_8xN,
                    m->map, m->size, c->moe_down_off_K,
                    wt_down_exps_MxNxE->abs_offset, down_total,
                    K, n_ff_exp, n_embd, c->moe_expert_mid_8xM))
                return 0;
        } else {
            if (!l26f_metal_fused_moe_iq4nl(c->moe_expert_down_8xN,
                    m->map, m->size, c->moe_down_off_K,
                    wt_down_exps_MxNxE->abs_offset, down_total,
                    K, n_ff_exp, n_embd, c->moe_expert_mid_8xM, 1))
                return 0;
        }
        if (!XPROF_MOE_STAGE_END(XPROF_MOE_DOWN, layer, XPROF_MOE_T0)) return 0;
    }

    if (!XPROF_MOE_STAGE_BEGIN(&XPROF_MOE_T0)) return 0;
    {
        const uint32_t sh_bs = l26f_types[wt_gate_sh_NxM->type].block_size;
        const uint32_t sh_ts = l26f_types[wt_gate_sh_NxM->type].type_size;
        const uint64_t sh_gate_bytes = (uint64_t)wt_gate_sh_NxM->dim[1] * ((uint64_t)wt_gate_sh_NxM->dim[0] / sh_bs) * sh_ts;
        const uint64_t sh_up_bytes   = (uint64_t)wt_up_sh_NxM->dim[1] * ((uint64_t)wt_up_sh_NxM->dim[0] / sh_bs) * sh_ts;

        if (wt_gate_sh_NxM->type == 20 && wt_up_sh_NxM->type == 20) {
            if (!l26f_metal_shared_gate_up_swiglu_iq4nl(c->ffn_mid_1xF,
                    m->map, m->size,
                    wt_gate_sh_NxM->abs_offset, sh_gate_bytes,
                    wt_up_sh_NxM->abs_offset, sh_up_bytes,
                    wt_gate_sh_NxM->dim[0], wt_gate_sh_NxM->dim[1],
                    c->ffn_normed_1xN))
                return 0;
        } else {
            if (!l26f_metal_matvec_quant(c->ffn_gate_1xF, c->ffn_normed_1xN,
                    m->map, m->size, wt_gate_sh_NxM->abs_offset,
                    wt_gate_sh_NxM->dim[0], wt_gate_sh_NxM->dim[1], wt_gate_sh_NxM->type, 1))
                return 0;
            if (!l26f_metal_matvec_quant(c->ffn_up_1xF, c->ffn_normed_1xN,
                    m->map, m->size, wt_up_sh_NxM->abs_offset,
                    wt_up_sh_NxM->dim[0], wt_up_sh_NxM->dim[1], wt_up_sh_NxM->type, 1))
                return 0;
            if (!ds4_metal_swiglu_tensor(c->ffn_mid_1xF, c->ffn_gate_1xF, c->ffn_up_1xF, n_ff_exp, 0.0f, 1.0f))
                return 0;
        }
    }

    if (!l26f_metal_matvec_quant(c->shexp_out_1xN, c->ffn_mid_1xF,
            m->map, m->size, wt_down_sh_MxN->abs_offset,
            wt_down_sh_MxN->dim[0], wt_down_sh_MxN->dim[1], wt_down_sh_MxN->type, 1))
        return 0;
    if (!XPROF_MOE_STAGE_END(XPROF_MOE_SHARED, layer, XPROF_MOE_T0)) return 0;

    if (!XPROF_MOE_STAGE_BEGIN(&XPROF_MOE_T0)) return 0;
    if (!l26f_metal_fused_accum_residual(out, inp,
            c->moe_expert_down_8xN, c->moe_sel_wt_K, c->shexp_out_1xN,
            K, n_embd))
        return 0;
    if (!XPROF_MOE_STAGE_END(XPROF_MOE_ACCUM, layer, XPROF_MOE_T0)) return 0;

    return 1;
}

#ifdef L26F_PREFILL_DEBUG
static void L26F_NAN_CHECK(const char *label, const float *data, uint32_t n, const char *ctx) {
    int first_nan = -1;
    for (uint32_t i = 0; i < n; i++) {
        if (isnan(data[i]) || isinf(data[i])) {
            first_nan = (int)i;
            break;
        }
    }
    if (first_nan >= 0) {
        fprintf(stderr, "  ** NaN/INF FIRST at %s [%s] elem %d val=%.6f (n=%u)\n",
                label, ctx, first_nan, data[first_nan], n);
    }
}

static void L26F_PREFILL_CKPT(const char *label, ds4_metal_tensor *t, uint64_t bytes,
                               uint32_t il, int t_tok) {
    int nans;
    float sum = l26f_tensor_checksum(t, bytes, &nans);
    fprintf(stderr, "  PREFILL L%u:t%d %-22s sum=%12.2f nans=%d\n", il, t_tok, label, sum, nans);
    if (nans > 0) {
        float *data = (float *)malloc(bytes);
        if (data) {
            ds4_metal_tensor_read(t, 0, data, bytes);
            L26F_NAN_CHECK(label, data, (uint32_t)(bytes / sizeof(float)),
                           il < 100 ? "layer" : "embed");
            free(data);
        }
    }
}
#define L26F_PREFILL_CKPT(label, t, bytes, il, tt) L26F_PREFILL_CKPT(label, t, bytes, il, tt)
#define L26F_NAN_CHECK(label, data, n, ctx) L26F_NAN_CHECK(label, data, n, ctx)
#define L26F_PREFILL_LOG(fmt, ...) fprintf(stderr, fmt "\n", ##__VA_ARGS__)
#else
#define L26F_PREFILL_CKPT(label, t, bytes, il, tt) ((void)0)
#define L26F_NAN_CHECK(label, data, n, ctx) ((void)0)
#define L26F_PREFILL_LOG(fmt, ...) ((void)0)
#endif

// =========================================================================
// Batch MoE FFN for prefill — uses mat-mat SGEMM kernel
// Processes T tokens through MoE FFN for one layer using expert indirection.
// inp_TxN: [T, N] input hidden states (post-attention)
// out_TxN: [T, N] output hidden states
// =========================================================================

static int l26f_moe_ffn_batch(
        l26f_session *s, uint32_t layer,
        ds4_metal_tensor *inp_TxN,
        ds4_metal_tensor *out_TxN,
        uint32_t T)
{
    l26f_model *m = s->model;
    l26f_prefill_compute *p = s->prefill;
    const uint32_t n_embd = m->n_embd;
    const uint32_t n_ff_exp = 1024;
    const uint32_t n_expert = 256;
    const uint32_t n_groups = 8;
    const uint32_t n_exp_per_group = n_expert / n_groups;
    const float w_scale = 2.5f;
    const int K = 8;

    l26f_tensor *wt_norm_N          = l26f_layer_tensor(m, layer, "ffn_norm.weight");
    l26f_tensor *wt_gate_inp_NxE    = l26f_layer_tensor(m, layer, "ffn_gate_inp.weight");
    l26f_tensor *wt_exp_b_1xE       = l26f_layer_tensor(m, layer, "exp_probs_b.bias");
    l26f_tensor *wt_gate_exps_NxMxE = l26f_layer_tensor(m, layer, "ffn_gate_exps.weight");
    l26f_tensor *wt_up_exps_NxMxE   = l26f_layer_tensor(m, layer, "ffn_up_exps.weight");
    l26f_tensor *wt_down_exps_MxNxE = l26f_layer_tensor(m, layer, "ffn_down_exps.weight");
    l26f_tensor *wt_gate_sh_NxM     = l26f_layer_tensor(m, layer, "ffn_gate_shexp.weight");
    l26f_tensor *wt_up_sh_NxM       = l26f_layer_tensor(m, layer, "ffn_up_shexp.weight");
    l26f_tensor *wt_down_sh_MxN     = l26f_layer_tensor(m, layer, "ffn_down_shexp.weight");
    if (!wt_norm_N || !wt_gate_inp_NxE || !wt_gate_exps_NxMxE || !wt_up_exps_NxMxE ||
        !wt_down_exps_MxNxE || !wt_gate_sh_NxM || !wt_up_sh_NxM || !wt_down_sh_MxN) {
        fprintf(stderr, "l26f: layer %u missing MoE tensors\n", layer);
        return 0;
    }

    if (!ds4_metal_rms_norm_weight_rows_tensor(p->normed_TxN, inp_TxN,
            m->map, m->size, wt_norm_N->abs_offset, n_embd, T, m->rms_norm_eps)) {
        fprintf(stderr, "  [batch_moe L%u] rms_norm failed\n", layer);
        return 0;
    }
    if (layer <= 1) L26F_PREFILL_CKPT("normed_TxN", p->normed_TxN, (uint64_t)T * n_embd * sizeof(float), layer, -1);

    {
        ds4_metal_tensor *router_row_1xE = ds4_metal_tensor_view(p->router_logits_TxE,
                (uint64_t)0, (uint64_t)n_expert * sizeof(float));
        if (!router_row_1xE) { fprintf(stderr, "  [batch_moe L%u] router view failed\n", layer); return 0; }
        int router_ok = 1;
        for (uint32_t t = 0; t < T; t++) {
            ds4_metal_tensor *normed_row_1xN = ds4_metal_tensor_view(p->normed_TxN,
                (uint64_t)t * n_embd * sizeof(float), (uint64_t)n_embd * sizeof(float));
            if (!normed_row_1xN) { router_ok = 0; break; }
            ds4_metal_tensor *logits_row_1xE = ds4_metal_tensor_view(p->router_logits_TxE,
                (uint64_t)t * n_expert * sizeof(float), (uint64_t)n_expert * sizeof(float));
            if (!logits_row_1xE) { ds4_metal_tensor_free(normed_row_1xN); router_ok = 0; break; }
            if (!ds4_metal_matmul_f32_tensor(logits_row_1xE, m->map, m->size,
                    wt_gate_inp_NxE->abs_offset, n_embd, n_expert, normed_row_1xN, 1)) {
                ds4_metal_tensor_free(normed_row_1xN);
                ds4_metal_tensor_free(logits_row_1xE);
                router_ok = 0;
                break;
            }
            ds4_metal_tensor_free(normed_row_1xN);
            ds4_metal_tensor_free(logits_row_1xE);
        }
        ds4_metal_tensor_free(router_row_1xE);
        if (!router_ok) {
            fprintf(stderr, "  [batch_moe L%u] router matmul failed at token %u\n", layer, T);
            return 0;
        }
    }
    if (layer <= 1) L26F_PREFILL_CKPT("router_logits_TxE", p->router_logits_TxE, (uint64_t)T * n_expert * sizeof(float), layer, -1);

    int32_t has_bias = (wt_exp_b_1xE != NULL) ? 1 : 0;
    uint64_t bias_off = has_bias ? wt_exp_b_1xE->abs_offset : 0;
    if (!l26f_metal_moe_route_batch(p->router_logits_TxE, m->map, m->size,
            bias_off, has_bias,
            p->sel_idx_TxK, p->sel_wt_TxK,
            n_expert, n_groups, n_exp_per_group, 4, K, w_scale, T)) {
        fprintf(stderr, "  [batch_moe L%u] route_batch failed\n", layer);
        return 0;
    }
    if (layer <= 1) {
        L26F_PREFILL_CKPT("sel_idx_TxK", p->sel_idx_TxK, (uint64_t)T * K * sizeof(int32_t), layer, -1);
        L26F_PREFILL_CKPT("sel_wt_TxK", p->sel_wt_TxK, (uint64_t)T * K * sizeof(float), layer, -1);
    }

    if (!l26f_metal_mul_mm_id_map0(p->sel_idx_TxK, p->tokens_per_expert_E,
            p->ids_buffer_ExT, n_expert, T)) {
        fprintf(stderr, "  [batch_moe L%u] map0 failed\n", layer);
        return 0;
    }
    if (layer <= 1) L26F_PREFILL_CKPT("tokens_per_expert_E", p->tokens_per_expert_E, (uint64_t)n_expert * sizeof(uint32_t), layer, -1);

    const uint32_t gate_bs = l26f_types[wt_gate_exps_NxMxE->type].block_size;
    const uint32_t gate_ts = l26f_types[wt_gate_exps_NxMxE->type].type_size;
    const uint64_t gate_exp_bytes = (uint64_t)wt_gate_exps_NxMxE->dim[1] * ((uint64_t)wt_gate_exps_NxMxE->dim[0] / gate_bs) * gate_ts;
    const uint64_t gate_total = (uint64_t)n_expert * gate_exp_bytes;

    const uint32_t up_bs = l26f_types[wt_up_exps_NxMxE->type].block_size;
    const uint32_t up_ts = l26f_types[wt_up_exps_NxMxE->type].type_size;
    const uint64_t up_exp_bytes = (uint64_t)wt_up_exps_NxMxE->dim[1] * ((uint64_t)wt_up_exps_NxMxE->dim[0] / up_bs) * up_ts;
    const uint64_t up_total = (uint64_t)n_expert * up_exp_bytes;

    const uint32_t down_bs = l26f_types[wt_down_exps_MxNxE->type].block_size;
    const uint32_t down_ts = l26f_types[wt_down_exps_MxNxE->type].type_size;
    const uint64_t down_exp_bytes = (uint64_t)wt_down_exps_MxNxE->dim[1] * ((uint64_t)wt_down_exps_MxNxE->dim[0] / down_bs) * down_ts;
    const uint64_t down_total = (uint64_t)n_expert * down_exp_bytes;

    if (!l26f_metal_mul_mm_id_iq4nl(p->gate_out_TxKxM, m->map, m->size,
            wt_gate_exps_NxMxE->abs_offset, gate_total,
            n_expert, n_embd, n_ff_exp,
            p->normed_TxN, p->tokens_per_expert_E, p->ids_buffer_ExT, T)) {
        fprintf(stderr, "  [batch_moe L%u] gate mul_mm_id failed\n", layer);
        return 0;
    }
    if (layer <= 1) L26F_PREFILL_CKPT("gate_out_TxKxM", p->gate_out_TxKxM, (uint64_t)T * K * n_ff_exp * sizeof(float), layer, -1);

    if (!l26f_metal_mul_mm_id_iq4nl(p->up_out_TxKxM, m->map, m->size,
            wt_up_exps_NxMxE->abs_offset, up_total,
            n_expert, n_embd, n_ff_exp,
            p->normed_TxN, p->tokens_per_expert_E, p->ids_buffer_ExT, T)) {
        fprintf(stderr, "  [batch_moe L%u] up mul_mm_id failed\n", layer);
        return 0;
    }
    if (layer <= 1) L26F_PREFILL_CKPT("up_out_TxKxM", p->up_out_TxKxM, (uint64_t)T * K * n_ff_exp * sizeof(float), layer, -1);

    if (!l26f_metal_batch_swiglu(p->mid_TxKxM, p->gate_out_TxKxM, p->up_out_TxKxM, T * K * n_ff_exp)) {
        fprintf(stderr, "  [batch_moe L%u] batch_swiglu failed\n", layer);
        return 0;
    }
    if (layer <= 1) L26F_PREFILL_CKPT("mid_TxKxM", p->mid_TxKxM, (uint64_t)T * K * n_ff_exp * sizeof(float), layer, -1);

    if (wt_down_exps_MxNxE->type == 20) {
        if (!l26f_metal_mul_mm_id_iq4nl(p->down_out_TxKxN, m->map, m->size,
                wt_down_exps_MxNxE->abs_offset, down_total,
                n_expert, n_ff_exp, n_embd,
                p->mid_TxKxM, p->tokens_per_expert_E, p->ids_buffer_ExT, T)) {
            fprintf(stderr, "  [batch_moe L%u] down mul_mm_id iq4nl failed\n", layer);
            return 0;
        }
    } else if (wt_down_exps_MxNxE->type == 13) {
        if (!l26f_metal_mul_mm_id_q5k(p->down_out_TxKxN, m->map, m->size,
                wt_down_exps_MxNxE->abs_offset, down_total,
                n_expert, n_ff_exp, n_embd,
                p->mid_TxKxM, p->tokens_per_expert_E, p->ids_buffer_ExT, T)) {
            fprintf(stderr, "  [batch_moe L%u] down mul_mm_id q5k failed\n", layer);
            return 0;
        }
    } else {
        fprintf(stderr, "  [batch_moe L%u] unsupported down type %u\n", layer, wt_down_exps_MxNxE->type);
        return 0;
    }
    if (layer <= 1) L26F_PREFILL_CKPT("down_out_TxKxN", p->down_out_TxKxN, (uint64_t)T * K * n_embd * sizeof(float), layer, -1);

    for (uint32_t t = 0; t < T; t++) {
        ds4_metal_tensor *normed_row_1xN = ds4_metal_tensor_view(p->normed_TxN,
            (uint64_t)t * n_embd * sizeof(float), (uint64_t)n_embd * sizeof(float));
        if (!normed_row_1xN) return 0;

        if (wt_gate_sh_NxM->type == 20 && wt_up_sh_NxM->type == 20) {
            const uint32_t sh_bs = l26f_types[wt_gate_sh_NxM->type].block_size;
            const uint32_t sh_ts = l26f_types[wt_gate_sh_NxM->type].type_size;
            const uint64_t sh_gate_bytes = (uint64_t)wt_gate_sh_NxM->dim[1] * ((uint64_t)wt_gate_sh_NxM->dim[0] / sh_bs) * sh_ts;
            const uint64_t sh_up_bytes   = (uint64_t)wt_up_sh_NxM->dim[1] * ((uint64_t)wt_up_sh_NxM->dim[0] / sh_bs) * sh_ts;

            if (!l26f_metal_shared_gate_up_swiglu_iq4nl(p->shexp_mid_1xF,
                    m->map, m->size,
                    wt_gate_sh_NxM->abs_offset, sh_gate_bytes,
                    wt_up_sh_NxM->abs_offset, sh_up_bytes,
                    wt_gate_sh_NxM->dim[0], wt_gate_sh_NxM->dim[1],
                    normed_row_1xN)) {
                ds4_metal_tensor_free(normed_row_1xN);
                return 0;
            }
        } else {
            if (!l26f_metal_matvec_quant(p->shexp_gate_1xF, normed_row_1xN,
                    m->map, m->size, wt_gate_sh_NxM->abs_offset,
                    wt_gate_sh_NxM->dim[0], wt_gate_sh_NxM->dim[1], wt_gate_sh_NxM->type, 1)) {
                ds4_metal_tensor_free(normed_row_1xN);
                return 0;
            }
            if (!l26f_metal_matvec_quant(p->shexp_up_1xF, normed_row_1xN,
                    m->map, m->size, wt_up_sh_NxM->abs_offset,
                    wt_up_sh_NxM->dim[0], wt_up_sh_NxM->dim[1], wt_up_sh_NxM->type, 1)) {
                ds4_metal_tensor_free(normed_row_1xN);
                return 0;
            }
            if (!ds4_metal_swiglu_tensor(p->shexp_mid_1xF, p->shexp_gate_1xF, p->shexp_up_1xF, n_ff_exp, 0.0f, 1.0f)) {
                ds4_metal_tensor_free(normed_row_1xN);
                return 0;
            }
        }

        if (!l26f_metal_matvec_quant(p->shexp_out_1xN, p->shexp_mid_1xF,
                m->map, m->size, wt_down_sh_MxN->abs_offset,
                wt_down_sh_MxN->dim[0], wt_down_sh_MxN->dim[1], wt_down_sh_MxN->type, 1)) {
            ds4_metal_tensor_free(normed_row_1xN);
            return 0;
        }

        float *sh_data_1xN = (float *)malloc((uint64_t)n_embd * sizeof(float));
        if (!sh_data_1xN) { ds4_metal_tensor_free(normed_row_1xN); return 0; }
        ds4_metal_tensor_read(p->shexp_out_1xN, 0, sh_data_1xN, (uint64_t)n_embd * sizeof(float));
        ds4_metal_tensor_write(p->moe_out_TxN, (uint64_t)t * n_embd * sizeof(float), sh_data_1xN, (uint64_t)n_embd * sizeof(float));
        free(sh_data_1xN);
        ds4_metal_tensor_free(normed_row_1xN);
    }

    if (!l26f_metal_batch_accum(p->moe_out_TxN, p->down_out_TxKxN, p->sel_wt_TxK,
            p->moe_out_TxN, T, K, n_embd))
        return 0;
    if (layer <= 1) L26F_PREFILL_CKPT("moe_out_TxN_after_accum", p->moe_out_TxN, (uint64_t)T * n_embd * sizeof(float), layer, -1);

    if (!ds4_metal_add_tensor(out_TxN, inp_TxN, p->moe_out_TxN, T * n_embd))
        return 0;
    if (layer <= 1) L26F_PREFILL_CKPT("out_TxN_after_add", out_TxN, (uint64_t)T * n_embd * sizeof(float), layer, -1);

    return 1;
}
// =========================================================================
// Runs the fused kernel path with optional per-stage breaks and checkpoints.
// When L26F_DBG_COMPARE is defined, also runs the per-expert path and compares.
// Enable: -DL26F_DBG_FUSED (fused only) or -DL26F_DBG_COMPARE (both paths)

#ifdef L26F_DBG_FUSED

static int l26f_moe_ffn_fused(
        l26f_session *s, uint32_t layer,
        ds4_metal_tensor *inp, ds4_metal_tensor *out) {
    l26f_model *m = s->model;
    l26f_compute *c = &s->comp;
    const uint32_t n_embd = m->n_embd;
    const uint32_t n_ff_exp = 1024;
    const uint32_t n_expert = 256;
    const uint32_t n_groups = 8;
    const uint32_t n_exp_per_group = n_expert / n_groups;
    const float w_scale = 2.5f;
    const int K = 8;
    const int dbg_breaks = l26f_dbg_get_breaks();

    l26f_tensor *wt_norm_N          = l26f_layer_tensor(m, layer, "ffn_norm.weight");
    l26f_tensor *wt_gate_inp_NxE    = l26f_layer_tensor(m, layer, "ffn_gate_inp.weight");
    l26f_tensor *wt_exp_b_1xE       = l26f_layer_tensor(m, layer, "exp_probs_b.bias");
    l26f_tensor *wt_gate_exps_NxMxE = l26f_layer_tensor(m, layer, "ffn_gate_exps.weight");
    l26f_tensor *wt_up_exps_NxMxE   = l26f_layer_tensor(m, layer, "ffn_up_exps.weight");
    l26f_tensor *wt_down_exps_MxNxE = l26f_layer_tensor(m, layer, "ffn_down_exps.weight");
    l26f_tensor *wt_gate_sh_NxM     = l26f_layer_tensor(m, layer, "ffn_gate_shexp.weight");
    l26f_tensor *wt_up_sh_NxM       = l26f_layer_tensor(m, layer, "ffn_up_shexp.weight");
    l26f_tensor *wt_down_sh_MxN     = l26f_layer_tensor(m, layer, "ffn_down_shexp.weight");

    // 1. RMS norm
    if (!ds4_metal_rms_norm_weight_tensor(c->ffn_normed_1xN, inp,
            m->map, m->size, wt_norm_N->abs_offset, n_embd, m->rms_norm_eps))
        return 0;
    L26F_DBG_CHECKPOINT("fused_rms_norm", c->ffn_normed_1xN, n_embd);

    // 2. Router matvec + routing
    if (!ds4_metal_matmul_f32_tensor(c->router_logits_1xE, m->map, m->size,
            wt_gate_inp_NxE->abs_offset, n_embd, n_expert, c->ffn_normed_1xN, 1))
        return 0;

    int32_t has_bias = (wt_exp_b_1xE != NULL) ? 1 : 0;
    uint64_t bias_off = has_bias ? wt_exp_b_1xE->abs_offset : 0;
    if (!l26f_metal_moe_route(c->router_logits_1xE, m->map, m->size,
            bias_off, has_bias,
            c->moe_sel_idx_K, c->moe_sel_wt_K,
            n_expert, n_groups, n_exp_per_group, 4, K, w_scale))
        return 0;

    // Read back selected indices + weights (needed for per-expert comparison
    // and for gather_offsets validation)
    int32_t selected_experts_K[8];
    float selected_weights_K[8];
    L26F_DBG_READ(c->moe_sel_idx_K, (float *)selected_experts_K, 8);
    ds4_metal_end_commands();
    ds4_metal_tensor_read(c->moe_sel_wt_K, 0, selected_weights_K, K * sizeof(float));
    ds4_metal_begin_commands();

    fprintf(stderr, "DBG fused selected experts: ");
    for (int i = 0; i < K; i++)
        fprintf(stderr, "%d(%.4f)%s", selected_experts_K[i], selected_weights_K[i],
                i < K-1 ? " " : "\n");

    // 3. Gather expert weight offsets
    if (!l26f_metal_gather_offsets_3(c->moe_sel_idx_K,
            s->moe_gate_all_off_1xE[layer], s->moe_up_all_off_1xE[layer], s->moe_down_all_off_1xE[layer],
            c->moe_gate_off_K, c->moe_up_off_K, c->moe_down_off_K, K))
        return 0;

    if (dbg_breaks) {
        L26F_DBG_CHECKPOINT("fused_gate_off", c->moe_gate_off_K, K);
        L26F_DBG_CHECKPOINT("fused_up_off", c->moe_up_off_K, K);
        L26F_DBG_CHECKPOINT("fused_down_off", c->moe_down_off_K, K);
    }

    // Compute total tensor sizes for wrap_model_range
    const uint32_t gate_bs = l26f_types[wt_gate_exps_NxMxE->type].block_size;
    const uint32_t gate_ts = l26f_types[wt_gate_exps_NxMxE->type].type_size;
    const uint64_t gate_exp_bytes = (uint64_t)wt_gate_exps_NxMxE->dim[1] * ((uint64_t)wt_gate_exps_NxMxE->dim[0] / gate_bs) * gate_ts;
    const uint64_t gate_total = (uint64_t)n_expert * gate_exp_bytes;

    const uint32_t up_bs = l26f_types[wt_up_exps_NxMxE->type].block_size;
    const uint32_t up_ts = l26f_types[wt_up_exps_NxMxE->type].type_size;

    if (dbg_breaks) {
        L26F_DBG_CHECKPOINT("fused_gate_8xM", c->moe_expert_gate_8xM, (uint64_t)K * n_ff_exp);
    }

    // 5. Fused up matvec
    if (!l26f_metal_fused_moe_iq4nl(c->moe_expert_up_8xM,
            m->map, m->size, c->moe_up_off_K,
            wt_up_exps_NxMxE->abs_offset, up_total,
            K, n_embd, n_ff_exp, c->ffn_normed_1xN, 0))
        return 0;

    if (dbg_breaks) {
        L26F_DBG_CHECKPOINT("fused_up_8xM", c->moe_expert_up_8xM, (uint64_t)K * n_ff_exp);
    }

    // 6. Fused SwiGLU
    if (!l26f_metal_fused_swiglu(c->moe_expert_mid_8xM,
            c->moe_expert_gate_8xM, c->moe_expert_up_8xM,
            K, n_ff_exp))
        return 0;

    if (dbg_breaks) {
        L26F_DBG_CHECKPOINT("fused_mid_8xM", c->moe_expert_mid_8xM, (uint64_t)K * n_ff_exp);
    }

    // 7. Fused down matvec
    if (wt_down_exps_MxNxE->type == 13) {
        if (!l26f_metal_fused_moe_q5k(c->moe_expert_down_8xN,
                m->map, m->size, c->moe_down_off_K,
                wt_down_exps_MxNxE->abs_offset, down_total,
                K, n_ff_exp, n_embd, c->moe_expert_mid_8xM))
            return 0;
    } else {
        if (!l26f_metal_fused_moe_iq4nl(c->moe_expert_down_8xN,
                m->map, m->size, c->moe_down_off_K,
                wt_down_exps_MxNxE->abs_offset, down_total,
                K, n_ff_exp, n_embd, c->moe_expert_mid_8xM, 1))
            return 0;
    }

    if (dbg_breaks) {
        L26F_DBG_CHECKPOINT("fused_down_8xN", c->moe_expert_down_8xN, (uint64_t)K * n_embd);
    }

    // 8. Shared expert (same as per-expert path)
    if (!l26f_metal_matvec_quant(c->ffn_gate_1xF, c->ffn_normed_1xN,
            m->map, m->size, wt_gate_sh_NxM->abs_offset,
            wt_gate_sh_NxM->dim[0], wt_gate_sh_NxM->dim[1], wt_gate_sh_NxM->type, 1))
        return 0;
    if (!l26f_metal_matvec_quant(c->ffn_up_1xF, c->ffn_normed_1xN,
            m->map, m->size, wt_up_sh_NxM->abs_offset,
            wt_up_sh_NxM->dim[0], wt_up_sh_NxM->dim[1], wt_up_sh_NxM->type, 1))
        return 0;
    if (!ds4_metal_swiglu_tensor(c->ffn_mid_1xF, c->ffn_gate_1xF, c->ffn_up_1xF, n_ff_exp, 0.0f, 1.0f))
        return 0;
    if (!l26f_metal_matvec_quant(c->shexp_out_1xN, c->ffn_mid_1xF,
            m->map, m->size, wt_down_sh_MxN->abs_offset,
            wt_down_sh_MxN->dim[0], wt_down_sh_MxN->dim[1], wt_down_sh_MxN->type, 1))
        return 0;

    if (dbg_breaks) {
        L26F_DBG_CHECKPOINT("fused_shexp_out", c->shexp_out_1xN, n_embd);
    }

    // 9. Fused accumulate: weighted sum of 8 experts + shared expert
    if (!l26f_metal_fused_accum(c->moe_out_1xN,
            c->moe_expert_down_8xN, c->moe_sel_wt_K, c->shexp_out_1xN,
            K, n_embd))
        return 0;

    L26F_DBG_CHECKPOINT("fused_moe_out", c->moe_out_1xN, n_embd);

    // 10. Residual add
    if (!ds4_metal_add_tensor(out, inp, c->moe_out_1xN, n_embd))
        return 0;

    L26F_DBG_CHECKPOINT("fused_final_out", out, n_embd);

    return 1;
}

#ifdef L26F_DBG_COMPARE

// Run both per-expert (path A) and fused (path B) for one MoE layer,
// compare at every pipeline stage.

static int l26f_moe_ffn_compare(
        l26f_session *s, uint32_t layer,
        ds4_metal_tensor *inp, ds4_metal_tensor *out)
{
    l26f_model *m = s->model;
    l26f_compute *c = &s->comp;
    const uint32_t n_embd = m->n_embd;
    const uint32_t n_ff_exp = 1024;
    const uint32_t n_expert = 256;
    const uint32_t n_groups = 8;
    const uint32_t n_exp_per_group = n_expert / n_groups;
    const float w_scale = 2.5f;
    const int K = 8;
    const uint64_t act_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t ffn_bytes = (uint64_t)n_ff_exp * sizeof(float);

    fprintf(stderr, "\n=== L26F_DBG_COMPARE layer %u ===\n", layer);

    l26f_tensor *wt_norm_N          = l26f_layer_tensor(m, layer, "ffn_norm.weight");
    l26f_tensor *wt_gate_inp_NxE    = l26f_layer_tensor(m, layer, "ffn_gate_inp.weight");
    l26f_tensor *wt_exp_b_1xE       = l26f_layer_tensor(m, layer, "exp_probs_b.bias");
    l26f_tensor *wt_gate_exps_NxMxE = l26f_layer_tensor(m, layer, "ffn_gate_exps.weight");
    l26f_tensor *wt_up_exps_NxMxE   = l26f_layer_tensor(m, layer, "ffn_up_exps.weight");
    l26f_tensor *wt_down_exps_MxNxE = l26f_layer_tensor(m, layer, "ffn_down_exps.weight");
    l26f_tensor *wt_gate_sh_NxM     = l26f_layer_tensor(m, layer, "ffn_gate_shexp.weight");
    l26f_tensor *wt_up_sh_NxM       = l26f_layer_tensor(m, layer, "ffn_up_shexp.weight");
    l26f_tensor *wt_down_sh_MxN     = l26f_layer_tensor(m, layer, "ffn_down_shexp.weight");

    // Shared RMS norm (same for both paths)
    if (!ds4_metal_rms_norm_weight_tensor(c->ffn_normed_1xN, inp,
            m->map, m->size, wt_norm_N->abs_offset, n_embd, m->rms_norm_eps))
        return 0;
    L26F_DBG_CHECKPOINT("shared_rms_norm", c->ffn_normed_1xN, n_embd);

    // Save ffn_normed for fused path (both paths read from same buffer, but
    // per-expert path doesn't modify it, so this is safe)
    float *ffn_normed_copy = (float *)malloc(act_bytes);
    if (!ffn_normed_copy) return 0;
    L26F_DBG_READ(c->ffn_normed_1xN, ffn_normed_copy, n_embd);

    // ---- PATH A: Per-expert (working) ----
    fprintf(stderr, "--- PATH A: Per-expert ---\n");

    // Router matvec
    if (!ds4_metal_matmul_f32_tensor(c->router_logits_1xE, m->map, m->size,
            wt_gate_inp_NxE->abs_offset, n_embd, n_expert, c->ffn_normed_1xN, 1))
        return 0;
    L26F_DBG_CHECKPOINT("A_router_logits", c->router_logits_1xE, n_expert);

    // Routing
    int32_t has_bias = (wt_exp_b_1xE != NULL) ? 1 : 0;
    uint64_t bias_off = has_bias ? wt_exp_b_1xE->abs_offset : 0;
    if (!l26f_metal_moe_route(c->router_logits_1xE, m->map, m->size,
            bias_off, has_bias,
            c->moe_sel_idx_K, c->moe_sel_wt_K,
            n_expert, n_groups, n_exp_per_group, 4, K, w_scale))
        return 0;

    int32_t selected_experts_K[8];
    float selected_weights_K[8];
    ds4_metal_end_commands();
    ds4_metal_tensor_read(c->moe_sel_idx_K, 0, selected_experts_K, K * sizeof(int32_t));
    ds4_metal_tensor_read(c->moe_sel_wt_K,  0, selected_weights_K, K * sizeof(float));
    ds4_metal_begin_commands();

    fprintf(stderr, "A selected: ");
    for (int i = 0; i < K; i++)
        fprintf(stderr, "%d(%.4f)%s", selected_experts_K[i], selected_weights_K[i],
                i < K-1 ? " " : "\n");

    // Per-expert byte strides
    const uint32_t gate_bs = l26f_types[wt_gate_exps_NxMxE->type].block_size;
    const uint32_t gate_ts = l26f_types[wt_gate_exps_NxMxE->type].type_size;
    const uint64_t gate_exp_bytes = (uint64_t)wt_gate_exps_NxMxE->dim[1] * ((uint64_t)wt_gate_exps_NxMxE->dim[0] / gate_bs) * gate_ts;
    const uint32_t up_bs = l26f_types[wt_up_exps_NxMxE->type].block_size;
    const uint32_t up_ts = l26f_types[wt_up_exps_NxMxE->type].type_size;
    const uint64_t up_exp_bytes = (uint64_t)wt_up_exps_NxMxE->dim[1] * ((uint64_t)wt_up_exps_NxMxE->dim[0] / up_bs) * up_ts;
    const uint32_t down_bs = l26f_types[wt_down_exps_MxNxE->type].block_size;
    const uint32_t down_ts = l26f_types[wt_down_exps_MxNxE->type].type_size;
    const uint64_t down_exp_bytes = (uint64_t)wt_down_exps_MxNxE->dim[1] * ((uint64_t)wt_down_exps_MxNxE->dim[0] / down_bs) * down_ts;

    // Allocate CPU buffers for per-expert intermediate results
    float *A_gate_8xM = (float *)malloc((uint64_t)K * n_ff_exp * sizeof(float));
    float *A_up_8xM   = (float *)malloc((uint64_t)K * n_ff_exp * sizeof(float));
    float *A_mid_8xM  = (float *)malloc((uint64_t)K * n_ff_exp * sizeof(float));
    float *A_down_8xN = (float *)malloc((uint64_t)K * n_embd * sizeof(float));
    float *A_moe_out  = (float *)malloc(act_bytes);
    if (!A_gate_8xM || !A_up_8xM || !A_mid_8xM || !A_down_8xN || !A_moe_out)
        return 0;

    ds4_metal_tensor_fill(c->moe_out_1xN, 0.0f);

    for (int i = 0; i < K; i++) {
        int e = selected_experts_K[i];
        uint64_t gate_off = wt_gate_exps_NxMxE->abs_offset + (uint64_t)e * gate_exp_bytes;
        uint64_t up_off   = wt_up_exps_NxMxE->abs_offset   + (uint64_t)e * up_exp_bytes;
        uint64_t down_off = wt_down_exps_MxNxE->abs_offset + (uint64_t)e * down_exp_bytes;

        // Gate
        if (!l26f_metal_matvec_quant(c->ffn_gate_1xF, c->ffn_normed_1xN,
                m->map, m->size, gate_off,
                n_embd, n_ff_exp, wt_gate_exps_NxMxE->type, 1))
            return 0;
        ds4_metal_end_commands();
        ds4_metal_tensor_read(c->ffn_gate_1xF, 0, A_gate_8xM + (uint64_t)i * n_ff_exp, ffn_bytes);
        ds4_metal_begin_commands();

        // Up
        if (!l26f_metal_matvec_quant(c->ffn_up_1xF, c->ffn_normed_1xN,
                m->map, m->size, up_off,
                n_embd, n_ff_exp, wt_up_exps_NxMxE->type, 1))
            return 0;
        ds4_metal_end_commands();
        ds4_metal_tensor_read(c->ffn_up_1xF, 0, A_up_8xM + (uint64_t)i * n_ff_exp, ffn_bytes);
        ds4_metal_begin_commands();

        // SwiGLU
        if (!ds4_metal_swiglu_tensor(c->ffn_mid_1xF, c->ffn_gate_1xF, c->ffn_up_1xF, n_ff_exp, 0.0f, 1.0f))
            return 0;
        ds4_metal_end_commands();
        ds4_metal_tensor_read(c->ffn_mid_1xF, 0, A_mid_8xM + (uint64_t)i * n_ff_exp, ffn_bytes);
        ds4_metal_begin_commands();

        // Down
        if (!l26f_metal_matvec_quant(c->ffn_down_1xN, c->ffn_mid_1xF,
                m->map, m->size, down_off,
                n_ff_exp, n_embd, wt_down_exps_MxNxE->type, 1))
            return 0;
        ds4_metal_end_commands();
        ds4_metal_tensor_read(c->ffn_down_1xN, 0, A_down_8xN + (uint64_t)i * n_embd, act_bytes);
        ds4_metal_begin_commands();

        // axpy
        if (!l26f_metal_axpy(c->moe_out_1xN, c->ffn_down_1xN, selected_weights_K[i], n_embd))
            return 0;
    }

    // Shared expert
    if (!l26f_metal_matvec_quant(c->ffn_gate_1xF, c->ffn_normed_1xN,
            m->map, m->size, wt_gate_sh_NxM->abs_offset,
            wt_gate_sh_NxM->dim[0], wt_gate_sh_NxM->dim[1], wt_gate_sh_NxM->type, 1))
        return 0;
    if (!l26f_metal_matvec_quant(c->ffn_up_1xF, c->ffn_normed_1xN,
            m->map, m->size, wt_up_sh_NxM->abs_offset,
            wt_up_sh_NxM->dim[0], wt_up_sh_NxM->dim[1], wt_up_sh_NxM->type, 1))
        return 0;
    if (!ds4_metal_swiglu_tensor(c->ffn_mid_1xF, c->ffn_gate_1xF, c->ffn_up_1xF, n_ff_exp, 0.0f, 1.0f))
        return 0;
    if (!l26f_metal_matvec_quant(c->shexp_out_1xN, c->ffn_mid_1xF,
            m->map, m->size, wt_down_sh_MxN->abs_offset,
            wt_down_sh_MxN->dim[0], wt_down_sh_MxN->dim[1], wt_down_sh_MxN->type, 1))
        return 0;
    if (!l26f_metal_axpy(c->moe_out_1xN, c->shexp_out_1xN, 1.0f, n_embd))
        return 0;

    ds4_metal_end_commands();
    ds4_metal_tensor_read(c->moe_out_1xN, 0, A_moe_out, act_bytes);
    ds4_metal_begin_commands();
    L26F_DBG_CHECKPOINT("A_moe_out", c->moe_out_1xN, n_embd);

    // Residual
    if (!ds4_metal_add_tensor(out, inp, c->moe_out_1xN, n_embd))
        return 0;
    float *A_final = (float *)malloc(act_bytes);
    if (!A_final) return 0;
    ds4_metal_end_commands();
    ds4_metal_tensor_read(out, 0, A_final, act_bytes);
    ds4_metal_begin_commands();

    fprintf(stderr, "--- PATH A complete ---\n\n");

    // ---- PATH B: Fused ----
    fprintf(stderr, "--- PATH B: Fused ---\n");

    // Restore ffn_normed (should be unchanged since per-expert path only reads it)
    ds4_metal_tensor_write(c->ffn_normed_1xN, 0, ffn_normed_copy, act_bytes);

    // Need separate output buffer for fused path
    ds4_metal_tensor *fused_out = ds4_metal_tensor_alloc(act_bytes);
    if (!fused_out) return 0;
    ds4_metal_tensor_fill(fused_out, 0.0f);

    if (!l26f_moe_ffn_fused(s, layer, inp, fused_out)) {
        fprintf(stderr, "Fused path failed!\n");
        ds4_metal_tensor_free(fused_out);
        return 0;
    }

    float *B_final = (float *)malloc(act_bytes);
    if (!B_final) return 0;
    ds4_metal_end_commands();
    ds4_metal_tensor_read(fused_out, 0, B_final, act_bytes);
    ds4_metal_begin_commands();

    // Also read fused intermediates
    float *B_gate_8xM = (float *)malloc((uint64_t)K * n_ff_exp * sizeof(float));
    float *B_up_8xM   = (float *)malloc((uint64_t)K * n_ff_exp * sizeof(float));
    float *B_mid_8xM  = (float *)malloc((uint64_t)K * n_ff_exp * sizeof(float));
    float *B_down_8xN = (float *)malloc((uint64_t)K * n_embd * sizeof(float));
    if (!B_gate_8xM || !B_up_8xM || !B_mid_8xM || !B_down_8xN) return 0;

    ds4_metal_end_commands();
    ds4_metal_tensor_read(c->moe_expert_gate_8xM, 0, B_gate_8xM, (uint64_t)K * n_ff_exp * sizeof(float));
    ds4_metal_tensor_read(c->moe_expert_up_8xM, 0, B_up_8xM, (uint64_t)K * n_ff_exp * sizeof(float));
    ds4_metal_tensor_read(c->moe_expert_mid_8xM, 0, B_mid_8xM, (uint64_t)K * n_ff_exp * sizeof(float));
    ds4_metal_tensor_read(c->moe_expert_down_8xN, 0, B_down_8xN, (uint64_t)K * n_embd * sizeof(float));
    ds4_metal_begin_commands();

    // ---- COMPARE ----
    fprintf(stderr, "\n=== COMPARISON: Path A (per-expert) vs Path B (fused) ===\n");
    L26F_DBG_ASSERT_EQ("gate_8xM", A_gate_8xM, B_gate_8xM, (uint64_t)K * n_ff_exp);
    L26F_DBG_ASSERT_EQ("up_8xM",   A_up_8xM,   B_up_8xM,   (uint64_t)K * n_ff_exp);
    L26F_DBG_ASSERT_EQ("mid_8xM",  A_mid_8xM,  B_mid_8xM,  (uint64_t)K * n_ff_exp);
    L26F_DBG_ASSERT_EQ("down_8xN", A_down_8xN, B_down_8xN, (uint64_t)K * n_embd);
    L26F_DBG_ASSERT_EQ("final",    A_final,     B_final,    n_embd);

    // Copy fused output to actual output
    ds4_metal_tensor_write(out, 0, B_final, act_bytes);

    free(A_gate_8xM); free(A_up_8xM); free(A_mid_8xM); free(A_down_8xN); free(A_moe_out); free(A_final);
    free(B_gate_8xM); free(B_up_8xM); free(B_mid_8xM); free(B_down_8xN); free(B_final);
    free(ffn_normed_copy);
    ds4_metal_tensor_free(fused_out);

    fprintf(stderr, "=== END COMPARE ===\n\n");
    return 1;
}

#endif // L26F_DBG_COMPARE
#endif // L26F_DBG_FUSED

static int l26f_session_init(l26f_session *s, l26f_model *m) {
    memset(s, 0, sizeof(*s));
    s->model = m;

    const uint32_t n_embd = m->n_embd;
    const uint32_t n_ff = m->n_ff;
    const uint32_t S = 128, H = m->n_head;
    const uint64_t act_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t qkv_bytes = 3ULL * n_embd * sizeof(float);
    const uint64_t ffn_bytes = (uint64_t)n_ff * sizeof(float);
    const uint64_t gla_state_bytes = (uint64_t)S * S * H * sizeof(float);
    const uint64_t gla_out_bytes = act_bytes + gla_state_bytes;

    // Allocate compute buffers (zeroed — Metal shared memory is uninitialized)
    s->comp.normed_1xN      = ds4_metal_tensor_alloc(act_bytes);
    s->comp.qkv_1x3N         = ds4_metal_tensor_alloc(qkv_bytes);
    s->comp.gate_out_1xN    = ds4_metal_tensor_alloc(act_bytes);
    s->comp.q_rope_1xN      = ds4_metal_tensor_alloc(act_bytes);
    s->comp.k_rope_1xN      = ds4_metal_tensor_alloc(act_bytes);
    s->comp.gla_out_1xNxSxSxH     = ds4_metal_tensor_alloc(gla_out_bytes);
    s->comp.gated_gla_1xN   = ds4_metal_tensor_alloc(act_bytes);
    s->comp.attn_proj_1xN   = ds4_metal_tensor_alloc(act_bytes);
    s->comp.post_attn_1xN   = ds4_metal_tensor_alloc(act_bytes);
    s->comp.ffn_normed_1xN  = ds4_metal_tensor_alloc(act_bytes);
    s->comp.ffn_gate_1xF    = ds4_metal_tensor_alloc(ffn_bytes);
    s->comp.ffn_up_1xF      = ds4_metal_tensor_alloc(ffn_bytes);
    s->comp.ffn_mid_1xF     = ds4_metal_tensor_alloc(ffn_bytes);
    s->comp.ffn_down_1xN    = ds4_metal_tensor_alloc(act_bytes);
    s->comp.moe_out_1xN     = ds4_metal_tensor_alloc(act_bytes);
    s->comp.shexp_out_1xN   = ds4_metal_tensor_alloc(act_bytes);
    s->comp.router_logits_1xE = ds4_metal_tensor_alloc(256 * sizeof(float));
    s->comp.moe_sel_idx_K    = ds4_metal_tensor_alloc(8 * sizeof(int32_t));
    s->comp.moe_sel_wt_K     = ds4_metal_tensor_alloc(8 * sizeof(float));
    s->comp.moe_expert_gate_8xM = ds4_metal_tensor_alloc((uint64_t)8 * n_ff * sizeof(float));
    s->comp.moe_expert_up_8xM   = ds4_metal_tensor_alloc((uint64_t)8 * n_ff * sizeof(float));
    s->comp.moe_expert_mid_8xM  = ds4_metal_tensor_alloc((uint64_t)8 * n_ff * sizeof(float));
    s->comp.moe_expert_down_8xN = ds4_metal_tensor_alloc((uint64_t)8 * n_embd * sizeof(float));
    s->comp.moe_gate_off_K  = ds4_metal_tensor_alloc(8 * sizeof(uint64_t));
    s->comp.moe_up_off_K    = ds4_metal_tensor_alloc(8 * sizeof(uint64_t));
    s->comp.moe_down_off_K  = ds4_metal_tensor_alloc(8 * sizeof(uint64_t));

    // Expert weight caches for fused MoE (8 experts contiguous)
    s->comp.moe_gate_cache_8xMxN = ds4_metal_tensor_alloc((uint64_t)8 * 4096 * (1024 / 32) * 18);
    s->comp.moe_up_cache_8xMxN   = ds4_metal_tensor_alloc((uint64_t)8 * 4096 * (1024 / 32) * 18);
    s->comp.moe_down_cache_8xNxM = ds4_metal_tensor_alloc((uint64_t)8 * 1024 * (4096 / 256) * 176);

    s->hidden_1xN         = ds4_metal_tensor_alloc(act_bytes);
    s->output_normed_1xN  = ds4_metal_tensor_alloc(act_bytes);
    s->logits_1xV         = ds4_metal_tensor_alloc((uint64_t)m->n_vocab * sizeof(float));
    s->sample_idx_1xI     = ds4_metal_tensor_alloc(sizeof(int32_t));

    // Zero all compute buffers to eliminate uninitialized-memory non-determinism
    ds4_metal_tensor_fill(s->comp.normed_1xN,      0.0f);
    ds4_metal_tensor_fill(s->comp.qkv_1x3N,         0.0f);
    ds4_metal_tensor_fill(s->comp.gate_out_1xN,    0.0f);
    ds4_metal_tensor_fill(s->comp.q_rope_1xN,      0.0f);
    ds4_metal_tensor_fill(s->comp.k_rope_1xN,      0.0f);
    ds4_metal_tensor_fill(s->comp.gla_out_1xNxSxSxH,     0.0f);
    ds4_metal_tensor_fill(s->comp.gated_gla_1xN,   0.0f);
    ds4_metal_tensor_fill(s->comp.attn_proj_1xN,   0.0f);
    ds4_metal_tensor_fill(s->comp.post_attn_1xN,   0.0f);
    ds4_metal_tensor_fill(s->comp.ffn_normed_1xN,  0.0f);
    ds4_metal_tensor_fill(s->comp.ffn_gate_1xF,    0.0f);
    ds4_metal_tensor_fill(s->comp.ffn_up_1xF,      0.0f);
    ds4_metal_tensor_fill(s->comp.ffn_mid_1xF,     0.0f);
    ds4_metal_tensor_fill(s->comp.ffn_down_1xN,    0.0f);
    ds4_metal_tensor_fill(s->comp.moe_out_1xN,     0.0f);
    ds4_metal_tensor_fill(s->comp.shexp_out_1xN,   0.0f);
    ds4_metal_tensor_fill(s->comp.moe_sel_idx_K,   0.0f);
    ds4_metal_tensor_fill(s->comp.moe_sel_wt_K,    0.0f);
    ds4_metal_tensor_fill(s->comp.moe_expert_gate_8xM, 0.0f);
    ds4_metal_tensor_fill(s->comp.moe_expert_up_8xM,   0.0f);
    ds4_metal_tensor_fill(s->comp.moe_expert_mid_8xM,  0.0f);
    ds4_metal_tensor_fill(s->comp.moe_expert_down_8xN, 0.0f);
    ds4_metal_tensor_fill(s->comp.moe_gate_off_K,  0.0f);
    ds4_metal_tensor_fill(s->comp.moe_up_off_K,    0.0f);
    ds4_metal_tensor_fill(s->comp.moe_down_off_K,  0.0f);
    ds4_metal_tensor_fill(s->comp.moe_gate_cache_8xMxN, 0.0f);
    ds4_metal_tensor_fill(s->comp.moe_up_cache_8xMxN,   0.0f);
    ds4_metal_tensor_fill(s->comp.moe_down_cache_8xNxM, 0.0f);
    ds4_metal_tensor_fill(s->hidden_1xN,           0.0f);
    ds4_metal_tensor_fill(s->output_normed_1xN,    0.0f);
    ds4_metal_tensor_fill(s->logits_1xV,           0.0f);
    ds4_metal_tensor_fill(s->sample_idx_1xI,       0);

    // GLA states for all GLA layers (28 layers: all except 7, 15, 23, 31)
    for (uint32_t i = 0; i < 32; i++) {
        if (m->is_mla[i]) continue;
        s->gla_states[i].state = ds4_metal_tensor_alloc(gla_state_bytes);
        if (!s->gla_states[i].state) { fprintf(stderr, "l26f: OOM GLA state %u\n", i); return 0; }
        s->gla_slopes_1xN[i] = ds4_metal_tensor_alloc(act_bytes);
        if (!s->gla_slopes_1xN[i]) { fprintf(stderr, "l26f: OOM GLA slope %u\n", i); return 0; }
        if (!l26f_write_gla_slopes_1xN(m, i, s->gla_slopes_1xN[i])) {
            fprintf(stderr, "l26f: failed to initialize GLA slope %u\n", i);
            return 0;
        }
        void *zeros = calloc(1, gla_state_bytes);
        if (zeros) {
            ds4_metal_tensor_write(s->gla_states[i].state, 0, zeros, gla_state_bytes);
            free(zeros);
        }
    }

    // MLA KV caches for MLA layers (7, 15, 23, 31)
    const uint32_t kv_dim = m->kv_lora_rank + m->n_rot;  // 512 + 64 = 576
    for (uint32_t i = 0; i < 32; i++) {
        if (!m->is_mla[i]) continue;
        if (use_gpu_mla()) {
            s->mla_kv_gpu[i] = l26f_mla_kv_cache_gpu_alloc(4096, kv_dim);
            if (!s->mla_kv_gpu[i]) { fprintf(stderr, "l26f: OOM MLA GPU KV %u\n", i); return 0; }
        } else {
            s->mla_kv_cpu[i] = l26f_mla_kv_cache_alloc(4096, kv_dim);
            if (!s->mla_kv_cpu[i]) { fprintf(stderr, "l26f: OOM MLA CPU KV %u\n", i); return 0; }
        }
    }

    // MLA GPU compute buffers (shared across all MLA layers) — only if GPU path
    if (use_gpu_mla()) {
        s->mla_comp = l26f_mla_compute_alloc(n_embd, m->n_head, m->q_lora_rank,
                                               m->kv_lora_rank, m->n_rot, 192);
        if (!s->mla_comp) { fprintf(stderr, "l26f: OOM MLA compute buffers\n"); return 0; }
    }

    // Precompute per-layer expert weight offset tables (256 entries × 8 bytes each)
    for (uint32_t il = 0; il < 32; il++) {
        if (il == 0) continue;  // layer 0 uses dense FFN, not MoE

        l26f_tensor *wt_gate = l26f_layer_tensor(m, il, "ffn_gate_exps.weight");
        l26f_tensor *wt_up   = l26f_layer_tensor(m, il, "ffn_up_exps.weight");
        l26f_tensor *wt_down = l26f_layer_tensor(m, il, "ffn_down_exps.weight");
        if (!wt_gate || !wt_up || !wt_down) continue;

        const uint32_t gate_bs = l26f_types[wt_gate->type].block_size;
        const uint32_t gate_ts = l26f_types[wt_gate->type].type_size;
        const uint32_t up_bs   = l26f_types[wt_up->type].block_size;
        const uint32_t up_ts   = l26f_types[wt_up->type].type_size;
        const uint32_t down_bs = l26f_types[wt_down->type].block_size;
        const uint32_t down_ts = l26f_types[wt_down->type].type_size;
        const uint64_t gate_exp_bytes = (uint64_t)wt_gate->dim[1] * ((uint64_t)wt_gate->dim[0] / gate_bs) * gate_ts;
        const uint64_t up_exp_bytes   = (uint64_t)wt_up->dim[1]   * ((uint64_t)wt_up->dim[0]   / up_bs)   * up_ts;
        const uint64_t down_exp_bytes = (uint64_t)wt_down->dim[1] * ((uint64_t)wt_down->dim[0] / down_bs) * down_ts;

        uint64_t gate_offs[256], up_offs[256], down_offs[256];
        for (int e = 0; e < 256; e++) {
            gate_offs[e] = (uint64_t)e * gate_exp_bytes;
            up_offs[e]   = (uint64_t)e * up_exp_bytes;
            down_offs[e] = (uint64_t)e * down_exp_bytes;
        }

        uint64_t gate_cache_offs[8], up_cache_offs[8], down_cache_offs[8];
        for (int e = 0; e < 8; e++) {
            gate_cache_offs[e] = (uint64_t)e * gate_exp_bytes;
            up_cache_offs[e]   = (uint64_t)e * up_exp_bytes;
            down_cache_offs[e] = (uint64_t)e * down_exp_bytes;
        }

        s->moe_gate_all_off_1xE[il] = ds4_metal_tensor_alloc(256 * sizeof(uint64_t));
        s->moe_up_all_off_1xE[il]   = ds4_metal_tensor_alloc(256 * sizeof(uint64_t));
        s->moe_down_all_off_1xE[il] = ds4_metal_tensor_alloc(256 * sizeof(uint64_t));
        s->moe_gate_cache_off_K[il] = ds4_metal_tensor_alloc(8 * sizeof(uint64_t));
        s->moe_up_cache_off_K[il]   = ds4_metal_tensor_alloc(8 * sizeof(uint64_t));
        s->moe_down_cache_off_K[il] = ds4_metal_tensor_alloc(8 * sizeof(uint64_t));
        if (!s->moe_gate_all_off_1xE[il] || !s->moe_up_all_off_1xE[il] || !s->moe_down_all_off_1xE[il] ||
            !s->moe_gate_cache_off_K[il] || !s->moe_up_cache_off_K[il] || !s->moe_down_cache_off_K[il]) {
            fprintf(stderr, "l26f: OOM MoE offset tables\n"); return 0;
        }
        ds4_metal_tensor_write(s->moe_gate_all_off_1xE[il], 0, gate_offs, sizeof(gate_offs));
        ds4_metal_tensor_write(s->moe_up_all_off_1xE[il],   0, up_offs,   sizeof(up_offs));
        ds4_metal_tensor_write(s->moe_down_all_off_1xE[il], 0, down_offs, sizeof(down_offs));
        ds4_metal_tensor_write(s->moe_gate_cache_off_K[il], 0, gate_cache_offs, sizeof(gate_cache_offs));
        ds4_metal_tensor_write(s->moe_up_cache_off_K[il],   0, up_cache_offs,   sizeof(up_cache_offs));
        ds4_metal_tensor_write(s->moe_down_cache_off_K[il], 0, down_cache_offs, sizeof(down_cache_offs));
    }

    if (!s->comp.normed_1xN || !s->comp.qkv_1x3N || !s->comp.gate_out_1xN ||
        !s->comp.q_rope_1xN || !s->comp.k_rope_1xN ||
        !s->comp.gla_out_1xNxSxSxH || !s->comp.gated_gla_1xN ||
        !s->comp.attn_proj_1xN || !s->comp.post_attn_1xN ||
        !s->comp.ffn_normed_1xN || !s->comp.ffn_gate_1xF || !s->comp.ffn_up_1xF ||
        !s->comp.ffn_mid_1xF || !s->comp.ffn_down_1xN ||
        !s->hidden_1xN || !s->output_normed_1xN || !s->logits_1xV || !s->sample_idx_1xI) {
        fprintf(stderr, "l26f: OOM compute buffers\n");
        return 0;
    }

    return 1;
}

static void l26f_session_free(l26f_session *s) {
    ds4_metal_tensor_free(s->comp.normed_1xN);
    ds4_metal_tensor_free(s->comp.qkv_1x3N);
    ds4_metal_tensor_free(s->comp.gate_out_1xN);
    ds4_metal_tensor_free(s->comp.q_rope_1xN);
    ds4_metal_tensor_free(s->comp.k_rope_1xN);
    ds4_metal_tensor_free(s->comp.gla_out_1xNxSxSxH);
    ds4_metal_tensor_free(s->comp.gated_gla_1xN);
    ds4_metal_tensor_free(s->comp.attn_proj_1xN);
    ds4_metal_tensor_free(s->comp.post_attn_1xN);
    ds4_metal_tensor_free(s->comp.ffn_normed_1xN);
    ds4_metal_tensor_free(s->comp.ffn_gate_1xF);
    ds4_metal_tensor_free(s->comp.ffn_up_1xF);
    ds4_metal_tensor_free(s->comp.ffn_mid_1xF);
    ds4_metal_tensor_free(s->comp.ffn_down_1xN);
    ds4_metal_tensor_free(s->comp.moe_out_1xN);
    ds4_metal_tensor_free(s->comp.shexp_out_1xN);
    ds4_metal_tensor_free(s->comp.moe_sel_idx_K);
    ds4_metal_tensor_free(s->comp.moe_sel_wt_K);
    ds4_metal_tensor_free(s->comp.moe_expert_gate_8xM);
    ds4_metal_tensor_free(s->comp.moe_expert_up_8xM);
    ds4_metal_tensor_free(s->comp.moe_expert_mid_8xM);
    ds4_metal_tensor_free(s->comp.moe_expert_down_8xN);
    ds4_metal_tensor_free(s->comp.moe_gate_off_K);
    ds4_metal_tensor_free(s->comp.moe_up_off_K);
    ds4_metal_tensor_free(s->comp.moe_down_off_K);
    ds4_metal_tensor_free(s->comp.moe_gate_cache_8xMxN);
    ds4_metal_tensor_free(s->comp.moe_up_cache_8xMxN);
    ds4_metal_tensor_free(s->comp.moe_down_cache_8xNxM);
    ds4_metal_tensor_free(s->comp.router_logits_1xE);
    ds4_metal_tensor_free(s->hidden_1xN);
    ds4_metal_tensor_free(s->output_normed_1xN);
    ds4_metal_tensor_free(s->logits_1xV);
    ds4_metal_tensor_free(s->sample_idx_1xI);
    for (uint32_t i = 0; i < 32; i++) {
        if (s->gla_states[i].state)
            ds4_metal_tensor_free(s->gla_states[i].state);
        if (s->gla_slopes_1xN[i])
            ds4_metal_tensor_free(s->gla_slopes_1xN[i]);
        if (s->mla_kv_gpu[i])
            l26f_mla_kv_cache_gpu_free(s->mla_kv_gpu[i]);
        if (s->mla_kv_cpu[i])
            l26f_mla_kv_cache_free(s->mla_kv_cpu[i]);
    }
    if (s->mla_comp)
        l26f_mla_compute_free(s->mla_comp);
    for (uint32_t i = 0; i < 32; i++) {
        if (s->moe_gate_all_off_1xE[i]) ds4_metal_tensor_free(s->moe_gate_all_off_1xE[i]);
        if (s->moe_up_all_off_1xE[i])   ds4_metal_tensor_free(s->moe_up_all_off_1xE[i]);
        if (s->moe_down_all_off_1xE[i]) ds4_metal_tensor_free(s->moe_down_all_off_1xE[i]);
        if (s->moe_gate_cache_off_K[i]) ds4_metal_tensor_free(s->moe_gate_cache_off_K[i]);
        if (s->moe_up_cache_off_K[i])   ds4_metal_tensor_free(s->moe_up_cache_off_K[i]);
        if (s->moe_down_cache_off_K[i]) ds4_metal_tensor_free(s->moe_down_cache_off_K[i]);
    }
}

static l26f_prefill_compute *l26f_prefill_alloc(uint32_t n_embd, uint32_t n_ff_exp) {
    const uint32_t Tmax = L26F_PREFILL_MAX_T;
    const uint32_t K = 8;
    const uint32_t E = 256;
    const uint64_t act_T_bytes = (uint64_t)Tmax * n_embd * sizeof(float);
    const uint64_t ffn_T_bytes = (uint64_t)Tmax * n_ff_exp * sizeof(float);

    l26f_prefill_compute *p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->max_tokens = Tmax;

    p->hidden_TxN          = ds4_metal_tensor_alloc(act_T_bytes);
    p->post_attn_TxN       = ds4_metal_tensor_alloc(act_T_bytes);
    p->normed_TxN          = ds4_metal_tensor_alloc(act_T_bytes);
    p->router_logits_TxE   = ds4_metal_tensor_alloc((uint64_t)Tmax * E * sizeof(float));
    p->sel_idx_TxK         = ds4_metal_tensor_alloc((uint64_t)Tmax * K * sizeof(int32_t));
    p->sel_wt_TxK          = ds4_metal_tensor_alloc((uint64_t)Tmax * K * sizeof(float));
    p->tokens_per_expert_E = ds4_metal_tensor_alloc((uint64_t)E * sizeof(uint32_t));
    p->ids_buffer_ExT      = ds4_metal_tensor_alloc((uint64_t)E * Tmax * sizeof(int32_t));
    p->gate_out_TxKxM            = ds4_metal_tensor_alloc((uint64_t)Tmax * K * n_ff_exp * sizeof(float));
    p->up_out_TxKxM              = ds4_metal_tensor_alloc((uint64_t)Tmax * K * n_ff_exp * sizeof(float));
    p->mid_TxKxM                 = ds4_metal_tensor_alloc((uint64_t)Tmax * K * n_ff_exp * sizeof(float));
    p->down_out_TxKxN            = ds4_metal_tensor_alloc((uint64_t)Tmax * K * n_embd * sizeof(float));
    p->moe_out_TxN         = ds4_metal_tensor_alloc(act_T_bytes);
    p->shexp_gate_1xF      = ds4_metal_tensor_alloc((uint64_t)n_ff_exp * sizeof(float));
    p->shexp_up_1xF        = ds4_metal_tensor_alloc((uint64_t)n_ff_exp * sizeof(float));
    p->shexp_mid_1xF       = ds4_metal_tensor_alloc((uint64_t)n_ff_exp * sizeof(float));
    p->shexp_out_1xN       = ds4_metal_tensor_alloc((uint64_t)n_embd * sizeof(float));

    if (!p->hidden_TxN || !p->post_attn_TxN || !p->normed_TxN ||
        !p->router_logits_TxE || !p->sel_idx_TxK || !p->sel_wt_TxK ||
        !p->tokens_per_expert_E || !p->ids_buffer_ExT ||
        !p->gate_out_TxKxM || !p->up_out_TxKxM || !p->mid_TxKxM || !p->down_out_TxKxN ||
        !p->moe_out_TxN ||
        !p->shexp_gate_1xF || !p->shexp_up_1xF || !p->shexp_mid_1xF || !p->shexp_out_1xN) {
        fprintf(stderr, "l26f: OOM prefill buffers\n");
        return NULL;
    }

    ds4_metal_tensor_fill(p->hidden_TxN, 0.0f);
    ds4_metal_tensor_fill(p->post_attn_TxN, 0.0f);
    ds4_metal_tensor_fill(p->moe_out_TxN, 0.0f);
    return p;
}

static void l26f_prefill_free(l26f_prefill_compute *p) {
    if (!p) return;
    ds4_metal_tensor_free(p->hidden_TxN);
    ds4_metal_tensor_free(p->post_attn_TxN);
    ds4_metal_tensor_free(p->normed_TxN);
    ds4_metal_tensor_free(p->router_logits_TxE);
    ds4_metal_tensor_free(p->sel_idx_TxK);
    ds4_metal_tensor_free(p->sel_wt_TxK);
    ds4_metal_tensor_free(p->tokens_per_expert_E);
    ds4_metal_tensor_free(p->ids_buffer_ExT);
    ds4_metal_tensor_free(p->gate_out_TxKxM);
    ds4_metal_tensor_free(p->up_out_TxKxM);
    ds4_metal_tensor_free(p->mid_TxKxM);
    ds4_metal_tensor_free(p->down_out_TxKxN);
    ds4_metal_tensor_free(p->moe_out_TxN);
    ds4_metal_tensor_free(p->shexp_gate_1xF);
    ds4_metal_tensor_free(p->shexp_up_1xF);
    ds4_metal_tensor_free(p->shexp_mid_1xF);
    ds4_metal_tensor_free(p->shexp_out_1xN);
    free(p);
}

static int l26f_dequant_q8_0_row(const uint8_t *base, uint64_t row_offset,
        float *out, uint32_t n_elements) {
    const uint32_t block_size = 32;
    const uint32_t type_size = 34;
    const uint32_t n_blocks = n_elements / block_size;
    const uint8_t *row = base + row_offset;
    for (uint32_t b = 0; b < n_blocks; b++) {
        const uint8_t *block = row + (uint64_t)b * type_size;
        uint16_t scale_bits;
        memcpy(&scale_bits, block, 2);
        float scale = (float)*(const __fp16 *)&scale_bits;
        const int8_t *qs = (const int8_t *)(block + 2);
        for (uint32_t j = 0; j < block_size; j++) {
            out[b * block_size + j] = scale * (float)qs[j];
        }
    }
    return 1;
}

static int l26f_embed_token(l26f_session *s, uint32_t token) {
    l26f_model *m = s->model;
    l26f_tensor *embd = l26f_model_find_tensor(m, "token_embd.weight");
    if (!embd) { fprintf(stderr, "l26f: no token_embd\n"); return 0; }
    if (token >= m->n_vocab) { fprintf(stderr, "l26f: token %u >= vocab %u\n", token, m->n_vocab); return 0; }

    const uint32_t n_embd = m->n_embd;
    const uint64_t row_bytes = (uint64_t)(n_embd / 32) * 34;
    const uint64_t offset = (uint64_t)token * row_bytes;
    if (embd->abs_offset + offset + row_bytes > m->size) {
        fprintf(stderr, "l26f: embedding OOB\n"); return 0;
    }

    float *data = (float *)malloc((uint64_t)n_embd * sizeof(float));
    if (!data) return 0;
    l26f_dequant_q8_0_row(m->map + embd->abs_offset, offset, data, n_embd);
    int ok = ds4_metal_tensor_write(s->hidden_1xN, 0, data, (uint64_t)n_embd * sizeof(float));
    free(data);
    return ok;
}

static int l26f_output_logits(l26f_session *s) {
    l26f_model *m = s->model;
    l26f_tensor *wt_norm_N   = l26f_model_find_tensor(m, "output_norm.weight");
    l26f_tensor *wt_out_NxV  = l26f_model_find_tensor(m, "output.weight");
    double XPROF_T0 = 0.0;
    if (!wt_norm_N || !wt_out_NxV) {
        fprintf(stderr, "l26f: missing output tensors\n"); return 0;
    }

    if (!XPROF_STANDALONE_BEGIN(&XPROF_T0)) return 0;
    if (!ds4_metal_rms_norm_weight_tensor(s->output_normed_1xN, s->hidden_1xN,
            m->map, m->size, wt_norm_N->abs_offset, m->n_embd, m->rms_norm_eps))
        return 0;
    if (!XPROF_STANDALONE_END(XPROF_LOGITS_NORM, 999, XPROF_T0)) return 0;

    if (!XPROF_STANDALONE_BEGIN(&XPROF_T0)) return 0;
    if (!l26f_metal_matvec_quant(s->logits_1xV, s->output_normed_1xN,
            m->map, m->size, wt_out_NxV->abs_offset,
            wt_out_NxV->dim[0], wt_out_NxV->dim[1], wt_out_NxV->type, 1))
        return 0;
    if (!XPROF_STANDALONE_END(XPROF_LOGITS_HEAD, 999, XPROF_T0)) return 0;

    return 1;
}

typedef struct {
    float temperature;
    int32_t top_k;
    float top_p;
    uint64_t seed;
} l26f_sample_params;

static uint64_t l26f_rng_next(uint64_t *state) {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    return *state;
}

static float l26f_rng_float(uint64_t *state) {
    return (float)(l26f_rng_next(state) & 0xFFFFFF) / (float)0x1000000;
}

typedef struct {
    float val;
    int32_t idx;
} l26f_logit_entry;

static int l26f_logit_cmp_desc(const void *a, const void *b) {
    float fa = ((const l26f_logit_entry *)a)->val;
    float fb = ((const l26f_logit_entry *)b)->val;
    return (fa < fb) - (fa > fb);
}

static int32_t l26f_sample(l26f_session *s, const l26f_sample_params *p, uint64_t *rng) {
    const uint32_t V = s->model->n_vocab;

    if (p->temperature <= 0.0f) {
        if (!l26f_metal_argmax(s->sample_idx_1xI, s->logits_1xV, V))
            return -1;
        int32_t result = -1;
        ds4_metal_tensor_read(s->sample_idx_1xI, 0, &result, sizeof(int32_t));
        return result;
    }

    float *logits = (float *)malloc((uint64_t)V * sizeof(float));
    if (!logits) return -1;
    ds4_metal_tensor_read(s->logits_1xV, 0, logits, (uint64_t)V * sizeof(float));

    for (uint32_t i = 0; i < V; i++) logits[i] /= p->temperature;

    l26f_logit_entry *entries = (l26f_logit_entry *)malloc((uint64_t)V * sizeof(l26f_logit_entry));
    for (uint32_t i = 0; i < V; i++) {
        entries[i].val = logits[i];
        entries[i].idx = (int32_t)i;
    }

    int32_t k = V;
    if (p->top_k > 0 && p->top_k < (int32_t)V) k = p->top_k;
    int32_t partial = (k < V) ? k : V;
    if (partial < V) {
        qsort(entries, V, sizeof(l26f_logit_entry), l26f_logit_cmp_desc);
    }

    float max_v = entries[0].val;
    for (int32_t i = 0; i < partial; i++)
        entries[i].val = expf(entries[i].val - max_v);

    if (p->top_p < 1.0f) {
        float cumsum = 0.0f, total = 0.0f;
        for (int32_t i = 0; i < partial; i++) total += entries[i].val;
        for (int32_t i = 0; i < partial; i++) {
            cumsum += entries[i].val;
            if (cumsum / total > p->top_p) {
                partial = i + 1;
                break;
            }
        }
    }

    float sum = 0.0f;
    for (int32_t i = 0; i < partial; i++) sum += entries[i].val;
    float r = l26f_rng_float(rng) * sum;
    int32_t result = entries[partial - 1].idx;
    float cumsum = 0.0f;
    for (int32_t i = 0; i < partial; i++) {
        cumsum += entries[i].val;
        if (cumsum >= r) { result = entries[i].idx; break; }
    }

    free(entries);
    free(logits);
    return result;
}

static int32_t l26f_output_greedy_token(l26f_session *s) {
    const uint32_t V = s->model->n_vocab;
    if (!ds4_metal_begin_commands()) return -1;
    if (!l26f_output_logits(s)) {
        (void)ds4_metal_end_commands();
        return -1;
    }
    if (!l26f_metal_argmax(s->sample_idx_1xI, s->logits_1xV, V)) {
        (void)ds4_metal_end_commands();
        return -1;
    }
    if (!ds4_metal_end_commands()) return -1;

    int32_t result = -1;
    if (!ds4_metal_tensor_read(s->sample_idx_1xI, 0, &result, sizeof(int32_t))) {
        return -1;
    }
    return result;
}

static int32_t l26f_argmax(l26f_session *s) {
    l26f_sample_params p = {0, 0, 1.0f, 0};
    uint64_t rng = 0;
    return l26f_sample(s, &p, &rng);
}

static int l26f_batch_layers(void) {
    static int initialized = 0;
    static int layers = 32;
    if (!initialized) {
        const char *env = getenv("L26F_BATCH_LAYERS");
        if (env && env[0]) {
            int v = atoi(env);
            if (v > 0 && v < 32) layers = v;
        }
        initialized = 1;
    }
    return layers;
}

// =========================================================================
// Prefill forward pass: T tokens through all 32 layers
// Layer-by-layer: sequential GLA/MLA per token, then batch MoE FFN
// =========================================================================

static int l26f_forward_pass_prefill(l26f_session *session, l26f_model *model,
        int32_t *tokens, int n_tokens) {
    const uint32_t n_embd = model->n_embd;
    const uint64_t row_bytes = (uint64_t)n_embd * sizeof(float);
    l26f_prefill_compute *p = session->prefill;
    if (!p || (uint32_t)n_tokens > p->max_tokens) {
        fprintf(stderr, "l26f: prefill not initialized or too many tokens (%d > %u)\n",
                n_tokens, p ? p->max_tokens : 0);
        return 0;
    }

    L26F_PREFILL_LOG("PREFILL: embedding %d tokens into hidden_TxN", n_tokens);
    for (int t = 0; t < n_tokens; t++) {
        if (!l26f_embed_token(session, (uint32_t)tokens[t])) {
            fprintf(stderr, "l26f: embed failed for prompt token %d\n", tokens[t]);
            return 0;
        }
        float *row_1xN = (float *)malloc(row_bytes);
        if (!row_1xN) return 0;
        ds4_metal_tensor_read(session->hidden_1xN, 0, row_1xN, row_bytes);
        { char _ctx[32]; snprintf(_ctx, sizeof(_ctx), "embed t=%d", t);
          L26F_NAN_CHECK("embed→hidden_1xN", row_1xN, n_embd, _ctx); }
        ds4_metal_tensor_write(p->hidden_TxN, (uint64_t)t * row_bytes, row_1xN, row_bytes);
        free(row_1xN);
    }
    L26F_PREFILL_CKPT("hidden_TxN_after_embed", p->hidden_TxN, (uint64_t)n_tokens * row_bytes, 255, -1);

    for (uint32_t il = 0; il < 32; il++) {
        L26F_PREFILL_LOG("PREFILL: layer %u (%s) n_tokens=%d",
                il, model->is_mla[il] ? "MLA" : "GLA", n_tokens);

        for (int t = 0; t < n_tokens; t++) {
            float *row_1xN = (float *)malloc(row_bytes);
            if (!row_1xN) return 0;
            ds4_metal_tensor_read(p->hidden_TxN, (uint64_t)t * row_bytes, row_1xN, row_bytes);
            ds4_metal_tensor_write(session->hidden_1xN, 0, row_1xN, row_bytes);
            free(row_1xN);

            int position = t;
            ds4_metal_tensor *inp_1xN = session->hidden_1xN;
            ds4_metal_tensor *out_1xN = session->comp.post_attn_1xN;

            if (model->is_mla[il]) {
                if (use_gpu_mla()) {
                    if (!l26f_mla_layer_gpu(model, il, position, session->mla_kv_gpu[il],
                                              session->mla_comp, inp_1xN, out_1xN)) {
                        fprintf(stderr, "MLA GPU layer %u token %d failed\n", il, t);
                        return 0;
                    }
                } else {
                    float *hidden_cpu_1xN = (float *)malloc(row_bytes);
                    float *hidden_out_1xN = (float *)malloc(row_bytes);
                    ds4_metal_tensor_read(inp_1xN, 0, hidden_cpu_1xN, row_bytes);
                    if (!l26f_mla_layer_cpu(model, il, position, session->mla_kv_cpu[il],
                                              hidden_cpu_1xN, hidden_out_1xN)) {
                        fprintf(stderr, "MLA CPU layer %u token %d failed\n", il, t);
                        free(hidden_cpu_1xN); free(hidden_out_1xN);
                        return 0;
                    }
                    ds4_metal_tensor_write(out_1xN, 0, hidden_out_1xN, row_bytes);
                    free(hidden_cpu_1xN);
                    free(hidden_out_1xN);
                }
            } else {
                if (!l26f_gla_layer(session, il, position, inp_1xN, out_1xN)) {
                    fprintf(stderr, "GLA layer %u token %d failed\n", il, t);
                    return 0;
                }
            }

            float *attn_row_1xN = (float *)malloc(row_bytes);
            if (!attn_row_1xN) return 0;
            ds4_metal_tensor_read(out_1xN, 0, attn_row_1xN, row_bytes);
            { char _ctx[32]; snprintf(_ctx, sizeof(_ctx), "L%u:t%d attn_out", il, t);
              L26F_NAN_CHECK("attn_out_1xN", attn_row_1xN, n_embd, _ctx); }
            ds4_metal_tensor_write(p->post_attn_TxN, (uint64_t)t * row_bytes,
                attn_row_1xN, row_bytes);
            free(attn_row_1xN);
        }

        if (il == 0) {
            L26F_PREFILL_LOG("PREFILL: L0 dense_ffn (per-token)");
            for (int t = 0; t < n_tokens; t++) {
                float *row_in_1xN = (float *)malloc(row_bytes);
                if (!row_in_1xN) return 0;
                ds4_metal_tensor_read(p->post_attn_TxN, (uint64_t)t * row_bytes,
                    row_in_1xN, row_bytes);
                ds4_metal_tensor_write(session->comp.post_attn_1xN, 0,
                    row_in_1xN, row_bytes);
                free(row_in_1xN);

                if (!l26f_dense_ffn(session, session->comp.post_attn_1xN, session->hidden_1xN)) {
                    fprintf(stderr, "Dense FFN layer 0 token %d failed\n", t);
                    return 0;
                }

                float *row_out_1xN = (float *)malloc(row_bytes);
                if (!row_out_1xN) return 0;
                ds4_metal_tensor_read(session->hidden_1xN, 0, row_out_1xN, row_bytes);
                { char _ctx[32]; snprintf(_ctx, sizeof(_ctx), "L%u:t%d ffn_out", il, t);
                  L26F_NAN_CHECK("dense_ffn_out", row_out_1xN, n_embd, _ctx); }
                ds4_metal_tensor_write(p->hidden_TxN, (uint64_t)t * row_bytes,
                    row_out_1xN, row_bytes);
                free(row_out_1xN);
            }
        } else {
            L26F_PREFILL_LOG("PREFILL: L%u moe_ffn_batch T=%d", il, n_tokens);
            L26F_PREFILL_CKPT("pre_moe_post_attn_TxN", p->post_attn_TxN,
                              (uint64_t)n_tokens * row_bytes, il, -1);
            if (!l26f_moe_ffn_batch(session, il, p->post_attn_TxN, p->hidden_TxN, n_tokens)) {
                fprintf(stderr, "Batch MoE FFN layer %u failed\n", il);
                return 0;
            }
            L26F_PREFILL_CKPT("post_moe_hidden_TxN", p->hidden_TxN,
                              (uint64_t)n_tokens * row_bytes, il, -1);
        }
    }

    L26F_PREFILL_LOG("PREFILL: done, copying last token to hidden_1xN");
    float *last_row_1xN = (float *)malloc(row_bytes);
    if (!last_row_1xN) return 0;
    ds4_metal_tensor_read(p->hidden_TxN, (uint64_t)(n_tokens - 1) * row_bytes,
        last_row_1xN, row_bytes);
    L26F_NAN_CHECK("final_hidden_1xN", last_row_1xN, n_embd, "final");
    ds4_metal_tensor_write(session->hidden_1xN, 0, last_row_1xN, row_bytes);
    free(last_row_1xN);

    return 1;
}

static int l26f_forward_pass(l26f_session *session, l26f_model *model, int position, bool verbose) {
    const uint64_t act_bytes = (uint64_t)model->n_embd * sizeof(float);
    const int batch_layers = l26f_batch_layers();
    int layers_in_batch = 0;

    if (!ds4_metal_gpu_profile_active()) {
        if (!ds4_metal_begin_commands()) {
            fprintf(stderr, "begin_commands failed\n");
            return 0;
        }
    }

    for (uint32_t il = 0; il < 32; il++) {
        char prof_label[64];
        snprintf(prof_label, sizeof(prof_label), "L%02u_%s",
                 il, model->is_mla[il] ? "MLA" : "GLA");
        ds4_metal_gpu_profile_begin(prof_label);

        if (model->is_mla[il]) {
            if (use_gpu_mla()) {
                ds4_metal_tensor *mla_out_1xN = session->comp.post_attn_1xN;
                double XPROF_T0 = 0.0;
                if (!XPROF_STAGE_BEGIN(&XPROF_T0)) return 0;
                if (!l26f_mla_layer_gpu(model, il, position, session->mla_kv_gpu[il],
                                          session->mla_comp, session->hidden_1xN, mla_out_1xN)) {
                    fprintf(stderr, "MLA GPU layer %u failed\n", il);
                    return 0;
                }
                if (!XPROF_STAGE_END(XPROF_MLA, il, XPROF_T0)) return 0;
#ifdef L26F_DBG_COMPARE
                if ((int)il == l26f_dbg_get_layer()) {
                    if (!XPROF_STAGE_BEGIN(&XPROF_T0)) return 0;
                    if (!l26f_moe_ffn_compare(session, il, mla_out_1xN, session->hidden_1xN)) {
                        fprintf(stderr, "MoE COMPARE layer %u failed\n", il);
                        return 0;
                    }
                    if (!XPROF_STAGE_END(XPROF_MOE, il, XPROF_T0)) return 0;
                } else
#elif defined(L26F_DBG_FUSED)
                if ((int)il == l26f_dbg_get_layer()) {
                    if (!XPROF_STAGE_BEGIN(&XPROF_T0)) return 0;
                    if (!l26f_moe_ffn_fused(session, il, mla_out_1xN, session->hidden_1xN)) {
                        fprintf(stderr, "MoE FUSED layer %u failed\n", il);
                        return 0;
                    }
                    if (!XPROF_STAGE_END(XPROF_MOE, il, XPROF_T0)) return 0;
                } else
#endif
                {
                    if (!XPROF_STAGE_BEGIN(&XPROF_T0)) return 0;
                    if (!(l26f_use_hybrid_moe() ?
                          l26f_moe_ffn_hybrid(session, il, mla_out_1xN, session->hidden_1xN) :
                          (l26f_use_fused_moe() ?
                           l26f_moe_ffn_fused_prod(session, il, mla_out_1xN, session->hidden_1xN) :
                           l26f_moe_ffn(session, il, mla_out_1xN, session->hidden_1xN)))) {
                        fprintf(stderr, "MoE FFN layer %u failed\n", il);
                        return 0;
                    }
                    if (!XPROF_STAGE_END(XPROF_MOE, il, XPROF_T0)) return 0;
                }
            } else {
                ds4_metal_tensor *out_1xN = session->comp.post_attn_1xN;
                ds4_metal_end_commands();
                float *hidden_cpu_1xN = (float *)malloc(act_bytes);
                float *hidden_out_1xN = (float *)malloc(act_bytes);
                ds4_metal_tensor_read(session->hidden_1xN, 0, hidden_cpu_1xN, act_bytes);
                if (!l26f_mla_layer_cpu(model, il, position, session->mla_kv_cpu[il],
                                          hidden_cpu_1xN, hidden_out_1xN)) {
                    fprintf(stderr, "MLA CPU layer %u failed\n", il);
                    free(hidden_cpu_1xN); free(hidden_out_1xN);
                    return 0;
                }
                ds4_metal_tensor_write(session->hidden_1xN, 0, hidden_out_1xN, act_bytes);
                ds4_metal_tensor_write(out_1xN, 0, hidden_out_1xN, act_bytes);
                free(hidden_cpu_1xN);
                free(hidden_out_1xN);
                ds4_metal_begin_commands();
                layers_in_batch = 0;
#ifdef L26F_DBG_COMPARE
                if ((int)il == l26f_dbg_get_layer()) {
                    double XPROF_T0 = 0.0;
                    if (!XPROF_STAGE_BEGIN(&XPROF_T0)) return 0;
                    if (!l26f_moe_ffn_compare(session, il, out_1xN, session->hidden_1xN)) {
                        fprintf(stderr, "MoE COMPARE layer %u failed\n", il);
                        return 0;
                    }
                    if (!XPROF_STAGE_END(XPROF_MOE, il, XPROF_T0)) return 0;
                } else
#elif defined(L26F_DBG_FUSED)
                if ((int)il == l26f_dbg_get_layer()) {
                    double XPROF_T0 = 0.0;
                    if (!XPROF_STAGE_BEGIN(&XPROF_T0)) return 0;
                    if (!l26f_moe_ffn_fused(session, il, out_1xN, session->hidden_1xN)) {
                        fprintf(stderr, "MoE FUSED layer %u failed\n", il);
                        return 0;
                    }
                    if (!XPROF_STAGE_END(XPROF_MOE, il, XPROF_T0)) return 0;
                } else
#endif
                {
                    double XPROF_T0 = 0.0;
                    if (!XPROF_STAGE_BEGIN(&XPROF_T0)) return 0;
                    if (!(l26f_use_hybrid_moe() ?
                          l26f_moe_ffn_hybrid(session, il, out_1xN, session->hidden_1xN) :
                          (l26f_use_fused_moe() ?
                           l26f_moe_ffn_fused_prod(session, il, out_1xN, session->hidden_1xN) :
                           l26f_moe_ffn(session, il, out_1xN, session->hidden_1xN)))) {
                        fprintf(stderr, "MoE FFN layer %u failed\n", il);
                        return 0;
                    }
                    if (!XPROF_STAGE_END(XPROF_MOE, il, XPROF_T0)) return 0;
                }
            }
            if (verbose) printf("  layer %u: MLA %s + MoE\n", il, use_gpu_mla() ? "GPU" : "CPU");
            layers_in_batch++;
            if (!ds4_metal_gpu_profile_active() && il + 1 < 32 && layers_in_batch >= batch_layers) {
                if (!ds4_metal_end_commands()) {
                    fprintf(stderr, "end_commands failed after layer %u\n", il);
                    return 0;
                }
                XREG_PRINT_LAYER_HIDDEN_1xN(il, session->hidden_1xN, act_bytes);
                if (!ds4_metal_begin_commands()) {
                    fprintf(stderr, "begin_commands failed after layer %u\n", il);
                    return 0;
                }
                layers_in_batch = 0;
            }
            continue;
        }

        ds4_metal_tensor *inp_1xN = session->hidden_1xN;
        ds4_metal_tensor *out_1xN = session->comp.post_attn_1xN;

        double XPROF_T0 = 0.0;
        if (!XPROF_STAGE_BEGIN(&XPROF_T0)) return 0;
        if (!l26f_gla_layer(session, il, position, inp_1xN, out_1xN)) {
            fprintf(stderr, "GLA layer %u failed\n", il);
            return 0;
        }
        if (!XPROF_STAGE_END(XPROF_GLA, il, XPROF_T0)) return 0;

        if (il == 0) {
            if (!XPROF_STAGE_BEGIN(&XPROF_T0)) return 0;
            if (!l26f_dense_ffn(session, out_1xN, session->hidden_1xN)) {
                fprintf(stderr, "Dense FFN layer 0 failed\n");
                return 0;
            }
            if (!XPROF_STAGE_END(XPROF_DENSE, il, XPROF_T0)) return 0;
            if (verbose) printf("  layer %u: GLA + dense FFN\n", il);
        } else {
#ifdef L26F_DBG_COMPARE
            if ((int)il == l26f_dbg_get_layer()) {
                if (!XPROF_STAGE_BEGIN(&XPROF_T0)) return 0;
                if (!l26f_moe_ffn_compare(session, il, out_1xN, session->hidden_1xN)) {
                    fprintf(stderr, "MoE COMPARE layer %u failed\n", il);
                    return 0;
                }
                if (!XPROF_STAGE_END(XPROF_MOE, il, XPROF_T0)) return 0;
            } else
#elif defined(L26F_DBG_FUSED)
            if ((int)il == l26f_dbg_get_layer()) {
                if (!XPROF_STAGE_BEGIN(&XPROF_T0)) return 0;
                if (!l26f_moe_ffn_fused(session, il, out_1xN, session->hidden_1xN)) {
                    fprintf(stderr, "MoE FUSED layer %u failed\n", il);
                    return 0;
                }
                if (!XPROF_STAGE_END(XPROF_MOE, il, XPROF_T0)) return 0;
            } else
#endif
            {
                if (!XPROF_STAGE_BEGIN(&XPROF_T0)) return 0;
                if (!(l26f_use_hybrid_moe() ?
                      l26f_moe_ffn_hybrid(session, il, out_1xN, session->hidden_1xN) :
                      (l26f_use_fused_moe() ?
                       l26f_moe_ffn_fused_prod(session, il, out_1xN, session->hidden_1xN) :
                       l26f_moe_ffn(session, il, out_1xN, session->hidden_1xN)))) {
                    fprintf(stderr, "MoE FFN layer %u failed\n", il);
                    return 0;
                }
                if (!XPROF_STAGE_END(XPROF_MOE, il, XPROF_T0)) return 0;
            }
            if (verbose && il < 8) printf("  layer %u: GLA + MoE\n", il);
        }

#ifdef L26F_DBG_FUSED
        {
            ds4_metal_end_commands();
            int nans;
            float sum = l26f_tensor_checksum(session->hidden_1xN, act_bytes, &nans);
            fprintf(stderr, "LAYER %2u hidden sum=%.4f nans=%d\n", il, sum, nans);
            ds4_metal_begin_commands();
            layers_in_batch = 0;
        }
#endif

        layers_in_batch++;
        if (!ds4_metal_gpu_profile_active() && il + 1 < 32 && layers_in_batch >= batch_layers) {
            if (!ds4_metal_end_commands()) {
                fprintf(stderr, "end_commands failed after layer %u\n", il);
                return 0;
            }
            XREG_PRINT_LAYER_HIDDEN_1xN(il, session->hidden_1xN, act_bytes);
            if (!ds4_metal_begin_commands()) {
                fprintf(stderr, "begin_commands failed after layer %u\n", il);
                return 0;
            }
            layers_in_batch = 0;
        }
    }

    if (!ds4_metal_end_commands()) {
        fprintf(stderr, "end_commands failed\n");
        return 0;
    }
    XREG_PRINT_LAYER_HIDDEN_1xN(31, session->hidden_1xN, act_bytes);
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [prompt_or_token] [n_gen] [temp] [top_k] [top_p]\n", argv[0]);
        fprintf(stderr, "  If prompt starts with a digit, treated as token ID\n");
        fprintf(stderr, "  Otherwise, tokenized as text\n");
        fprintf(stderr, "  temp=0 → greedy (argmax), default=0.8\n");
        return 1;
    }

    printf("Loading model...\n");
    l26f_model model;
    l26f_model_open(&model, argv[1]);

    printf("Loading tokenizer...\n");
    l26f_tokenizer *tok = l26f_tokenizer_from_model(&model);
    if (!tok) { fprintf(stderr, "Tokenizer load failed\n"); return 1; }
    printf("  vocab: %u tokens, %u merges, BOS=%d EOS=%d\n",
           tok->n_tokens, tok->n_merges, tok->bos_id, tok->eos_id);

    bool is_text = false;
    int32_t prompt_tokens[256];
    int n_prompt = 0;
    if (argc > 2) {
        const char *arg = argv[2];
        is_text = false;
        for (int i = 0; arg[i]; i++) {
            if (!isdigit((unsigned char)arg[i]) && arg[i] != '-') { is_text = true; break; }
        }
        if (is_text) {
            n_prompt = l26f_text_encode(tok, arg, prompt_tokens, 256);
            if (n_prompt == 0) { fprintf(stderr, "Encoding failed\n"); return 1; }
            printf("  encoded '%s' → %d tokens\n", arg, n_prompt);
        } else {
            prompt_tokens[0] = atoi(argv[2]);
            n_prompt = 1;
        }
    } else {
        prompt_tokens[0] = 1;
        n_prompt = 1;
    }
    int n_gen = argc > 3 ? atoi(argv[3]) : 16;

    l26f_sample_params sample_params = {
        .temperature = argc > 4 ? (float)atof(argv[4]) : 0.0f,
        .top_k       = argc > 5 ? atoi(argv[5]) : 40,
        .top_p       = argc > 6 ? (float)atof(argv[6]) : 0.95f,
        .seed        = 42
    };
    uint64_t rng = sample_params.seed;

    printf("Metal init...\n");
    if (!ds4_metal_init()) { fprintf(stderr, "Metal init failed\n"); return 1; }

    printf("Model map...\n");
    uint64_t model_tensor_off = 0;
    uint64_t model_tensor_bytes = model.size;
    if (!XMODEL_TENSOR_RANGE(&model, &model_tensor_off, &model_tensor_bytes)) {
        fprintf(stderr, "Model tensor range failed\n"); return 1;
    }
    if (!ds4_metal_set_model_map_range(model.map, model.size,
            model_tensor_off, model_tensor_bytes)) {
        fprintf(stderr, "Model map failed\n"); return 1;
    }

    l26f_session session;
    if (!l26f_session_init(&session, &model)) {
        fprintf(stderr, "Session init failed\n"); return 1;
    }

    printf("Prefilling %d prompt tokens...\n", n_prompt);
    if (n_prompt > 1) {
        session.prefill = l26f_prefill_alloc(model.n_embd, 1024);
        if (!session.prefill) {
            fprintf(stderr, "Prefill buffer alloc failed\n"); return 1;
        }
        for (int i = 0; i < n_prompt; i++) {
            char tokbuf_prompt[256];
            l26f_token_decode(tok, prompt_tokens[i], tokbuf_prompt, sizeof(tokbuf_prompt));
            printf("  [%d] token %d: \"%s\"\n", i, prompt_tokens[i], tokbuf_prompt);
        }
        if (!l26f_forward_pass_prefill(&session, &model, prompt_tokens, n_prompt)) {
            fprintf(stderr, "Batch prefill failed\n"); return 1;
        }
    } else {
        char tokbuf_prompt[256];
        l26f_token_decode(tok, prompt_tokens[0], tokbuf_prompt, sizeof(tokbuf_prompt));
        printf("  [0] token %d: \"%s\"\n", prompt_tokens[0], tokbuf_prompt);
        if (!l26f_embed_token(&session, (uint32_t)prompt_tokens[0])) {
            fprintf(stderr, "Embed failed for prompt token %d\n", prompt_tokens[0]);
            return 1;
        }
        if (!l26f_forward_pass(&session, &model, 0, false)) {
            fprintf(stderr, "Forward pass failed for prompt token %d\n", prompt_tokens[0]);
            return 1;
        }
    }

    printf("Generating %d tokens (temp=%.2f, top_k=%d, top_p=%.2f)...\n",
           n_gen, sample_params.temperature, sample_params.top_k, sample_params.top_p);
    int32_t current_token;
    {
        double XPROF_T0 = 0.0;
        if (sample_params.temperature <= 0.0f && !XPROF_ENABLED()) {
            current_token = l26f_output_greedy_token(&session);
        } else {
            if (!l26f_output_logits(&session)) {
                fprintf(stderr, "Output projection failed\n"); return 1;
            }
            if (!XPROF_STANDALONE_BEGIN(&XPROF_T0)) return 1;
            current_token = l26f_sample(&session, &sample_params, &rng);
            if (!XPROF_STANDALONE_END(XPROF_SAMPLE, 999, XPROF_T0)) return 1;
        }
        if (current_token < 0) {
            fprintf(stderr, "Sample failed\n"); return 1;
        }
    }
    char tokbuf[256];
    l26f_token_decode(tok, current_token, tokbuf, sizeof(tokbuf));
    printf("  -> token %d: \"%s\"\n", current_token, tokbuf);

    int32_t *generated = (int32_t *)malloc((n_gen + 1) * sizeof(int32_t));
    generated[0] = current_token;
    int n_generated = 1;

    int gpu_trace_gen = -1;
    char gpu_trace_path[1024] = {0};
    {
        const char *env = getenv("L26F_GPU_TRACE");
        if (env) {
            const char *colon = strchr(env, ':');
            if (colon) {
                gpu_trace_gen = atoi(env);
                size_t plen = strlen(colon + 1);
                if (plen >= sizeof(gpu_trace_path)) plen = sizeof(gpu_trace_path) - 1;
                memcpy(gpu_trace_path, colon + 1, plen);
                gpu_trace_path[plen] = '\0';
                fprintf(stderr, "GPU trace: will capture gen=%d -> %s\n", gpu_trace_gen, gpu_trace_path);
            } else {
                fprintf(stderr, "L26F_GPU_TRACE format: gen_index:/path/to/capture.gputrace\n");
            }
        }
    }

    int gpu_profile_gen = -1;
    {
        const char *env = getenv("L26F_GPU_PROFILE");
        if (env) {
            gpu_profile_gen = atoi(env);
            fprintf(stderr, "GPU profile: will profile gen=%d\n", gpu_profile_gen);
        }
    }

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    for (int gen = 0; gen < n_gen; gen++) {
        if (gen == gpu_trace_gen && gpu_trace_path[0]) {
            if (!ds4_metal_start_capture(gpu_trace_path)) {
                fprintf(stderr, "GPU trace start failed\n");
            }
        }
        if (gen == gpu_profile_gen) {
            ds4_metal_gpu_profile_init();
        }

        if (!l26f_embed_token(&session, (uint32_t)current_token)) {
            fprintf(stderr, "Embed failed for token %d\n", current_token);
            return 1;
        }

        if (!l26f_forward_pass(&session, &model, n_prompt + gen, false)) {
            fprintf(stderr, "Forward pass %d failed\n", gen);
            return 1;
        }

        double XPROF_T0 = 0.0;
        int32_t next_token;
        if (sample_params.temperature <= 0.0f && !XPROF_ENABLED()) {
            next_token = l26f_output_greedy_token(&session);
        } else {
            if (!l26f_output_logits(&session)) {
                fprintf(stderr, "Output projection failed\n"); return 1;
            }

            if (!XPROF_STANDALONE_BEGIN(&XPROF_T0)) return 1;
            next_token = l26f_sample(&session, &sample_params, &rng);
            if (!XPROF_STANDALONE_END(XPROF_SAMPLE, 999, XPROF_T0)) return 1;
        }
        if (next_token < 0) {
            fprintf(stderr, "Sample failed\n"); return 1;
        }

        if (gen == gpu_profile_gen) {
            ds4_metal_gpu_profile_print();
        }

        l26f_token_decode(tok, next_token, tokbuf, sizeof(tokbuf));
        printf("  -> token %d: \"%s\"\n", next_token, tokbuf);
        generated[n_generated++] = next_token;

        if (gen == gpu_trace_gen && gpu_trace_path[0]) {
            if (!ds4_metal_stop_capture()) {
                fprintf(stderr, "GPU trace stop failed\n");
            }
            gpu_trace_path[0] = '\0';
        }

        if (next_token == tok->eos_id) {
            printf("  (EOS reached)\n");
            break;
        }

        current_token = next_token;
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double elapsed = (t_end.tv_sec - t_start.tv_sec) + (t_end.tv_nsec - t_start.tv_nsec) / 1e9;
    printf("\nFull text: ");
    int32_t *all_tokens = (int32_t *)malloc((n_prompt + n_generated) * sizeof(int32_t));
    memcpy(all_tokens, prompt_tokens, n_prompt * sizeof(int32_t));
    memcpy(all_tokens + n_prompt, generated, n_generated * sizeof(int32_t));
    l26f_text_decode(tok, all_tokens, n_prompt + n_generated, tokbuf, sizeof(tokbuf));
    printf("%s\n", tokbuf);
    printf("\n%d prompt + %d generated tokens\n", n_prompt, n_generated);
    printf("Decode: %d tokens in %.3fs (%.1f tok/s)\n", n_generated, elapsed,
           n_generated > 0 ? (double)n_generated / elapsed : 0.0);

    l26f_prefill_free(session.prefill);
    l26f_session_free(&session);
    ds4_metal_cleanup();
    l26f_model_close(&model);
    l26f_tokenizer_close(tok);
    free(generated);
    free(all_tokens);
    return 0;
}
