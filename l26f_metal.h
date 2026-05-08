#ifndef L26F_METAL_H
#define L26F_METAL_H

#include <stdbool.h>
#include <stdint.h>

// Opaque device tensor
typedef struct l26f_metal_tensor l26f_metal_tensor;

// Device lifecycle
int  l26f_metal_init(void);
void l26f_metal_cleanup(void);

// Tensor management
l26f_metal_tensor *l26f_metal_tensor_alloc(uint64_t bytes);
l26f_metal_tensor *l26f_metal_tensor_view(const l26f_metal_tensor *base,
                                           uint64_t offset, uint64_t bytes);
void l26f_metal_tensor_free(l26f_metal_tensor *tensor);
uint64_t l26f_metal_tensor_bytes(const l26f_metal_tensor *tensor);
void *l26f_metal_tensor_contents(l26f_metal_tensor *tensor);
int  l26f_metal_tensor_write(l26f_metal_tensor *tensor, uint64_t offset,
                              const void *data, uint64_t bytes);
int  l26f_metal_tensor_read(const l26f_metal_tensor *tensor, uint64_t offset,
                             void *data, uint64_t bytes);
int  l26f_metal_tensor_fill(l26f_metal_tensor *tensor, float value);

// Model mapping (zero-copy from mmap)
int  l26f_metal_set_model_map(const void *model_map, uint64_t model_size);

// Command batching
int  l26f_metal_begin_commands(void);
int  l26f_metal_flush_commands(void);
int  l26f_metal_end_commands(void);
int  l26f_metal_synchronize(void);

// --- Compute Kernels ---

// GLA: Gated Linear Attention
//
// k, v, q, g:  [S, H, n_tokens]  where S=head_dim=128, H=n_heads=32
// state:       [S*S*H, n_seqs]   recurrent state (in-place: read old, write new)
// output:      [S*H, n_tokens]   attention output
// state_out:   [S*S*H, n_seqs]   updated recurrent state
int l26f_metal_gla(
    l26f_metal_tensor       *output,
    l26f_metal_tensor       *state_out,
    const l26f_metal_tensor *k,
    const l26f_metal_tensor *v,
    const l26f_metal_tensor *q,
    const l26f_metal_tensor *g,
    const l26f_metal_tensor *state_in,
    uint32_t n_tokens,
    uint32_t n_seqs,
    uint32_t head_dim,
    uint32_t n_heads,
    float    scale);

// Quantized matvec (decode): src0[N,M] * src1[M,R,R] -> dst[M,R,R]
// weight_type: 0=f16, 1=q8_0, 2=q5_k, 3=q6_k, 4=iq4_nl
int l26f_metal_matvec(
    l26f_metal_tensor       *dst,
    const l26f_metal_tensor *src1,        // input activation [M, R, R]
    const void              *model_map,
    uint64_t                 model_size,
    uint64_t                 weight_offset,
    uint64_t                 in_dim,       // N
    uint64_t                 out_dim,      // M
    int                      weight_type,
    uint64_t                 n_tok);

// RMS Norm (with weight)
int l26f_metal_rms_norm_weight(
    l26f_metal_tensor       *out,
    const l26f_metal_tensor *x,
    const void              *model_map,
    uint64_t                 model_size,
    uint64_t                 weight_offset,
    uint32_t                 n,
    float                    eps);

// SiLU activation
int l26f_metal_silu(
    l26f_metal_tensor       *out,
    const l26f_metal_tensor *x,
    uint32_t                 n);

// Element-wise multiply
int l26f_metal_mul(
    l26f_metal_tensor       *out,
    const l26f_metal_tensor *a,
    const l26f_metal_tensor *b,
    uint32_t                 n);

// Element-wise add
int l26f_metal_add(
    l26f_metal_tensor       *out,
    const l26f_metal_tensor *a,
    const l26f_metal_tensor *b,
    uint32_t                 n);

// Token embedding lookup
int l26f_metal_embed_tokens(
    l26f_metal_tensor       *out,
    const uint32_t          *tokens,
    const void              *model_map,
    uint64_t                 model_size,
    uint64_t                 weight_offset,
    uint32_t                 n_vocab,
    uint32_t                 n_tokens,
    uint32_t                 n_embd);

// Memory reporting
void l26f_metal_print_memory_report(const char *label);

#endif
