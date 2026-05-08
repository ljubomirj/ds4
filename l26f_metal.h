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
// k, v, q, g:  [S, H, n_tokens]  where S=head_dim, H=n_heads
// state:       [S*S*H, n_seqs]   recurrent state (in/out: updated in-place)
// output:      [S*H, n_tokens + S*S*H*n_seqs]  activations + final state
//   Layout: first S*H*n_tokens elements = attention output
//           then  S*S*H*n_seqs elements = updated recurrent state
int l26f_metal_gla(
    l26f_metal_tensor       *output,       // [S*H, n_tokens + S*S*H*n_seqs]
    l26f_metal_tensor       *state,        // [S*S*H, n_seqs] recurrent state (in: prev, out: updated)
    const l26f_metal_tensor *k,
    const l26f_metal_tensor *v,
    const l26f_metal_tensor *q,
    const l26f_metal_tensor *g,
    uint32_t n_tokens,
    uint32_t n_seqs,
    uint32_t head_dim,
    uint32_t n_heads,
    float    scale);

// Quantized matvec (decode): routes to correct kernel by GGUF weight type
// weight_type: 1=F16, 8=Q8_0, 13=Q5_K, 14=Q6_K, 20=IQ4_NL
int l26f_metal_matvec_quant(
    l26f_metal_tensor       *dst,
    const l26f_metal_tensor *src1,
    const void              *model_map,
    uint64_t                 model_size,
    uint64_t                 weight_offset,
    uint64_t                 in_dim,
    uint64_t                 out_dim,
    uint32_t                 weight_type,
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
