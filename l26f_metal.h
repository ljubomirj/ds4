#ifndef L26F_METAL_H
#define L26F_METAL_H

#include <stdbool.h>
#include <stdint.h>
#include "ds4_metal.h"

// Opaque device tensor
typedef struct ds4_metal_tensor ds4_metal_tensor;

// Device lifecycle
int  l26f_metal_init(void);
void l26f_metal_cleanup(void);

// Tensor management
ds4_metal_tensor *ds4_metal_tensor_alloc(uint64_t bytes);
ds4_metal_tensor *ds4_metal_tensor_view(const ds4_metal_tensor *base,
                                           uint64_t offset, uint64_t bytes);
void ds4_metal_tensor_free(ds4_metal_tensor *tensor);
uint64_t ds4_metal_tensor_bytes(const ds4_metal_tensor *tensor);
void *ds4_metal_tensor_contents(ds4_metal_tensor *tensor);
int  ds4_metal_tensor_write(ds4_metal_tensor *tensor, uint64_t offset,
                              const void *data, uint64_t bytes);
int  ds4_metal_tensor_read(const ds4_metal_tensor *tensor, uint64_t offset,
                             void *data, uint64_t bytes);
int  ds4_metal_tensor_fill(ds4_metal_tensor *tensor, float value);

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
    ds4_metal_tensor       *output,       // [S*H, n_tokens + S*S*H*n_seqs]
    ds4_metal_tensor       *state,        // [S*S*H, n_seqs] recurrent state (in: prev, out: updated)
    const ds4_metal_tensor *k,
    const ds4_metal_tensor *v,
    const ds4_metal_tensor *q,
    const ds4_metal_tensor *g,
    uint32_t n_tokens,
    uint32_t n_seqs,
    uint32_t head_dim,
    uint32_t n_heads,
    float    scale);

// Quantized matvec (decode): routes to correct kernel by GGUF weight type
// weight_type: 1=F16, 8=Q8_0, 13=Q5_K, 14=Q6_K, 20=IQ4_NL
int l26f_metal_matvec_quant(
    ds4_metal_tensor       *dst,
    const ds4_metal_tensor *src1,
    const void              *model_map,
    uint64_t                 model_size,
    uint64_t                 weight_offset,
    uint64_t                 in_dim,
    uint64_t                 out_dim,
    uint32_t                 weight_type,
    uint64_t                 n_tok);

// RMS Norm (with weight)
int l26f_metal_rms_norm_weight(
    ds4_metal_tensor       *out,
    const ds4_metal_tensor *x,
    const void              *model_map,
    uint64_t                 model_size,
    uint64_t                 weight_offset,
    uint32_t                 n,
    float                    eps);

// SiLU activation
int l26f_metal_silu(
    ds4_metal_tensor       *out,
    const ds4_metal_tensor *x,
    uint32_t                 n);

// Element-wise multiply
int l26f_metal_mul(
    ds4_metal_tensor       *out,
    const ds4_metal_tensor *a,
    const ds4_metal_tensor *b,
    uint32_t                 n);

// Element-wise add
int l26f_metal_add(
    ds4_metal_tensor       *out,
    const ds4_metal_tensor *a,
    const ds4_metal_tensor *b,
    uint32_t                 n);

// Token embedding lookup
int l26f_metal_embed_tokens(
    ds4_metal_tensor       *out,
    const uint32_t          *tokens,
    const void              *model_map,
    uint64_t                 model_size,
    uint64_t                 weight_offset,
    uint32_t                 n_vocab,
    uint32_t                 n_tokens,
    uint32_t                 n_embd);

// Memory reporting
void l26f_metal_print_memory_report(const char *label);

// --- MLA Kernels ---

// RoPE: apply rotary position embedding to a single vector [n_dims]
int l26f_metal_rope(
    ds4_metal_tensor       *dst,
    const ds4_metal_tensor *src,
    uint32_t                n_dims,
    int32_t                 position,
    float                   theta);

// RoPE batched: apply to [n_vectors * n_dims] (one RoPE per vector)
int l26f_metal_rope_batch(
    ds4_metal_tensor       *dst,
    const ds4_metal_tensor *src,
    uint32_t                n_dims,
    uint32_t                n_vectors,
    int32_t                 position,
    float                   theta);

// MLA attention: MQA dot-product + softmax + weighted sum
// q_absorbed: [H * kv_lora_rank], q_pe: [H * n_rot], kv_cache: [n_cached * kv_dim]
// attn_out: [H * kv_lora_rank]
int l26f_mla_attn(
    ds4_metal_tensor       *attn_out,
    const ds4_metal_tensor *q_absorbed,
    const ds4_metal_tensor *q_pe,
    const ds4_metal_tensor *kv_cache,
    uint32_t                n_heads,
    uint32_t                kv_lora_rank,
    uint32_t                n_rot,
    uint32_t                n_cached,
    float                   scale);

// Batched IQ4_NL matvec: for each of n_heads, do [out_rows, in_dim] IQ4_NL matvec
// Input layout: [n_heads * in_dim]  (contiguous per head)
// Output layout: [n_heads * out_rows]
// Weights: model_map + weight_offset, with head_stride bytes between head slices
int l26f_metal_batch_iq4_nl_matvec(
    ds4_metal_tensor       *dst,
    const void              *model_map,
    uint64_t                 model_size,
    uint64_t                 weight_offset,
    uint64_t                 in_dim,
    uint64_t                 out_rows,
    uint32_t                 n_heads,
    uint64_t                 head_stride,
    const ds4_metal_tensor *input);

// AXPY: dst[i] += alpha * src[i]
int l26f_metal_axpy(
    ds4_metal_tensor       *dst,
    const ds4_metal_tensor *src,
    float                   alpha,
    uint32_t                n);

// Strided extract: extract n_vectors slices of slice_len from src with stride src_stride
// dst[v * slice_len + j] = src[v * src_stride + src_offset + j]
int l26f_metal_strided_extract(
    ds4_metal_tensor       *dst,
    const ds4_metal_tensor *src,
    uint32_t                n_vectors,
    uint32_t                slice_len,
    uint32_t                src_stride,
    uint32_t                src_offset);

// KV cache append: append kv_cmpr[C] + k_pe[R] to cache at position n_cached
int l26f_metal_kv_append(
    ds4_metal_tensor       *cache,
    const ds4_metal_tensor *kv_cmpr,
    const ds4_metal_tensor *k_pe,
    uint32_t                kv_lora_rank,
    uint32_t                n_rot,
    uint32_t                n_cached);

// MoE expert routing: softmax + group scoring + top-8 selection on GPU
int l26f_metal_moe_route(
    ds4_metal_tensor       *router_logits,
    const void             *model_map,
    uint64_t                model_size,
    uint64_t                bias_offset,
    int32_t                 has_bias,
    ds4_metal_tensor       *selected_indices,
    ds4_metal_tensor       *selected_weights,
    uint32_t                n_expert,
    uint32_t                n_groups,
    uint32_t                n_exp_per_group,
    uint32_t                n_top_groups,
    uint32_t                n_selected,
    float                   w_scale);

// Fused MoE expert matvec (IQ4_NL): all K experts in one dispatch
// offsets: [K] uint64 expert weight offsets (GPU buffer)
int l26f_metal_fused_moe_iq4nl(
    ds4_metal_tensor       *dst,
    const void             *model_map,
    uint64_t                model_size,
    const ds4_metal_tensor *offsets,
    uint64_t                weight_base,
    uint64_t                total_expert_bytes,
    uint32_t                n_experts,
    uint32_t                in_dim,
    uint32_t                out_rows,
    const ds4_metal_tensor *input);

// Fused MoE expert matvec (Q5_K): all K experts in one dispatch
int l26f_metal_fused_moe_q5k(
    ds4_metal_tensor       *dst,
    const void             *model_map,
    uint64_t                model_size,
    const ds4_metal_tensor *offsets,
    uint64_t                weight_base,
    uint64_t                total_expert_bytes,
    uint32_t                n_experts,
    uint32_t                in_dim,
    uint32_t                out_rows,
    const ds4_metal_tensor *input);

// Fused swiglu: mid[e] = gate[e] * sigmoid(gate[e]) * up[e] for K experts
int l26f_metal_fused_swiglu(
    ds4_metal_tensor       *mid,
    const ds4_metal_tensor *gate,
    const ds4_metal_tensor *up,
    uint32_t                n_experts,
    uint32_t                n_elements);

// Fused accumulate: moe_out[j] = sum_e(weight[e] * down_out[e][j]) + shared[j]
int l26f_metal_fused_accum(
    ds4_metal_tensor       *moe_out,
    const ds4_metal_tensor *expert_down,
    const ds4_metal_tensor *weights,
    const ds4_metal_tensor *shared_out,
    uint32_t                n_experts,
    uint32_t                n_elements);

// Gather expert offsets: map K selected indices to K weight offsets
int l26f_metal_gather_offsets(
    ds4_metal_tensor       *sel_idx,
    const ds4_metal_tensor *all_offsets,
    ds4_metal_tensor       *out_offsets,
    uint32_t                K);

#endif
