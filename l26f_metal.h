#ifndef L26F_METAL_H
#define L26F_METAL_H

// Required includes (must appear before this header in the .c file):
//   #include <stdint.h>      // uint32_t, int32_t, uint64_t
//   #include "ds4_metal.h"   // ds4_metal_tensor (opaque device tensor)

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

// GPU trace capture (macOS only, requires Xcode to view .gputrace)
int  ds4_metal_start_capture(const char *path);
int  ds4_metal_stop_capture(void);

// GPU timestamp profiler (works without Xcode, uses per-CB GPU timestamps)
int  ds4_metal_gpu_profile_init(void);
int  ds4_metal_gpu_profile_begin(const char *label);
int  ds4_metal_gpu_profile_end(const char *label);
int  ds4_metal_gpu_profile_print(void);
int  ds4_metal_gpu_profile_active(void);

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

int l26f_metal_matvec_iq4_nl_residual(
    ds4_metal_tensor       *dst,
    const ds4_metal_tensor *src1,
    const ds4_metal_tensor *residual,
    const void              *model_map,
    uint64_t                 model_size,
    uint64_t                 weight_offset,
    uint64_t                 in_dim,
    uint64_t                 out_dim);

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

// Element-wise multiply by an f32 weight vector stored in the mmap'd model
int l26f_metal_mul_model_weight(
    ds4_metal_tensor       *out,
    const ds4_metal_tensor *x,
    const void              *model_map,
    uint64_t                 model_size,
    uint64_t                 weight_offset,
    uint32_t                 n);

// Sigmoid activation
int l26f_metal_sigmoid(
    ds4_metal_tensor       *out,
    const ds4_metal_tensor *x,
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

// NeoX RoPE batched: pairs dim i with i + n_dims/2 inside each vector
int l26f_metal_rope_neox_batch(
    ds4_metal_tensor       *dst,
    const ds4_metal_tensor *src,
    uint32_t                n_dims,
    uint32_t                n_vectors,
    int32_t                 position,
    float                   theta);

// Ling GLA Q/K preparation: per-head RMSNorm with q/k weights, then NeoX RoPE.
int l26f_metal_gla_qk_norm_rope(
    ds4_metal_tensor       *q_rope_1xN,
    ds4_metal_tensor       *k_rope_1xN,
    const ds4_metal_tensor *q_1xN,
    const ds4_metal_tensor *k_1xN,
    const void              *model_map,
    uint64_t                 model_size,
    uint64_t                 q_weight_offset,
    uint64_t                 k_weight_offset,
    uint32_t                 head_dim,
    uint32_t                 n_heads,
    int32_t                  position,
    float                    theta,
    float                    eps);

// Ling GLA epilogue: group RMSNorm, layer output norm, sigmoid gate, multiply.
int l26f_metal_gla_epilogue(
    ds4_metal_tensor       *out_1xN,
    const ds4_metal_tensor *gla_act_1xN,
    const ds4_metal_tensor *gate_1xN,
    const void              *model_map,
    uint64_t                 model_size,
    uint64_t                 layer_out_weight_offset,
    uint32_t                 n_embd,
    uint32_t                 n_groups,
    float                    eps);

// MLA attention (naive, per-head loop — kept for reference/comparison)
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

// MLA flash attention (tiled, online softmax, SIMD-parallel)
// Q_abs: [H, C] absorbed q_nope, Q_pe: [H, R] RoPE'd q_pe
// K: [n_cached, C+R] KV cache, first C = V latent, last R = RoPE'd k_pe
// O: [H, C] attention output per head
int l26f_metal_mla_fattn(
    ds4_metal_tensor       *attn_out,
    const ds4_metal_tensor *Q_abs,
    const ds4_metal_tensor *Q_pe,
    const ds4_metal_tensor *K,
    uint32_t                n_heads,
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

// Gather K selected experts into contiguous cache
int l26f_metal_gather_experts(
    ds4_metal_tensor       *cache,
    const void             *model_map,
    uint64_t                model_size,
    uint64_t                weight_base,
    uint64_t                weight_total,
    uint64_t                expert_bytes,
    uint64_t                expert_stride,
    const ds4_metal_tensor *expert_ids,
    uint32_t                n_experts);

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

// MoE expert routing plus selected expert byte-offset gather in one dispatch.
int l26f_metal_moe_route_offsets(
    ds4_metal_tensor       *router_logits,
    const void             *model_map,
    uint64_t                model_size,
    uint64_t                bias_offset,
    int32_t                 has_bias,
    ds4_metal_tensor       *selected_indices,
    ds4_metal_tensor       *selected_weights,
    const ds4_metal_tensor *all_off_gate,
    const ds4_metal_tensor *all_off_up,
    const ds4_metal_tensor *all_off_down,
    ds4_metal_tensor       *out_off_gate,
    ds4_metal_tensor       *out_off_up,
    ds4_metal_tensor       *out_off_down,
    uint32_t                n_expert,
    uint32_t                n_groups,
    uint32_t                n_exp_per_group,
    uint32_t                n_top_groups,
    uint32_t                n_selected,
    float                   w_scale);

int l26f_metal_moe_route_batch(
    ds4_metal_tensor       *router_logits_TxE,
    const void             *model_map,
    uint64_t                model_size,
    uint64_t                bias_offset,
    int32_t                 has_bias,
    ds4_metal_tensor       *selected_indices_TxK,
    ds4_metal_tensor       *selected_weights_TxK,
    uint32_t                n_expert,
    uint32_t                n_groups,
    uint32_t                n_exp_per_group,
    uint32_t                n_top_groups,
    uint32_t                n_selected,
    float                   w_scale,
    uint32_t                n_tokens);

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
    const ds4_metal_tensor *input,
    int                     per_expert_input);

// Fused MoE gate+up matvec (IQ4_NL): computes gate AND up matvecs in one dispatch.
// Saves reading input twice and reduces dispatch overhead.
int l26f_metal_fused_moe_iq4nl_gate_up(
    ds4_metal_tensor       *gate_out,
    ds4_metal_tensor       *up_out,
    const void             *model_map,
    uint64_t                model_size,
    const ds4_metal_tensor *offsets_gate,
    const ds4_metal_tensor *offsets_up,
    uint64_t                weight_base_gate,
    uint64_t                total_gate_bytes,
    uint64_t                weight_base_up,
    uint64_t                total_up_bytes,
    uint32_t                n_experts,
    uint32_t                in_dim,
    uint32_t                out_rows,
    const ds4_metal_tensor *input);

// Shared expert gate+up+swiglu fusion (IQ4_NL): 3 dispatches → 1
int l26f_metal_shared_gate_up_swiglu_iq4nl(
    ds4_metal_tensor       *mid_out,
    const void             *model_map,
    uint64_t                model_size,
    uint64_t                gate_offset,
    uint64_t                gate_bytes,
    uint64_t                up_offset,
    uint64_t                up_bytes,
    uint32_t                in_dim,
    uint32_t                out_dim,
    const ds4_metal_tensor *input);

// Fused MoE expert gate+up+swiglu (IQ4_NL): gate+up with inline swiglu, 2→1 dispatch
int l26f_metal_fused_moe_iq4nl_gate_up_swiglu(
    ds4_metal_tensor       *mid_out,
    const void             *model_map,
    uint64_t                model_size,
    const ds4_metal_tensor *offsets_gate,
    const ds4_metal_tensor *offsets_up,
    uint64_t                weight_base_gate,
    uint64_t                total_gate_bytes,
    uint64_t                weight_base_up,
    uint64_t                total_up_bytes,
    uint32_t                n_experts,
    uint32_t                in_dim,
    uint32_t                out_rows,
    const ds4_metal_tensor *input);

// Fused MoE expert matvec (IQ4_NL down): cached variant
int l26f_metal_fused_moe_iq4nl_cached(
    ds4_metal_tensor       *dst,
    ds4_metal_tensor       *weights,
    const ds4_metal_tensor *offsets,
    uint32_t                n_experts,
    uint32_t                in_dim,
    uint32_t                out_rows,
    const ds4_metal_tensor *input,
    int                     per_expert_input);

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

// Cached variants: weights already in contiguous GPU buffers (expert cache)
int l26f_metal_fused_moe_iq4nl_gate_up_swiglu_cached(
    ds4_metal_tensor       *mid_out,
    ds4_metal_tensor       *weights_gate,
    ds4_metal_tensor       *weights_up,
    const ds4_metal_tensor *offsets_gate,
    const ds4_metal_tensor *offsets_up,
    uint32_t                n_experts,
    uint32_t                in_dim,
    uint32_t                out_rows,
    const ds4_metal_tensor *input);

int l26f_metal_fused_moe_q5k_cached(
    ds4_metal_tensor       *dst,
    ds4_metal_tensor       *weights,
    const ds4_metal_tensor *offsets,
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

// Fused accumulate plus residual add:
// out[j] = residual[j] + shared[j] + sum_e(weight[e] * down_out[e][j])
int l26f_metal_fused_accum_residual(
    ds4_metal_tensor       *out,
    const ds4_metal_tensor *residual,
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

// Fused gather: gate + up + down in one dispatch
int l26f_metal_gather_offsets_3(
    ds4_metal_tensor       *sel_idx,
    const ds4_metal_tensor *all_off_gate,
    const ds4_metal_tensor *all_off_up,
    const ds4_metal_tensor *all_off_down,
    ds4_metal_tensor       *out_off_gate,
    ds4_metal_tensor       *out_off_up,
    ds4_metal_tensor       *out_off_down,
    uint32_t                K);

// GPU-side argmax: find index of maximum value in a 1D float tensor.
// Result written as int32_t to dst.  Eliminates CPU logits readback.
int l26f_metal_argmax(
    ds4_metal_tensor       *dst,
    const ds4_metal_tensor *src,
    uint32_t                n);

// Group normalization (same as group RMS norm with mean subtraction).
// src layout: [ne00, ne01, ne02], ngrp groups splitting ne02.
int l26f_metal_group_norm(
    ds4_metal_tensor       *dst,
    const ds4_metal_tensor *src,
    uint32_t                ne00,
    uint32_t                ne01,
    uint32_t                ne02,
    uint32_t                ngrp,
    float                   eps);

// Gated Delta Net (GDN) — recurrent attention with delta rule.
// q, k, v: [S_v, H, n_tokens]  state: [S_v*S_v*H, n_seqs]
// output: [S_v, H*n_tokens*n_seqs] + final state
int l26f_metal_gdn(
    ds4_metal_tensor       *dst,
    const ds4_metal_tensor *q,
    const ds4_metal_tensor *k,
    const ds4_metal_tensor *v,
    const ds4_metal_tensor *g,
    const ds4_metal_tensor *b,
    const ds4_metal_tensor *s,
    uint32_t                S_v,
    uint32_t                H,
    uint32_t                n_tokens,
    uint32_t                n_seqs,
    uint32_t                G);

// Mat-mat SGEMM for MoE prefill (simdgroup_half8x8 tiled, IQ4_NL).
// Two-pass: map0 builds expert→token index, then mul_mm_id does tiled SGEMM.
// n_tokens_per_expert: output from map0, must be pre-allocated (n_expert uint32_t values).
// ids_buffer: output from map0, must be pre-allocated (n_expert * n_tokens int32_t values).
int l26f_metal_mul_mm_id_map0(
    ds4_metal_tensor       *src2_ids,
    ds4_metal_tensor       *tokens_per_expert,
    ds4_metal_tensor       *ids_buffer,
    uint32_t                n_expert,
    uint32_t                n_tokens);

int l26f_metal_mul_mm_id_iq4nl(
    ds4_metal_tensor       *dst,
    const void             *model_map,
    uint64_t                model_size,
    uint64_t                weight_offset,
    uint64_t                weight_bytes,
    uint32_t                n_expert,
    uint32_t                in_dim,
    uint32_t                out_dim,
    const ds4_metal_tensor *input,
    const ds4_metal_tensor *tokens_per_expert,
    const ds4_metal_tensor *ids_buffer,
    uint32_t                n_tokens);

int l26f_metal_mul_mm_id_q5k(
    ds4_metal_tensor       *dst,
    const void             *model_map,
    uint64_t                model_size,
    uint64_t                weight_offset,
    uint64_t                weight_bytes,
    uint32_t                n_expert,
    uint32_t                in_dim,
    uint32_t                out_dim,
    const ds4_metal_tensor *input,
    const ds4_metal_tensor *tokens_per_expert,
    const ds4_metal_tensor *ids_buffer,
    uint32_t                n_tokens);

int l26f_metal_batch_accum(
    ds4_metal_tensor       *moe_out_TxN,
    const ds4_metal_tensor *expert_down_TxKxN,
    const ds4_metal_tensor *sel_wt_TxK,
    const ds4_metal_tensor *shared_out_TxN,
    uint32_t                n_tokens,
    uint32_t                n_experts,
    uint32_t                n_elements);

int l26f_metal_batch_swiglu(
    ds4_metal_tensor       *mid,
    const ds4_metal_tensor *gate,
    const ds4_metal_tensor *up,
    uint32_t                n);

#endif
