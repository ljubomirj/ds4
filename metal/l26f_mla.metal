// L26F MLA (Multi-head Latent Attention) Metal Kernels
//
// Operations for GPU-accelerated MLA decode:
//   1. RoPE — rotary position embedding
//   2. Absorption — batched IQ4_NL matvec (wk_b × q_nope for each head)
//   3. Attention — MQA dot-product scores + softmax + weighted sum
//   4. V decompression — batched IQ4_NL matvec (wv_b × attn_out for each head)

// ---- RoPE ----
// Apply rotary position embedding to a contiguous vector.
// Each thread handles one pair (x[i], x[i+1]).
//
// Args:
//   x:     [n_dims] float, in-place (or dst != src for out-of-place)
//   dst:   [n_dims] float output
//   pos:   position in sequence
//   theta: base frequency (10000.0)

typedef struct {
    int32_t n_dims;
    int32_t position;
    float   theta;
} l26f_kargs_rope;

kernel void kernel_l26f_rope(
        constant l26f_kargs_rope & args [[buffer(0)]],
        device const float        * src [[buffer(1)]],
        device       float        * dst [[buffer(2)]],
        uint gid [[thread_position_in_grid]])
{
    const int i = (int)gid * 2;
    if (i + 1 >= args.n_dims) return;

    float freq = 1.0f / pow(args.theta, (float)i / (float)args.n_dims);
    float angle = (float)args.position * freq;
    float cos_a = cos(angle);
    float sin_a = sin(angle);
    float x0 = src[i];
    float x1 = src[i + 1];
    dst[i]     = x0 * cos_a - x1 * sin_a;
    dst[i + 1] = x0 * sin_a + x1 * cos_a;
}

// Batched RoPE: apply to multiple head vectors packed contiguously.
// Layout: x = [n_vectors * n_dims], each vector gets RoPE applied independently.
// grid = (n_dims/2, n_vectors, 1)

typedef struct {
    int32_t n_dims;
    int32_t n_vectors;
    int32_t position;
    float   theta;
} l26f_kargs_rope_batch;

kernel void kernel_l26f_rope_batch(
        constant l26f_kargs_rope_batch & args [[buffer(0)]],
        device const float              * src [[buffer(1)]],
        device       float              * dst [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]])
{
    const int d = (int)gid.x * 2;
    const int v = (int)gid.y;
    if (d + 1 >= args.n_dims || v >= args.n_vectors) return;

    const int idx = v * args.n_dims + d;
    float freq = 1.0f / pow(args.theta, (float)d / (float)args.n_dims);
    float angle = (float)args.position * freq;
    float cos_a = cos(angle);
    float sin_a = sin(angle);
    float x0 = src[idx];
    float x1 = src[idx + 1];
    dst[idx]     = x0 * cos_a - x1 * sin_a;
    dst[idx + 1] = x0 * sin_a + x1 * cos_a;
}

// ---- MLA Attention (MQA) ----
//
// Multi-Query Attention for MLA:
//   - H query heads, 1 KV head (shared compressed KV)
//   - For each head h:
//     Q_h = concat(q_absorbed[h][0:C], q_pe[h][0:R])  → [CR]
//     For each cached token t:
//       score[t] = dot(Q_h[0:C], K_cache[t][0:C]) + dot(Q_h[C:CR], K_cache[t][C:CR])
//     weights = softmax(scores * scale)
//     attn_out[h][0:C] = sum_t weights[t] * K_cache[t][0:C]
//
// KV cache layout: [max_seq * kv_dim] where kv_dim = C + R = 576
//   K_cache[t][0:C] = compressed KV latent
//   K_cache[t][C:CR] = RoPE'd k_pe
//
// Input:
//   q_absorbed: [H * C] — absorbed query (already contains q_nope absorbed into wk_b)
//   q_pe:       [H * R] — RoPE'd positional query
//   kv_cache:   [n_cached * CR] — full KV cache
// Output:
//   attn_out:   [H * C] — attention output per head (weighted sum of V=C part of cache)
//
// We dispatch one threadgroup per head.
// Each threadgroup loads Q for its head, computes scores, softmax, weighted sum.

typedef struct {
    int32_t n_heads;
    int32_t kv_lora_rank;
    int32_t n_rot;
    int32_t n_cached;
    float   scale;
} l26f_kargs_mla_attn;

kernel void kernel_l26f_mla_attn(
        constant l26f_kargs_mla_attn & args   [[buffer(0)]],
        device const float            * q_abs  [[buffer(1)]],
        device const float            * q_pe   [[buffer(2)]],
        device const float            * k_cache [[buffer(3)]],
        device       float            * out    [[buffer(4)]],
        uint gid [[thread_position_in_grid]])
{
    const int h = (int)gid;
    if (h >= args.n_heads) return;

    const int C = args.kv_lora_rank;
    const int R = args.n_rot;
    const int CR = C + R;
    const int n_cached = args.n_cached;
    const float scale = args.scale;

    device const float * q_abs_h = q_abs + h * C;
    device const float * q_pe_h  = q_pe  + h * R;

    // Compute attention scores
    // For small n_cached, this is fast per-thread
    float scores[4096];
    float max_score = -1e30f;

    for (int t = 0; t < n_cached; t++) {
        device const float * k_t = k_cache + (int64_t)t * CR;
        float dot = 0.0f;
        for (int d = 0; d < C; d++) {
            dot += q_abs_h[d] * k_t[d];
        }
        for (int d = 0; d < R; d++) {
            dot += q_pe_h[d] * k_t[C + d];
        }
        scores[t] = dot * scale;
        if (scores[t] > max_score) max_score = scores[t];
    }

    // Softmax
    float sum_exp = 0.0f;
    for (int t = 0; t < n_cached; t++) {
        scores[t] = exp(scores[t] - max_score);
        sum_exp += scores[t];
    }
    float inv_sum = 1.0f / sum_exp;
    for (int t = 0; t < n_cached; t++) {
        scores[t] *= inv_sum;
    }

    // Weighted sum of V (= C part of KV cache)
    device float * out_h = out + h * C;
    for (int d = 0; d < C; d++) {
        out_h[d] = 0.0f;
    }
    for (int t = 0; t < n_cached; t++) {
        device const float * v_t = k_cache + (int64_t)t * CR;
        float w = scores[t];
        for (int d = 0; d < C; d++) {
            out_h[d] += w * v_t[d];
        }
    }
}

// ---- Batched IQ4_NL matvec for absorption and V decompression ----
//
// For absorption:
//   wk_b layout: [P, C, H] = [128, 512, 32] IQ4_NL
//   For each head h: result[h][0:C] = wk_b[h] × q_nope[h][0:P]
//   P=128 (qk_nope), C=512 (kv_lora_rank), H=32 (n_heads)
//
// For V decompression:
//   wv_b layout: [C, P, H] = [512, 128, 32] IQ4_NL
//   For each head h: result[h][0:P] = wv_b[h] × attn_out[h][0:C]
//   C=512 (kv_lora_rank), P=128 (head_dim_v), H=32 (n_heads)
//
// Each head slice: [out_rows, in_dim] IQ4_NL
// head_stride = out_rows * (in_dim / 32) * 18  (IQ4_NL: 32 elements/block, 18 bytes/block)
//
// We dispatch one thread per output element (per head).
// Each thread dequantizes one row of IQ4_NL weights and dots with the input.

typedef struct {
    int32_t n_heads;
    int32_t in_dim;
    int32_t out_rows;
    uint64_t head_stride;
} l26f_kargs_batch_iq4_nl;

kernel void kernel_l26f_batch_iq4_nl_matvec(
        constant l26f_kargs_batch_iq4_nl & args  [[buffer(0)]],
        device const char               * weights [[buffer(1)]],
        device const float              * input   [[buffer(2)]],
        device       float              * output  [[buffer(3)]],
        uint gid [[thread_position_in_grid]])
{
    const int total = args.n_heads * args.out_rows;
    const int idx = (int)gid;
    if (idx >= total) return;

    const int h = idx / args.out_rows;
    const int row = idx % args.out_rows;

    device const block_iq4_nl * w_head = (device const block_iq4_nl *)
        (weights + (uint64_t)h * args.head_stride);
    const int n_blocks = args.in_dim / 32;

    device const float * in_h = input + (uint64_t)h * args.in_dim;

    float sum = 0.0f;
    for (int ib = 0; ib < n_blocks; ib++) {
        device const block_iq4_nl & blk = w_head[(uint64_t)row * n_blocks + ib];
        const float d = blk.d;
        device const uint8_t * qs = blk.qs;
        for (int j = 0; j < 16; j++) {
            int base = ib * 32 + j;
            if (base < args.in_dim)
                sum += d * kvalues_iq4nl_f[qs[j] & 0xf] * in_h[base];
            base = ib * 32 + j + 16;
            if (base < args.in_dim)
                sum += d * kvalues_iq4nl_f[qs[j] >> 4] * in_h[base];
        }
    }

    output[idx] = sum;
}

// ---- Strided Extract ----
//
// Extract n_vectors strided slices from src into contiguous dst.
// For vector v, element j:
//   dst[v * slice_len + j] = src[v * src_stride + src_offset + j]
//
// Used for:
//   - q_pe extraction: slice_len=R=64, src_stride=head_dim=192, src_offset=qk_nope=128
//   - q_nope extraction: slice_len=P=128, src_stride=head_dim=192, src_offset=0

typedef struct {
    int32_t n_vectors;
    int32_t slice_len;
    int32_t src_stride;
    int32_t src_offset;
} l26f_kargs_strided_extract;

kernel void kernel_l26f_strided_extract(
        constant l26f_kargs_strided_extract & args [[buffer(0)]],
        device const float                   * src  [[buffer(1)]],
        device       float                   * dst  [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]])
{
    const int v = (int)gid.y;
    const int j = (int)gid.x;
    if (v >= args.n_vectors || j >= args.slice_len) return;

    dst[(uint64_t)v * args.slice_len + j] =
        src[(uint64_t)v * args.src_stride + args.src_offset + j];
}

// ---- KV Cache Append ----
//
// Append one token's kv_cmpr[C] + k_pe[R] to GPU-side KV cache at position n_cached.
// cache: [max_seq * CR] float
// After append, cache[n_cached][0:C] = kv_cmpr, cache[n_cached][C:CR] = k_pe

typedef struct {
    int32_t kv_lora_rank;
    int32_t n_rot;
    int32_t n_cached;
} l26f_kargs_kv_append;

kernel void kernel_l26f_kv_append(
        constant l26f_kargs_kv_append & args   [[buffer(0)]],
        device const float             * kv_cmpr [[buffer(1)]],
        device const float             * k_pe    [[buffer(2)]],
        device       float             * cache   [[buffer(3)]],
        uint gid [[thread_position_in_grid]])
{
    const int C = args.kv_lora_rank;
    const int R = args.n_rot;
    const int CR = C + R;
    const int pos = args.n_cached;

    if ((int)gid < C) {
        cache[(uint64_t)pos * CR + gid] = kv_cmpr[gid];
    } else if ((int)gid < CR) {
        cache[(uint64_t)pos * CR + gid] = k_pe[gid - C];
    }
}
