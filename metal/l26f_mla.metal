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

// NeoX RoPE layout used by llama.cpp for Bailing/Ling GLA Q/K:
// dim i is paired with dim i + n_dims/2.
kernel void kernel_l26f_rope_neox_batch(
        constant l26f_kargs_rope_batch & args [[buffer(0)]],
        device const float              * src [[buffer(1)]],
        device       float              * dst [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]])
{
    const int d = (int)gid.x;
    const int v = (int)gid.y;
    const int half_dim = args.n_dims / 2;
    if (d >= half_dim || v >= args.n_vectors) return;

    const int idx0 = v * args.n_dims + d;
    const int idx1 = idx0 + half_dim;
    float freq = 1.0f / pow(args.theta, (float)(2 * d) / (float)args.n_dims);
    float angle = (float)args.position * freq;
    float cos_a = cos(angle);
    float sin_a = sin(angle);
    float x0 = src[idx0];
    float x1 = src[idx1];
    dst[idx0] = x0 * cos_a - x1 * sin_a;
    dst[idx1] = x0 * sin_a + x1 * cos_a;
}

// ---- MLA Attention (tiled, online softmax) ----
//
// Tiled MLA attention replacing the naive kernel with float scores[4096] on the stack.
// Computationally identical to the naive kernel but:
//  - Uses online softmax (single pass, stable)
//  - Uses tile-based processing (no large stack array)
//  - Uses SIMD parallelism: 32 threads per head, each computing partial dot products
//
// Grid: H threadgroups of 32 threads. One head per threadgroup.
// Each simdgroup cooperatively computes the attention for one head.

typedef struct {
    int32_t n_heads;
    int32_t kv_lora_rank;
    int32_t n_rot;
    int32_t n_cached;
    float   scale;
} l26f_kargs_mla_attn;

kernel void kernel_l26f_mla_attn_v2(
        constant l26f_kargs_mla_attn & args   [[buffer(0)]],
        device const float            * q_abs  [[buffer(1)]],
        device const float            * q_pe   [[buffer(2)]],
        device const float            * k_cache [[buffer(3)]],
        device       float            * out    [[buffer(4)]],
        uint  tgpig [[threadgroup_position_in_grid]],
        uint  tiisg [[thread_index_in_simdgroup]])
{
    const int h = (int)tgpig;
    if (h >= args.n_heads) return;

    const int C = args.kv_lora_rank;
    const int R = args.n_rot;
    const int CR = C + R;
    const int n_cached = args.n_cached;
    const float scale = args.scale;

    device const float * q_abs_h = q_abs + (uint64_t)h * C;
    device const float * q_pe_h  = q_pe  + (uint64_t)h * R;

    // Online softmax state
    float o_vals[16];    // 16 elements of V per thread (512/32=16)
    for (int i = 0; i < 16; i++) o_vals[i] = 0.0f;
    float m = -1e30f;
    float l = 0.0f;

    // Each thread handles chunk_size elements of the head dim
    const int chunk = C / 32;  // 512/32 = 16
    const int base_c = (int)tiisg * chunk;
    const int base_r = (int)tiisg * (R / 32);
    const int r_chunk = R / 32;

    for (int t = 0; t < n_cached; t++) {
        device const float * k_t = k_cache + (int64_t)t * CR;
        device const float * v_t = k_t;  // V is first C elements

        float dot = 0.0f;
        for (int d = 0; d < chunk; d++) {
            dot += q_abs_h[base_c + d] * k_t[base_c + d];
        }
        for (int d = 0; d < r_chunk; d++) {
            dot += q_pe_h[base_r + d] * k_t[C + base_r + d];
        }

        // SIMD reduction for the dot product
        float s = simd_sum(dot) * scale;

        // Online softmax update
        float m_new = max(m, s);
        float exp_scale = exp(m - m_new);
        l = l * exp_scale;
        for (int i = 0; i < chunk; i++) {
            o_vals[i] *= exp_scale;
        }
        float p = exp(s - m_new);
        l += p;
        // Weighted sum: o += V * p
        for (int i = 0; i < chunk; i++) {
            o_vals[i] += v_t[base_c + i] * p;
        }
        m = m_new;
    }

    // Normalize and write output
    float inv_l = 1.0f / (l + 1e-10f);
    device float * out_h = out + (uint64_t)h * C;
    for (int i = 0; i < chunk; i++) {
        out_h[base_c + i] = o_vals[i] * inv_l;
    }
}

// Original naive kernel kept for reference/comparison
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

    float sum_exp = 0.0f;
    for (int t = 0; t < n_cached; t++) {
        scores[t] = exp(scores[t] - max_score);
        sum_exp += scores[t];
    }
    float inv_sum = 1.0f / sum_exp;
    for (int t = 0; t < n_cached; t++) {
        scores[t] *= inv_sum;
    }

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

// SIMD-group MLA IQ4_NL matvec.
//
// Same logical operation as kernel_l26f_batch_iq4_nl_matvec, but one SIMD group
// cooperates on two rows for one head. This is the hot path for MLA V
// decompression and absorption, where the naive one-thread-per-row kernel
// serializes each dot product.
kernel void kernel_l26f_batch_iq4_nl_matvec_simd(
        constant l26f_kargs_batch_iq4_nl & args  [[buffer(0)]],
        device const char               * weights [[buffer(1)]],
        device const float              * input   [[buffer(2)]],
        device       float              * output  [[buffer(3)]],
        threadgroup float               * shmem   [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]])
{
    const int NR0 = 2;
    const int h = (int)tgpig.y;
    if (h >= args.n_heads) return;

    const int n_blocks = args.in_dim / 32;
    const int first_row = (int)tgpig.x * NR0;

    device const block_iq4_nl * w = (device const block_iq4_nl *)
        (weights + (uint64_t)h * args.head_stride);
    device const float * y = input + (uint64_t)h * args.in_dim;

    const short ix = tiisg / 2;
    const short it = tiisg % 2;

    shmem[tiisg] = kvalues_iq4nl_f[tiisg % 16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[2] = {0.0f, 0.0f};

    device const float * yb = y + ix * 32 + it * 8;

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    for (int ib = ix; ib < n_blocks; ib += 16) {
        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        for (short row = 0; row < NR0; row++) {
            if (first_row + row >= args.out_rows) break;
            device const block_iq4_nl & xb =
                w[(uint64_t)(first_row + row) * n_blocks + ib];
            device const uint16_t * q4 = (device const uint16_t *)(xb.qs + 8 * it);

            float4 acc1 = {0.0f}, acc2 = {0.0f};

            aux32[0] = q4[0] | (q4[1] << 16);
            aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
            aux32[0] &= 0x0f0f0f0f;
            float4 qf1 = {shmem[q8[0]], shmem[q8[1]], shmem[q8[2]], shmem[q8[3]]};
            float4 qf2 = {shmem[q8[4]], shmem[q8[5]], shmem[q8[6]], shmem[q8[7]]};
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;

            aux32[0] = q4[2] | (q4[3] << 16);
            aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
            aux32[0] &= 0x0f0f0f0f;
            qf1 = {shmem[q8[0]], shmem[q8[1]], shmem[q8[2]], shmem[q8[3]]};
            qf2 = {shmem[q8[4]], shmem[q8[5]], shmem[q8[6]], shmem[q8[7]]};
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;

            acc1 += acc2;
            sumf[row] += (float)xb.d * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
        }

        yb += 16 * 32;
    }

    for (int row = 0; row < NR0 && first_row + row < args.out_rows; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            output[(uint64_t)h * args.out_rows + first_row + row] = sum_all;
        }
    }
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
