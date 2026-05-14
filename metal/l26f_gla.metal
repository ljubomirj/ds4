// l26f GLA Metal Kernel
// Ported from llama.cpp ggml-metal.metal:2754-2873
// Gated Linear Attention — chunkwise parallel recurrent linear attention
//
// k, v, q, g: [S, H, n_tokens]  where S=head_dim, H=n_heads
// state:      [S*S*H, n_seqs]
// output:     [S*H, n_tokens] (activations) + [S*S*H*n_seqs] (final state)

#include <metal_stdlib>
using namespace metal;

typedef struct {
    int32_t ne00, ne01, ne02, ne03; // k dims: [S, H, T]
    int32_t nb02;                    // k stride between tokens (in floats)
    int32_t ne10, ne11, ne12, ne13; // v dims: [S, H, T]
    int32_t nb12;                    // v stride between tokens
    int32_t ne20, ne21, ne22, ne23; // q dims: [S, H, T]
    int32_t nb22;                    // q stride between tokens
    int32_t ne30, ne31, ne32, ne33; // g dims: [S, H, T]
    int32_t nb32;                    // g stride between tokens
    int32_t ne40, ne41, ne42, ne43; // state dims: [S*S, H, n_seqs]
    int32_t ne0,  ne1,  ne2,  ne3;  // output dims: [S*H, T + S*n_seqs]
    float   scale;
} l26f_kargs_gla;

// grid = (S/NSG, S/4, H*n_seqs), threads = (32, NSG, 1)
template<short NSG>
kernel void kernel_gla_impl(
        constant l26f_kargs_gla & args,
        device const float * k [[buffer(1)]],
        device const float * v [[buffer(2)]],
        device const float * q [[buffer(3)]],
        device const float * g [[buffer(4)]],
        device const float * s [[buffer(5)]],
        device       float * dst [[buffer(6)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]])
{
    const short S       = (short)args.ne00;
    const short H       = (short)args.ne01;
    const short n_seqs  = (short)args.ne41;
    const short T_total = (short)args.ne02;
    const short T_per_seq = T_total / n_seqs;

    const float scale = args.scale;

    const short tx = (short)tpitg.x;
    const short ty = (short)tpitg.y;

    const short n = (short)tgpig.z;
    const short h_idx = n % H;
    const short seq_idx = n / H;

    const short j = (short)tgpig.y * 4 + ty;
    const short i_start = (short)tgpig.x * NSG;

    const short t_start = seq_idx * T_per_seq;
    const short t_end = t_start + T_per_seq;

    // state for this (head, seq): [S, S]
    device const float * s_in = s + (seq_idx * H + h_idx) * S * S;

    float state_vals[NSG];
    FOR_UNROLL (short i = 0; i < NSG; i++) {
        const short ii = i_start + tx * NSG + i;
        state_vals[i] = (ii < S && j < S) ? s_in[ii * S + j] : 0.0f;
    }

    // output activations: [S*H, T_total]
    device float * dst_out = dst + t_start * S * H;

    for (short t = t_start; t < t_end; t++) {
        device const float * k_t = k + t * args.nb02 + h_idx * S;
        device const float * v_t = v + t * args.nb12 + h_idx * S;
        device const float * q_t = q + t * args.nb22 + h_idx * S;
        device const float * g_t = g + t * args.nb32 + h_idx * S;

        float v_val = (j < S) ? v_t[j] : 0.0f;
        float out_j = 0.0f;

        FOR_UNROLL (short i = 0; i < NSG; i++) {
            const short ii = i_start + tx * NSG + i;
            float k_val = (ii < S) ? k_t[ii] : 0.0f;
            float q_val = (ii < S) ? q_t[ii] * scale : 0.0f;
            float g_val = (ii < S) ? g_t[ii] : 0.0f;

            state_vals[i] = state_vals[i] * g_val + k_val * v_val;
            out_j += state_vals[i] * q_val;
        }

        out_j = simd_sum(out_j);

        if (tx == 0 && j < S) {
            dst_out[(t - t_start) * S * H + h_idx * S + j] = out_j;
        }
    }

    // store final state at end of output
    device float * dst_state = dst + (int64_t)T_total * S * H + (seq_idx * H + h_idx) * S * S;

    FOR_UNROLL (short i = 0; i < NSG; i++) {
        const short ii = i_start + tx * NSG + i;
        if (ii < S && j < S) {
            dst_state[ii * S + j] = state_vals[i];
        }
    }
}

typedef decltype(kernel_gla_impl<4>) kernel_gla_t;

template [[host_name("kernel_gla_1")]] kernel kernel_gla_t kernel_gla_impl<1>;
template [[host_name("kernel_gla_2")]] kernel kernel_gla_t kernel_gla_impl<2>;
template [[host_name("kernel_gla_4")]] kernel kernel_gla_t kernel_gla_impl<4>;
template [[host_name("kernel_gla_8")]] kernel kernel_gla_t kernel_gla_impl<8>;

// ---- GLA with state snapshots (for MTP rollback) ----
// Ported from r2 kernel_gated_linear_attn_impl<NSG, true>
// Saves state after each token (except last) for speculative decoding rollback.

template<short NSG>
kernel void kernel_gla_ki_impl(
        constant l26f_kargs_gla & args,
        device const float * k [[buffer(1)]],
        device const float * v [[buffer(2)]],
        device const float * q [[buffer(3)]],
        device const float * g [[buffer(4)]],
        device const float * s [[buffer(5)]],
        device       float * dst [[buffer(6)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]])
{
    const short S       = (short)args.ne00;
    const short H       = (short)args.ne01;
    const short n_seqs  = (short)args.ne41;
    const short T_total = (short)args.ne02;
    const short T_per_seq = T_total / n_seqs;

    const float scale = args.scale;

    const short tx = (short)tpitg.x;
    const short ty = (short)tpitg.y;

    const short n = (short)tgpig.z;
    const short h_idx = n % H;
    const short seq_idx = n / H;

    const short j = (short)tgpig.y * 4 + ty;
    const short i_start = (short)tgpig.x * NSG;

    const short t_start = seq_idx * T_per_seq;
    const short t_end = t_start + T_per_seq;

    device const float * s_in = s + (seq_idx * H + h_idx) * S * S;

    float state_vals[NSG];
    FOR_UNROLL (short i = 0; i < NSG; i++) {
        const short ii = i_start + tx * NSG + i;
        state_vals[i] = (ii < S && j < S) ? s_in[ii * S + j] : 0.0f;
    }

    device float * dst_out = dst + t_start * S * H;

    const int64_t state_base = (int64_t)T_total * S * H;

    for (short t = t_start; t < t_end; t++) {
        device const float * k_t = k + t * args.nb02 + h_idx * S;
        device const float * v_t = v + t * args.nb12 + h_idx * S;
        device const float * q_t = q + t * args.nb22 + h_idx * S;
        device const float * g_t = g + t * args.nb32 + h_idx * S;

        float v_val = (j < S) ? v_t[j] : 0.0f;
        float out_j = 0.0f;

        FOR_UNROLL (short i = 0; i < NSG; i++) {
            const short ii = i_start + tx * NSG + i;
            float k_val = (ii < S) ? k_t[ii] : 0.0f;
            float q_val = (ii < S) ? q_t[ii] * scale : 0.0f;
            float g_val = (ii < S) ? g_t[ii] : 0.0f;

            state_vals[i] = state_vals[i] * g_val + k_val * v_val;
            out_j += state_vals[i] * q_val;
        }

        out_j = simd_sum(out_j);

        if (tx == 0 && j < S) {
            dst_out[(t - t_start) * S * H + h_idx * S + j] = out_j;
        }

        // Store intermediate state for rollback (save after each token except last)
        if (t < t_end - 1) {
            const short t_in_seq = t - t_start;
            const int64_t state_elems_all = (int64_t)S * S * H * n_seqs;
            device float * snap = dst + state_base + state_elems_all
                + ((int64_t)t_in_seq * n_seqs + seq_idx) * H * S * S + h_idx * S * S;
            FOR_UNROLL (short i = 0; i < NSG; i++) {
                const short ii = i_start + tx * NSG + i;
                if (ii < S && j < S) {
                    snap[ii * S + j] = state_vals[i];
                }
            }
        }
    }

    // Store final state
    device float * dst_state = dst + state_base + (seq_idx * H + h_idx) * S * S;

    FOR_UNROLL (short i = 0; i < NSG; i++) {
        const short ii = i_start + tx * NSG + i;
        if (ii < S && j < S) {
            dst_state[ii * S + j] = state_vals[i];
        }
    }
}

typedef decltype(kernel_gla_ki_impl<4>) kernel_gla_ki_t;

template [[host_name("kernel_gla_ki_1")]] kernel kernel_gla_ki_t kernel_gla_ki_impl<1>;
template [[host_name("kernel_gla_ki_2")]] kernel kernel_gla_ki_t kernel_gla_ki_impl<2>;
template [[host_name("kernel_gla_ki_4")]] kernel kernel_gla_ki_t kernel_gla_ki_impl<4>;
template [[host_name("kernel_gla_ki_8")]] kernel kernel_gla_ki_t kernel_gla_ki_impl<8>;

typedef struct {
    int32_t head_dim;
    int32_t n_heads;
    int32_t position;
    float   theta;
    float   eps;
} l26f_kargs_gla_qk_norm_rope;

kernel void kernel_l26f_gla_qk_norm_rope(
        constant l26f_kargs_gla_qk_norm_rope & args [[buffer(0)]],
        device const float * q                 [[buffer(1)]],
        device const float * k                 [[buffer(2)]],
        device const float * q_weight          [[buffer(3)]],
        device const float * k_weight          [[buffer(4)]],
        device       float * q_rope            [[buffer(5)]],
        device       float * k_rope            [[buffer(6)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tpitg [[thread_position_in_threadgroup]])
{
    const int S = args.head_dim;
    const int h = (int)tgpig.x;
    const int which = (int)tgpig.y;
    const uint tid = tpitg.x;
    if (h >= args.n_heads || which > 1) return;

    device const float * src = which == 0 ? q : k;
    device const float * weight = which == 0 ? q_weight : k_weight;
    device float * dst = which == 0 ? q_rope : k_rope;
    const int base = h * S;

    threadgroup float sumsq[128];
    float local = 0.0f;
    for (int d = (int)tid; d < S; d += 128) {
        const float x = src[base + d];
        local += x * x;
    }
    sumsq[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride) sumsq[tid] += sumsq[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float scale = rsqrt(sumsq[0] / (float)S + args.eps);
    const int half_dim = S / 2;
    if (tid < (uint)half_dim) {
        const int d0 = (int)tid;
        const int d1 = d0 + half_dim;
        const float freq = 1.0f / pow(args.theta, (float)(2 * d0) / (float)S);
        const float angle = (float)args.position * freq;
        const float cos_a = cos(angle);
        const float sin_a = sin(angle);
        const float x0 = src[base + d0] * scale * weight[d0];
        const float x1 = src[base + d1] * scale * weight[d1];
        dst[base + d0] = x0 * cos_a - x1 * sin_a;
        dst[base + d1] = x0 * sin_a + x1 * cos_a;
    }
}

typedef struct {
    int32_t n_embd;
    int32_t n_groups;
    float   eps;
} l26f_kargs_gla_epilogue;

kernel void kernel_l26f_gla_epilogue(
        constant l26f_kargs_gla_epilogue & args [[buffer(0)]],
        device const float * gla_act            [[buffer(1)]],
        device const float * gate               [[buffer(2)]],
        device const float * weight             [[buffer(3)]],
        device       float * out                [[buffer(4)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tpitg [[thread_position_in_threadgroup]])
{
    const int group = (int)tgpig.x;
    const uint tid = tpitg.x;
    const int group_dim = args.n_embd / args.n_groups;
    if (group >= args.n_groups) return;

    const int base = group * group_dim;
    threadgroup float sumsq[256];
    float local = 0.0f;
    for (int d = (int)tid; d < group_dim; d += 256) {
        const float x = gla_act[base + d];
        local += x * x;
    }
    sumsq[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) sumsq[tid] += sumsq[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float rms = rsqrt(sumsq[0] / (float)group_dim + args.eps);
    for (int d = (int)tid; d < group_dim; d += 256) {
        const int idx = base + d;
        const float g = gate[idx];
        const float sig = 1.0f / (1.0f + exp(-g));
        out[idx] = gla_act[idx] * rms * weight[idx] * sig;
    }
}
