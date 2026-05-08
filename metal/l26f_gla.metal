// l26f GLA Metal Kernel
// Ported from llama.cpp ggml-metal.metal:2754-2873
// Gated Linear Attention — chunkwise parallel recurrent linear attention
//
// k, v, q, g: [S, H, n_tokens]  where S=head_dim, H=n_heads
// state:      [S*S*H, n_seqs]
// output:     [S*H, n_tokens] (activations) + [S*S*H*n_seqs] (final state)

#include <metal_stdlib>
using namespace metal;

#define FOR_UNROLL(n) for (short _i = 0; _i < (n); _i++)

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
