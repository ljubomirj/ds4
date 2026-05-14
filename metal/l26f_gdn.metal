// l26f GDN (Gated Delta Net) Metal Kernel
// Ported from llama.cpp ggml-metal.metal:2537-2752 (Ling-2.6-flash-r2 branch)
// Gated Delta Net — recurrent attention with delta rule
//
// q, k, v: [S_v, H, n_tokens]
// g:       [G, H*ne22*n_batches] (gating, G=1 for scalar, G=S_v for KDA)
// b:       [H*ne22*n_batches]     (beta)
// state:   [S_v*S_v*H, n_batches]
// output:  [S_v, H*ne22*n_batches] (activations) + [S_v*S_v*H*n_batches] (final state)

#include <metal_stdlib>
using namespace metal;

#define N_SIMDWIDTH 32

#ifndef FOR_UNROLL
#define FOR_UNROLL(expr) expr
#endif

typedef struct {
    int32_t  ne00, ne01, ne02, ne03;
    uint64_t nb00, nb01, nb02, nb03;
    int32_t  ne10, ne11, ne12, ne13;
    uint64_t nb10, nb11, nb12, nb13;
    int32_t  ne20, ne21, ne22, ne23;
    uint64_t nb20, nb21, nb22, nb23;
    int32_t  ne30; // g groups: 1=scalar, S_v=KDA
    int32_t  ns02, ns12, ns22;
    int32_t  ne0,  ne1,  ne2,  ne3;
    uint64_t nb0,  nb1,  nb2,  nb3;
} l26f_kargs_gdn;

// grid = (S_v/NSG, H*n_batches), threads = (32, NSG, 1)
template<short NSG>
kernel void kernel_gdn_impl(
        constant l26f_kargs_gdn & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * g,
        device const char * b,
        device const char * s,
        device       char * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]])  {
    const int S_v = args.ne20;
    const int G   = args.ne30;

    const uint tx = tpitg.x;
    const uint ty = tpitg.y;

    const uint i23 = tgpig.z;
    const uint i21 = tgpig.y;
    const uint i20 = tgpig.x * NSG + ty;

    const uint i01 = i21 % args.ne01;
    const uint i11 = i21 % args.ne11;

    const float scale = 1.0f / sqrt((float)S_v);

    device const float * s_ptr = (device const float *) (s) + (i23*args.ne21 + i21)*S_v*S_v + i20*S_v;

    float ls[NSG];

    FOR_UNROLL (short j = 0; j < NSG; j++) {
        const short is = tx*NSG + j;
        ls[j] = s_ptr[is];
    }

    device float * dst_attn = (device float *) (dst) + (i23*args.ne22*args.ne21 + i21)*S_v + i20;

    device const float * q_ptr = (device const float *) (q + i23*args.nb03 + i01*args.nb01);
    device const float * k_ptr = (device const float *) (k + i23*args.nb13 + i11*args.nb11);
    device const float * v_ptr = (device const float *) (v + i23*args.nb23 + i21*args.nb21);

    device const float * b_ptr = (device const float *) (b) + (i23*args.ne22*args.ne21 + i21);
    device const float * g_ptr = (device const float *) (g) + (i23*args.ne22*args.ne21 + i21)*G;

    for (short t = 0; t < args.ne22; t++) {
        float s_k = 0.0f;

        if (G == 1) {
            const float g_exp = exp(g_ptr[0]);

            FOR_UNROLL (short j = 0; j < NSG; j++) {
                const short is = tx*NSG + j;
                ls[j] *= g_exp;

                s_k += ls[j]*k_ptr[is];
            }
        } else {
            FOR_UNROLL (short j = 0; j < NSG; j++) {
                const short is = tx*NSG + j;
                ls[j] *= exp(g_ptr[is]);

                s_k += ls[j]*k_ptr[is];
            }
        }

        s_k = simd_sum(s_k);

        const float d = (v_ptr[i20] - s_k)*b_ptr[0];

        float y = 0.0f;

        FOR_UNROLL (short j = 0; j < NSG; j++) {
            const short is = tx*NSG + j;
            ls[j] += k_ptr[is]*d;

            y += ls[j]*q_ptr[is];
        }

        y = simd_sum(y);

        if (tx == 0) {
            dst_attn[t*args.ne21*S_v] = y*scale;
        }

        q_ptr += args.ns02;
        k_ptr += args.ns12;
        v_ptr += args.ns22;

        b_ptr += args.ne21;
        g_ptr += args.ne21*G;
    }

    device float * dst_state = (device float *) (dst) + args.ne23*args.ne22*args.ne21*S_v + (i23*args.ne21 + i21)*S_v*S_v + i20*S_v;

    FOR_UNROLL (short j = 0; j < NSG; j++) {
        const short is = tx*NSG + j;
        dst_state[is] = ls[j];
    }
}

typedef decltype(kernel_gdn_impl<4>) kernel_gdn_t;

template [[host_name("kernel_gdn_1")]] kernel kernel_gdn_t kernel_gdn_impl<1>;
template [[host_name("kernel_gdn_2")]] kernel kernel_gdn_t kernel_gdn_impl<2>;
template [[host_name("kernel_gdn_4")]] kernel kernel_gdn_t kernel_gdn_impl<4>;
