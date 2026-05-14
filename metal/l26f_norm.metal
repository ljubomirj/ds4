// l26f Group Norm Metal Kernel
// Ported from llama.cpp ggml-metal.metal:3239-3318 (Ling-2.6-flash-r2 branch)
// Group normalization — needed for GLA output normalization

#include <metal_stdlib>
using namespace metal;

#define N_SIMDWIDTH 32

typedef struct {
    int32_t ne00;
    int32_t ne01;
    int32_t ne02;
    int32_t ngrp;
    float   eps;
} l26f_kargs_group_norm;

kernel void kernel_group_norm_f32(
        constant l26f_kargs_group_norm & args,
        device const float * src0,
        device       float * dst,
        threadgroup float  * buf [[threadgroup(0)]],
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    const int64_t ne = args.ne00*args.ne01*args.ne02;
    const int64_t gs = args.ne00*args.ne01*((args.ne02 + args.ngrp - 1) / args.ngrp);

    int start = (int)(tgpig * gs);
    int end   = start + (int)gs;

    start += (int)tpitg;

    if (end >= ne) {
        end = (int)ne;
    }

    float tmp = 0.0f;

    for (int j = start; j < end; j += (int)ntg) {
        tmp += src0[j];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    tmp = simd_sum(tmp);
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = tmp;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        tmp = buf[tiisg];
        tmp = simd_sum(tmp);
    }

    const float mean = tmp / gs;
    tmp = 0.0f;

    for (int j = start; j < end; j += (int)ntg) {
        float xi = src0[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = simd_sum(tmp);
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = tmp;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        tmp = buf[tiisg];
        tmp = simd_sum(tmp);
    }

    const float variance = tmp / gs;
    const float scale = 1.0f/sqrt(variance + args.eps);
    for (int j = start; j < end; j += (int)ntg) {
        dst[j] *= scale;
    }
}
