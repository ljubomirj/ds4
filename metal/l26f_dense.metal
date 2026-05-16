#ifndef QK_K
#define QK_K 256
#endif

// L26F metal: IQ4_NL, Q5_K, Q6_K dequantize helpers and matvec kernels
//
// IQ4_NL block layout (18 bytes, 32 elements):
//   offset 0: half d     (2 bytes) — scale factor
//   offset 2: uint8_t qs[16] (16 bytes) — 32 x 4-bit packed values, 2 per byte
// Total: 18 bytes
//
// IQ4_NL kernel ported from ggml-metal.metal kernel_mul_mv_iq4_nl_f32_impl.

// ---- Dequantize helpers ----

static void dequantize_iq4_nl(device const block_iq4_nl * xb, short il, thread float4x4 & reg) {
    device const uint16_t * q4 = (device const uint16_t *)xb->qs;
    const float d = xb->d;
    uint32_t aux32;
    thread const uint8_t * q8 = (thread const uint8_t *)&aux32;
    for (int i = 0; i < 4; ++i) {
        aux32 = ((q4[2*i] | (q4[2*i+1] << 16)) >> 4*il) & 0x0f0f0f0f;
        reg[i][0] = d * kvalues_iq4nl_f[q8[0]];
        reg[i][1] = d * kvalues_iq4nl_f[q8[1]];
        reg[i][2] = d * kvalues_iq4nl_f[q8[2]];
        reg[i][3] = d * kvalues_iq4nl_f[q8[3]];
    }
}

static void dequantize_iq4_nl_t4(device const block_iq4_nl * xb, short il, thread float4 & reg) {
    device const uint16_t * q4 = (device const uint16_t *)xb->qs;
    const float d = xb->d;
    uint32_t aux32;
    thread const uint8_t * q8 = (thread const uint8_t *)&aux32;
    aux32 = ((q4[2*(il%4)] | (q4[2*(il%4)+1] << 16)) >> 4*(il/4)) & 0x0f0f0f0f;
    reg[0] = d * kvalues_iq4nl_f[q8[0]];
    reg[1] = d * kvalues_iq4nl_f[q8[1]];
    reg[2] = d * kvalues_iq4nl_f[q8[2]];
    reg[3] = d * kvalues_iq4nl_f[q8[3]];
}

// Q5_K: 5.4 bpw, 256 elements per block
static void dequantize_q5_K(device const block_q5_K *xb, short il, thread float4x4 & reg) {
    device const uint8_t * q  = xb->qs;
    device const uint8_t * qh = xb->qh;
    short is = (il/4) * 2;
    q  = q + 32 * (il/4) + 16 * (il&1);
    qh = qh + 16 * (il&1);
    uint8_t ul = 1 << (il/2);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d = il < 2 ? xb->d : xb->d / 16.f;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];
    const ushort mask  = il<2 ? 0x0F : 0xF0;
    const float qh_val = il<2 ? 16.f : 256.f;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * ((q[i] & mask) + (qh[i] & ul ? qh_val : 0)) - ml;
    }
}

// Q6_K: 6.6 bpw, 256 elements per block
static void dequantize_q6_K(device const block_q6_K *xb, short il, thread float4x4 & reg) {
    const half d_all = xb->d;
    device const uint16_t * ql = (device const uint16_t *)xb->ql;
    device const uint16_t * qh = (device const uint16_t *)xb->qh;
    device const int8_t * scales = (device const int8_t *)xb->scales;
    ql = ql + 32*(il/8) + 16*((il/2)&1) + 8*(il&1);
    qh = qh + 16*(il/8) + 8*(il&1);
    float sc = scales[(il%2) + 2 * ((il/2))];
    il = (il/2) & 3;
    const uint32_t kmask1 = il>1 ? (il>2 ? 0xC0C0C0C0 : 0x30303030) : (il>0 ? 0x0C0C0C0C : 0x03030303);
    const uint32_t kmask2 = il>1 ? 0xF0F0F0F0                       : 0x0F0F0F0F;
    const float ml = d_all * sc * 32.f;
    const float dl0 = d_all * sc;
    const float dl1 = dl0 / 256.f;
    const float dl2 = dl0 / (256.f * 256.f);
    const float dl3 = dl0 / (256.f * 256.f * 256.f);
    const uint8_t shr_h = il>2 ? 2 : 0;
    const uint8_t shl_h = il>1 ? 0 : (il>0 ? 2 : 4);
    const uint8_t shr_l = il>1 ? 4 : 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t  low = (ql[2*i] | (uint32_t)(ql[2*i+1] << 16)) & kmask2;
        const uint32_t high = (qh[2*i] | (uint32_t)(qh[2*i+1] << 16)) & kmask1;
        const uint32_t q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg[i][0] = dl0 *  ((half)(q & 0xFF))       - ml;
        reg[i][1] = dl1 * ((float)(q & 0xFF00))     - ml;
        reg[i][2] = dl2 * ((float)(q & 0xFF0000))   - ml;
        reg[i][3] = dl3 * ((float)(q & 0xFF000000)) - ml;
    }
}

// ---- IQ4_NL single-token decode matvec ----
// Ported from ggml-metal.metal kernel_mul_mv_iq4_nl_f32_impl.
// Uses shared memory for the kvalues lookup table, processes NR0=2 rows per threadgroup.

struct l26f_args_mul_mv_iq4_nl {
    int32_t ne00, ne01, ne02;
    int32_t nb00, nb01, nb02;
    int32_t ne10, ne11, ne12;
    int32_t nb10, nb11;
    int32_t ne0,  ne1;
    int32_t nsg;
    int32_t add_residual;
    int16_t r2, r3;
};

kernel void kernel_mul_mv_iq4_nl_f32(
    constant l26f_args_mul_mv_iq4_nl & args [[buffer(0)]],
    device const char               * src0 [[buffer(1)]],
    device const float              * src1 [[buffer(2)]],
    device       float              * dst  [[buffer(3)]],
    device const float              * residual [[buffer(4)]],
    threadgroup float               * shmem [[threadgroup(0)]],
    uint3  tgpig[[threadgroup_position_in_grid]],
    ushort tiisg[[thread_index_in_simdgroup]],
    ushort sgitg[[simdgroup_index_in_threadgroup]])
{
    const int NR0 = 2;
    const int nb  = args.ne00 / 32;
    const int nbl = args.nb01 / args.nb00;

    const int first_row = ((int)tgpig.x * args.nsg + (int)sgitg) * NR0;

    device const block_iq4_nl * x = (device const block_iq4_nl *)(src0 + (uint64_t)first_row * args.nb01);
    device const float        * y = src1;

    const short ix = tiisg / 2;
    const short it = tiisg % 2;

    shmem[tiisg] = kvalues_iq4nl_f[tiisg % 16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[2] = {0.f};

    device const float * yb = y + ix * 32 + it * 8;

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    float4 qf1, qf2;

    for (int ib = ix; ib < nb; ib += 16) {
        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        for (short row = 0; row < NR0; row++) {
            if (first_row + row >= args.ne01) break;
            device const block_iq4_nl & xb = x[(uint64_t)row * nbl + ib];
            device const uint16_t * q4 = (device const uint16_t *)(xb.qs + 8*it);

            float4 acc1 = {0.f}, acc2 = {0.f};

            aux32[0] = q4[0] | (q4[1] << 16);
            aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
            aux32[0] &= 0x0f0f0f0f;
            qf1 = {shmem[q8[0]], shmem[q8[1]], shmem[q8[2]], shmem[q8[3]]};
            qf2 = {shmem[q8[4]], shmem[q8[5]], shmem[q8[6]], shmem[q8[7]]};
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

    for (int row = 0; row < NR0 && first_row + row < args.ne01; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            const int out_row = first_row + row;
            dst[out_row] = sum_all + (args.add_residual ? residual[out_row] : 0.0f);
        }
    }
}

typedef struct {
    int32_t in_dim;
    int32_t qkv_rows;
    int32_t gate_rows;
    int32_t nsg;
    uint64_t qkv_row_bytes;
    uint64_t gate_row_bytes;
} l26f_args_gla_qkv_gate_iq4nl;

kernel void kernel_l26f_gla_qkv_gate_iq4_nl_f32(
    constant l26f_args_gla_qkv_gate_iq4nl & args [[buffer(0)]],
    device const char               * qkv_weights  [[buffer(1)]],
    device const char               * gate_weights [[buffer(2)]],
    device const float              * input        [[buffer(3)]],
    device       float              * qkv_out      [[buffer(4)]],
    device       float              * gate_out     [[buffer(5)]],
    threadgroup float               * shmem        [[threadgroup(0)]],
    uint3  tgpig[[threadgroup_position_in_grid]],
    ushort tiisg[[thread_index_in_simdgroup]],
    ushort sgitg[[simdgroup_index_in_threadgroup]])
{
    const int NR0 = 2;
    const int total_rows = args.qkv_rows + args.gate_rows;
    const int n_blocks = args.in_dim / 32;
    const int first_row = ((int)tgpig.x * args.nsg + (int)sgitg) * NR0;

    const short ix = tiisg / 2;
    const short it = tiisg % 2;

    shmem[tiisg] = kvalues_iq4nl_f[tiisg % 16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[2] = {0.f};

    device const float * yb = input + ix * 32 + it * 8;

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    float4 qf1, qf2;

    for (int ib = ix; ib < n_blocks; ib += 16) {
        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        for (short row = 0; row < NR0; row++) {
            const int out_row = first_row + row;
            if (out_row >= total_rows) break;

            const bool is_qkv = out_row < args.qkv_rows;
            const int local_row = is_qkv ? out_row : out_row - args.qkv_rows;
            device const block_iq4_nl * w = is_qkv ?
                (device const block_iq4_nl *)(qkv_weights + (uint64_t)local_row * args.qkv_row_bytes) :
                (device const block_iq4_nl *)(gate_weights + (uint64_t)local_row * args.gate_row_bytes);
            device const block_iq4_nl & xb = w[ib];
            device const uint16_t * q4 = (device const uint16_t *)(xb.qs + 8 * it);

            float4 acc1 = {0.f}, acc2 = {0.f};

            aux32[0] = q4[0] | (q4[1] << 16);
            aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
            aux32[0] &= 0x0f0f0f0f;
            qf1 = {shmem[q8[0]], shmem[q8[1]], shmem[q8[2]], shmem[q8[3]]};
            qf2 = {shmem[q8[4]], shmem[q8[5]], shmem[q8[6]], shmem[q8[7]]};
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

    for (int row = 0; row < NR0 && first_row + row < total_rows; ++row) {
        const int out_row = first_row + row;
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            if (out_row < args.qkv_rows) {
                qkv_out[out_row] = sum_all;
            } else {
                gate_out[out_row - args.qkv_rows] = sum_all;
            }
        }
    }
}

typedef struct {
    int32_t in_dim;
    int32_t qkv_rows;
    int32_t gate_rows;
    uint64_t qkv_row_bytes;
    uint64_t gate_row_bytes;
} l26f_args_gla_qkv_gate_q5k;

kernel void kernel_l26f_gla_qkv_gate_q5_k_f32(
    constant l26f_args_gla_qkv_gate_q5k & args [[buffer(0)]],
    device const char               * qkv_weights  [[buffer(1)]],
    device const char               * gate_weights [[buffer(2)]],
    device const float              * input        [[buffer(3)]],
    device       float              * qkv_out      [[buffer(4)]],
    device       float              * gate_out     [[buffer(5)]],
    uint3  tgpig[[threadgroup_position_in_grid]],
    ushort tiisg[[thread_index_in_simdgroup]],
    ushort sgitg[[simdgroup_index_in_threadgroup]])
{
    const short NSG = 2;
    const int total_rows = args.qkv_rows + args.gate_rows;
    const int out_row = (int)tgpig.x * NSG + (int)sgitg;
    if (out_row >= total_rows) return;

    const bool is_qkv = out_row < args.qkv_rows;
    const int row_in_tensor = is_qkv ? out_row : out_row - args.qkv_rows;
    device const char * weights = is_qkv ?
        qkv_weights + (uint64_t)row_in_tensor * args.qkv_row_bytes :
        gate_weights + (uint64_t)row_in_tensor * args.gate_row_bytes;
    device const block_q5_K * x = (device const block_q5_K *)weights;
    device const float * yy = input;

    const int nb = args.in_dim / QK_K;

    float sumf = 0.f;
    float yl[16], yh[16];

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short tid = tiisg / 4;
    const short ix  = tiisg % 4;
    const short iq  = tid / 4;
    const short ir  = tid % 4;

    const short l0 = 8 * ir;
    const short q_offset = 32 * iq + l0;
    const short y_offset = 64 * iq + l0;

    const uint8_t hm1 = 1u << (2 * iq);
    const uint8_t hm2 = hm1 << 1;
    const uint8_t hm3 = hm1 << 4;
    const uint8_t hm4 = hm2 << 4;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    device const float * y1 = yy + ix * QK_K + y_offset;

    for (int i = ix; i < nb; i += 4) {
        device const uint8_t * q1 = x[i].qs + q_offset;
        device const uint8_t * qh = x[i].qh + l0;
        device const half * dh = &x[i].d;
        device const uint16_t * a = (device const uint16_t *)x[i].scales + iq;

        device const float * y2 = y1 + 128;
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short l = 0; l < 8; ++l) {
            yl[l+0] = y1[l +  0]; sumy[0] += yl[l+0];
            yl[l+8] = y1[l + 32]; sumy[1] += yl[l+8];
            yh[l+0] = y2[l +  0]; sumy[2] += yh[l+0];
            yh[l+8] = y2[l + 32]; sumy[3] += yh[l+8];
        }

        device const uint8_t * q2 = q1 + 64;

        sc16[0] = a[0] & kmask1;
        sc16[1] = a[2] & kmask1;
        sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
        sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);

        float4 acc1 = {0.f};
        float4 acc2 = {0.f};
        FOR_UNROLL (short l = 0; l < 8; ++l) {
            uint8_t h = qh[l];
            acc1[0] += yl[l+0] * (q1[l] & 0x0F);
            acc1[1] += yl[l+8] * (q1[l] & 0xF0);
            acc1[2] += yh[l+0] * (q2[l] & 0x0F);
            acc1[3] += yh[l+8] * (q2[l] & 0xF0);
            acc2[0] += h & hm1 ? yl[l+0] : 0.f;
            acc2[1] += h & hm2 ? yl[l+8] : 0.f;
            acc2[2] += h & hm3 ? yh[l+0] : 0.f;
            acc2[3] += h & hm4 ? yh[l+8] : 0.f;
        }

        sumf += dh[0] * (sc8[0] * (acc1[0]      + 16.f*acc2[0]) +
                         sc8[1] * (acc1[1]/16.f + 16.f*acc2[1]) +
                         sc8[4] * (acc1[2]      + 16.f*acc2[2]) +
                         sc8[5] * (acc1[3]/16.f + 16.f*acc2[3])) -
                dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

        y1 += 4 * QK_K;
    }

    const float total = simd_sum(sumf);
    if (tiisg == 0) {
        if (is_qkv) {
            qkv_out[row_in_tensor] = total;
        } else {
            gate_out[row_in_tensor] = total;
        }
    }
}

kernel void kernel_mul_mv_iq4_nl_f32_nr4(
    constant l26f_args_mul_mv_iq4_nl & args [[buffer(0)]],
    device const char               * src0 [[buffer(1)]],
    device const float              * src1 [[buffer(2)]],
    device       float              * dst  [[buffer(3)]],
    device const float              * residual [[buffer(4)]],
    threadgroup float               * shmem [[threadgroup(0)]],
    uint3  tgpig[[threadgroup_position_in_grid]],
    ushort tiisg[[thread_index_in_simdgroup]],
    ushort sgitg[[simdgroup_index_in_threadgroup]])
{
    const int NR0 = 4;
    const int nb  = args.ne00 / 32;
    const int nbl = args.nb01 / args.nb00;

    const int first_row = ((int)tgpig.x * args.nsg + (int)sgitg) * NR0;

    device const block_iq4_nl * x = (device const block_iq4_nl *)(src0 + (uint64_t)first_row * args.nb01);
    device const float        * y = src1;

    const short ix = tiisg / 2;
    const short it = tiisg % 2;

    shmem[tiisg] = kvalues_iq4nl_f[tiisg % 16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[4];
    sumf[0] = 0.f;
    sumf[1] = 0.f;
    sumf[2] = 0.f;
    sumf[3] = 0.f;

    device const float * yb = y + ix * 32 + it * 8;

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    float4 qf1, qf2;

    for (int ib = ix; ib < nb; ib += 16) {
        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        for (short row = 0; row < NR0; row++) {
            if (first_row + row >= args.ne01) break;
            device const block_iq4_nl & xb = x[(uint64_t)row * nbl + ib];
            device const uint16_t * q4 = (device const uint16_t *)(xb.qs + 8*it);

            float4 acc1 = {0.f}, acc2 = {0.f};

            aux32[0] = q4[0] | (q4[1] << 16);
            aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
            aux32[0] &= 0x0f0f0f0f;
            qf1 = {shmem[q8[0]], shmem[q8[1]], shmem[q8[2]], shmem[q8[3]]};
            qf2 = {shmem[q8[4]], shmem[q8[5]], shmem[q8[6]], shmem[q8[7]]};
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

    for (int row = 0; row < NR0 && first_row + row < args.ne01; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            const int out_row = first_row + row;
            dst[out_row] = sum_all + (args.add_residual ? residual[out_row] : 0.0f);
        }
    }
}

// ---- IQ4_NL small-batch matvec instantiations (reuse ds4 template) ----

typedef decltype(kernel_mul_mv_ext_q4_f32_disp<2, block_iq4_nl, 32, dequantize_iq4_nl_t4>) mul_mv_ext_iq4_nl_t;

template [[host_name("kernel_mul_mv_ext_iq4_nl_f32_r1_2")]] kernel mul_mv_ext_iq4_nl_t kernel_mul_mv_ext_q4_f32_disp<2, block_iq4_nl, 32, dequantize_iq4_nl_t4>;
template [[host_name("kernel_mul_mv_ext_iq4_nl_f32_r1_3")]] kernel mul_mv_ext_iq4_nl_t kernel_mul_mv_ext_q4_f32_disp<3, block_iq4_nl, 32, dequantize_iq4_nl_t4>;
template [[host_name("kernel_mul_mv_ext_iq4_nl_f32_r1_4")]] kernel mul_mv_ext_iq4_nl_t kernel_mul_mv_ext_q4_f32_disp<4, block_iq4_nl, 32, dequantize_iq4_nl_t4>;
template [[host_name("kernel_mul_mv_ext_iq4_nl_f32_r1_5")]] kernel mul_mv_ext_iq4_nl_t kernel_mul_mv_ext_q4_f32_disp<5, block_iq4_nl, 32, dequantize_iq4_nl_t4>;

// ---- Q5_K SIMD-optimized matvec (ported from ggml-metal) ----
// Uses 32-thread SIMD groups, NR0=1 rows per SG, N_SG=2 SGs per threadgroup.
// Direct bit extraction instead of full dequantize — much faster than naive version.

kernel void kernel_mul_mv_q5_K_f32(
    constant ds4_metal_args_mul_mv & args [[buffer(0)]],
    device const char   * src0 [[buffer(1)]],
    device const char   * src1 [[buffer(2)]],
    device       char   * dst  [[buffer(3)]],
    uint3  tgpig[[threadgroup_position_in_grid]],
    ushort tiisg[[thread_index_in_simdgroup]],
    ushort sgitg[[simdgroup_index_in_threadgroup]])
{
    const short NSG = 2;
    const short nr0 = 1;
    const int nb = args.ne00 / QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;

    const uint64_t offset0 = first_row * args.nb01 + (i12 / args.r2) * args.nb02 + (i13 / args.r3) * args.nb03;
    const uint64_t offset1 =        r1 * args.nb11 + (i12        ) * args.nb12 + (i13        ) * args.nb13;

    device const block_q5_K * x = (device const block_q5_K *) (src0 + offset0);
    device const float      * yy = (device const float      *) (src1 + offset1);

    float sumf[nr0];
    for (short i = 0; i < nr0; ++i) sumf[i] = 0.f;

    float yl[16], yh[16];

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short tid = tiisg / 4;
    const short ix  = tiisg % 4;
    const short iq  = tid / 4;
    const short ir  = tid % 4;

    const short l0 = 8 * ir;
    const short q_offset = 32 * iq + l0;
    const short y_offset = 64 * iq + l0;

    const uint8_t hm1 = 1u << (2 * iq);
    const uint8_t hm2 = hm1 << 1;
    const uint8_t hm3 = hm1 << 4;
    const uint8_t hm4 = hm2 << 4;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    device const float * y1 = yy + ix * QK_K + y_offset;

    for (int i = ix; i < nb; i += 4) {
        device const uint8_t * q1 = x[i].qs + q_offset;
        device const uint8_t * qh = x[i].qh + l0;
        device const half * dh = &x[i].d;
        device const uint16_t * a = (device const uint16_t *)x[i].scales + iq;

        device const float * y2 = y1 + 128;
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short l = 0; l < 8; ++l) {
            yl[l+0] = y1[l +  0]; sumy[0] += yl[l+0];
            yl[l+8] = y1[l + 32]; sumy[1] += yl[l+8];
            yh[l+0] = y2[l +  0]; sumy[2] += yh[l+0];
            yh[l+8] = y2[l + 32]; sumy[3] += yh[l+8];
        }

        for (short row = 0; row < nr0; ++row) {
            device const uint8_t * q2 = q1 + 64;

            sc16[0] = a[0] & kmask1;
            sc16[1] = a[2] & kmask1;
            sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
            sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);

            float4 acc1 = {0.f};
            float4 acc2 = {0.f};
            FOR_UNROLL (short l = 0; l < 8; ++l) {
                uint8_t h = qh[l];
                acc1[0] += yl[l+0] * (q1[l] & 0x0F);
                acc1[1] += yl[l+8] * (q1[l] & 0xF0);
                acc1[2] += yh[l+0] * (q2[l] & 0x0F);
                acc1[3] += yh[l+8] * (q2[l] & 0xF0);
                acc2[0] += h & hm1 ? yl[l+0] : 0.f;
                acc2[1] += h & hm2 ? yl[l+8] : 0.f;
                acc2[2] += h & hm3 ? yh[l+0] : 0.f;
                acc2[3] += h & hm4 ? yh[l+8] : 0.f;
            }

            sumf[row] += dh[0] * (sc8[0] * (acc1[0]      + 16.f*acc2[0]) +
                                   sc8[1] * (acc1[1]/16.f + 16.f*acc2[1]) +
                                   sc8[4] * (acc1[2]      + 16.f*acc2[2]) +
                                   sc8[5] * (acc1[3]/16.f + 16.f*acc2[3])) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += args.nb01;
            qh += args.nb01;
            dh += args.nb01 / 2;
            a  += args.nb01 / 2;
        }

        y1 += 4 * QK_K;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = tot;
        }
    }
}

// ---- Q6_K SIMD-optimized matvec (ported from ggml-metal) ----
// Uses 32-thread SIMD groups, NR0=2 rows per SG, N_SG=2 SGs per threadgroup.
// Direct bit extraction instead of full dequantize — much faster than naive version.

kernel void kernel_mul_mv_q6_K_f32(
    constant ds4_metal_args_mul_mv & args [[buffer(0)]],
    device const char   * src0 [[buffer(1)]],
    device const char   * src1 [[buffer(2)]],
    device       char   * dst  [[buffer(3)]],
    uint3  tgpig[[threadgroup_position_in_grid]],
    ushort tiisg[[thread_index_in_simdgroup]],
    ushort sgitg[[simdgroup_index_in_threadgroup]])
{
    const short NSG = 2;
    const short nr0 = 2;

    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;

    const int nb = args.ne00 / QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;

    const uint64_t offset0 = first_row * args.nb01 + (i12 / args.r2) * args.nb02 + (i13 / args.r3) * args.nb03;
    const uint64_t offset1 =        r1 * args.nb11 + (i12        ) * args.nb12 + (i13        ) * args.nb13;

    device const block_q6_K * x = (device const block_q6_K *) (src0 + offset0);
    device const float      * yy = (device const float      *) (src1 + offset1);

    float sumf[nr0] = { 0.f };

    float yl[16];

    const short tid = tiisg / 2;
    const short ix  = tiisg % 2;
    const short ip  = tid / 8;
    const short il  = tid % 8;
    const short l0  = 4 * il;
    const short is  = 8 * ip + l0 / 16;

    const short y_offset   = 128 * ip + l0;
    const short q_offset_l =  64 * ip + l0;
    const short q_offset_h =  32 * ip + l0;

    for (int i = ix; i < nb; i += 2) {
        device const uint8_t * q1 = x[i].ql + q_offset_l;
        device const uint8_t * q2 = q1 + 32;
        device const uint8_t * qh = x[i].qh + q_offset_h;
        device const int8_t  * sc = x[i].scales + is;
        device const half    * dh = &x[i].d;

        device const float * y = yy + i * QK_K + y_offset;

        for (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = y[l +  0];
            yl[4*l + 1] = y[l + 32];
            yl[4*l + 2] = y[l + 64];
            yl[4*l + 3] = y[l + 96];
        }

        for (short row = 0; row < nr0; ++row) {
            float4 sums = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * ((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * ((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }

            sumf[row] += dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);

            q1 += args.nb01;
            q2 += args.nb01;
            qh += args.nb01;
            sc += args.nb01;
            dh += args.nb01 / 2;
        }
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

// AXPY: dst[i] += alpha * src[i]
kernel void kernel_l26f_axpy(
        device        float * dst   [[buffer(0)]],
        device  const float * src   [[buffer(1)]],
        constant      float  & alpha [[buffer(2)]],
        constant      uint   & n     [[buffer(3)]],
        uint tg [[thread_position_in_threadgroup]],
        uint tg_total [[threads_per_threadgroup]]) {
    for (uint i = tg; i < n; i += tg_total) {
        dst[i] += alpha * src[i];
    }
}

// ---- MoE Expert Router ----
//
// Input: router_logits[E] (raw logits from gate_inp matvec)
// Input: exp_probs_b[E] (bias, may be NULL — pass same as logits to skip)
// Output: selected_indices[K] (top-8 expert indices)
// Output: selected_weights[K] (normalized weights, scaled by w_scale=2.5)
//
// Algorithm:
//   1. Add bias to logits
//   2. Softmax over all E=256 experts
//   3. Group scoring: 8 groups × 32 experts, score = sum(top-2 probs per group)
//   4. Select top-4 groups
//   5. Mask: zero out experts not in top-4 groups
//   6. Select top-8 from masked pool
//   7. Normalize weights, scale by w_scale
//
// Single 256-threadgroup route kernel. E=256 is small, but serializing the
// softmax exp/sum path in one GPU thread costs ~10 tok/s on decode.

typedef struct {
    int32_t n_expert;
    int32_t n_groups;
    int32_t n_exp_per_group;
    int32_t n_top_groups;
    int32_t n_selected;
    float   w_scale;
    int32_t has_bias;
} l26f_kargs_moe_route;

typedef struct {
    int32_t n_expert;
    int32_t n_groups;
    int32_t n_exp_per_group;
    int32_t n_top_groups;
    int32_t n_selected;
    float   w_scale;
    int32_t has_bias;
    int32_t n_tokens;
} l26f_kargs_moe_route_batch;

kernel void kernel_l26f_moe_route(
        constant l26f_kargs_moe_route & args  [[buffer(0)]],
        device const float             * logits [[buffer(1)]],
        device const float             * bias   [[buffer(2)]],
        device       int32_t           * sel_idx [[buffer(3)]],
        device       float             * sel_wt  [[buffer(4)]],
        uint tid [[thread_index_in_threadgroup]])
{
    const int E = args.n_expert;
    const int G = args.n_groups;
    const int EPG = args.n_exp_per_group;

    threadgroup float probs[256];
    threadgroup float scratch[256];
    threadgroup float group_scores[8];
    threadgroup int   group_selected[8];

    float v = -INFINITY;
    if ((int)tid < E) {
        v = logits[tid];
        if (args.has_bias) v += bias[tid];
        probs[tid] = v;
    }
    scratch[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride && scratch[tid + stride] > scratch[tid]) {
            scratch[tid] = scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float maxv = scratch[0];

    float ev = 0.0f;
    if ((int)tid < E) {
        ev = exp(probs[tid] - maxv);
        probs[tid] = ev;
    }
    scratch[tid] = ev;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float sum = scratch[0];

    if ((int)tid < E) {
        probs[tid] /= sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if ((int)tid < G) {
        const int g = (int)tid;
        float t0 = 0, t1 = 0;
        for (int i = 0; i < EPG; i++) {
            float p = probs[g * EPG + i];
            if (p > t0) { t1 = t0; t0 = p; }
            else if (p > t1) { t1 = p; }
        }
        group_scores[g] = t0 + t1;
        group_selected[g] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        int top_groups[8];
        for (int i = 0; i < G; i++) top_groups[i] = i;
        for (int i = 0; i < args.n_top_groups; i++) {
            int best = i;
            for (int j = i + 1; j < G; j++) {
                if (group_scores[top_groups[j]] > group_scores[top_groups[best]])
                    best = j;
            }
            int tmp = top_groups[i]; top_groups[i] = top_groups[best]; top_groups[best] = tmp;
            group_selected[top_groups[i]] = 1;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if ((int)tid < E) {
        int g = (int)tid / EPG;
        scratch[tid] = group_selected[g] ? probs[tid] : -1e30f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        for (int i = 0; i < args.n_selected; i++) {
            int best_e = 0;
            float best_p = -1e30f;
            for (int e = 0; e < E; e++) {
                if (scratch[e] > best_p) { best_p = scratch[e]; best_e = e; }
            }
            sel_idx[i] = best_e;
            sel_wt[i] = probs[best_e];
            scratch[best_e] = -1e30f;
        }

        float wsum = 0;
        for (int i = 0; i < args.n_selected; i++) wsum += sel_wt[i];
        if (wsum > 1e-6f) {
            for (int i = 0; i < args.n_selected; i++) sel_wt[i] /= wsum;
        }
        for (int i = 0; i < args.n_selected; i++) sel_wt[i] *= args.w_scale;
    }
}

kernel void kernel_l26f_moe_route_offsets(
        constant l26f_kargs_moe_route & args       [[buffer(0)]],
        device const float            * logits     [[buffer(1)]],
        device const float            * bias       [[buffer(2)]],
        device       int32_t          * sel_idx    [[buffer(3)]],
        device       float            * sel_wt     [[buffer(4)]],
        device const uint64_t         * all_gate   [[buffer(5)]],
        device const uint64_t         * all_up     [[buffer(6)]],
        device const uint64_t         * all_down   [[buffer(7)]],
        device       uint64_t         * out_gate   [[buffer(8)]],
        device       uint64_t         * out_up     [[buffer(9)]],
        device       uint64_t         * out_down   [[buffer(10)]],
        uint tid [[thread_index_in_threadgroup]])
{
    const int E = args.n_expert;
    const int G = args.n_groups;
    const int EPG = args.n_exp_per_group;

    threadgroup float probs[256];
    threadgroup float scratch[256];
    threadgroup float group_scores[8];
    threadgroup int   group_selected[8];

    float v = -INFINITY;
    if ((int)tid < E) {
        v = logits[tid];
        if (args.has_bias) v += bias[tid];
        probs[tid] = v;
    }
    scratch[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride && scratch[tid + stride] > scratch[tid]) {
            scratch[tid] = scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float maxv = scratch[0];

    float ev = 0.0f;
    if ((int)tid < E) {
        ev = exp(probs[tid] - maxv);
        probs[tid] = ev;
    }
    scratch[tid] = ev;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float sum = scratch[0];

    if ((int)tid < E) {
        probs[tid] /= sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if ((int)tid < G) {
        const int g = (int)tid;
        float t0 = 0, t1 = 0;
        for (int i = 0; i < EPG; i++) {
            float p = probs[g * EPG + i];
            if (p > t0) { t1 = t0; t0 = p; }
            else if (p > t1) { t1 = p; }
        }
        group_scores[g] = t0 + t1;
        group_selected[g] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        int top_groups[8];
        for (int i = 0; i < G; i++) top_groups[i] = i;
        for (int i = 0; i < args.n_top_groups; i++) {
            int best = i;
            for (int j = i + 1; j < G; j++) {
                if (group_scores[top_groups[j]] > group_scores[top_groups[best]])
                    best = j;
            }
            int tmp = top_groups[i]; top_groups[i] = top_groups[best]; top_groups[best] = tmp;
            group_selected[top_groups[i]] = 1;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if ((int)tid < E) {
        int g = (int)tid / EPG;
        scratch[tid] = group_selected[g] ? probs[tid] : -1e30f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        for (int i = 0; i < args.n_selected; i++) {
            int best_e = 0;
            float best_p = -1e30f;
            for (int e = 0; e < E; e++) {
                if (scratch[e] > best_p) { best_p = scratch[e]; best_e = e; }
            }
            sel_idx[i] = best_e;
            sel_wt[i] = probs[best_e];
            out_gate[i] = all_gate[best_e];
            out_up[i]   = all_up[best_e];
            out_down[i] = all_down[best_e];
            scratch[best_e] = -1e30f;
        }

        float wsum = 0;
        for (int i = 0; i < args.n_selected; i++) wsum += sel_wt[i];
        if (wsum > 1e-6f) {
            for (int i = 0; i < args.n_selected; i++) sel_wt[i] /= wsum;
        }
        for (int i = 0; i < args.n_selected; i++) sel_wt[i] *= args.w_scale;
    }
}

kernel void kernel_l26f_moe_route_batch(
        constant l26f_kargs_moe_route_batch & args  [[buffer(0)]],
        device const float             * logits [[buffer(1)]],
        device const float             * bias   [[buffer(2)]],
        device       int32_t           * sel_idx [[buffer(3)]],
        device       float             * sel_wt  [[buffer(4)]],
        uint gid [[thread_position_in_grid]])
{
    const int T = args.n_tokens;
    if ((int)gid >= T) return;

    const int E = args.n_expert;
    const int G = args.n_groups;
    const int EPG = args.n_exp_per_group;
    const int K = args.n_selected;

    device const float *logits_t = logits + (uint64_t)gid * E;
    device int32_t     *idx_t    = sel_idx + (uint64_t)gid * K;
    device float       *wt_t     = sel_wt  + (uint64_t)gid * K;

    float probs[256];
    for (int i = 0; i < E; i++) {
        probs[i] = logits_t[i];
        if (args.has_bias) probs[i] += bias[i];
    }

    float maxv = probs[0];
    for (int i = 1; i < E; i++) if (probs[i] > maxv) maxv = probs[i];
    float sum = 0;
    for (int i = 0; i < E; i++) { probs[i] = exp(probs[i] - maxv); sum += probs[i]; }
    for (int i = 0; i < E; i++) probs[i] /= sum;

    float group_scores[8];
    for (int g = 0; g < G; g++) {
        float t0 = 0, t1 = 0;
        for (int i = 0; i < EPG; i++) {
            float p = probs[g * EPG + i];
            if (p > t0) { t1 = t0; t0 = p; }
            else if (p > t1) { t1 = p; }
        }
        group_scores[g] = t0 + t1;
    }

    int top_groups[8];
    for (int i = 0; i < G; i++) top_groups[i] = i;
    for (int i = 0; i < args.n_top_groups; i++) {
        int best = i;
        for (int j = i + 1; j < G; j++) {
            if (group_scores[top_groups[j]] > group_scores[top_groups[best]])
                best = j;
        }
        int tmp = top_groups[i]; top_groups[i] = top_groups[best]; top_groups[best] = tmp;
    }

    float masked[256];
    for (int i = 0; i < E; i++) masked[i] = probs[i];
    for (int g = 0; g < G; g++) {
        bool sel = false;
        for (int i = 0; i < args.n_top_groups; i++)
            if (top_groups[i] == g) { sel = true; break; }
        if (!sel) {
            for (int i = 0; i < EPG; i++)
                masked[g * EPG + i] = -1e30f;
        }
    }

    for (int i = 0; i < K; i++) {
        int best_e = 0;
        float best_p = -1e30f;
        for (int e = 0; e < E; e++) {
            if (masked[e] > best_p) { best_p = masked[e]; best_e = e; }
        }
        idx_t[i] = best_e;
        wt_t[i] = probs[best_e];
        masked[best_e] = -1e30f;
    }

    float wsum = 0;
    for (int i = 0; i < K; i++) wsum += wt_t[i];
    if (wsum > 1e-6f) {
        for (int i = 0; i < K; i++) wt_t[i] /= wsum;
    }
    for (int i = 0; i < K; i++) wt_t[i] *= args.w_scale;
}

// ---- Fused MoE Expert Matvec (IQ4_NL) ----
//
// For each of K=8 selected experts, compute one row of the weight matvec.
// Layout: grid(K * out_rows, 1, 1), each thread handles one output element.
//
// Arguments:
//   weights:    base pointer to expert weights (from mmap)
//   input:      [in_dim] float input vector (shared across experts)
//   output:     [K * out_rows] float output (contiguous per expert)
//   offsets:    [K] uint64 expert weight offsets
//   sel_idx:    [K] int32 selected expert indices
//   expert_row_bytes: [256] uint64 row bytes per expert (for variable types)
//
// This kernel allows dispatching all 8 expert matvecs in a single GPU call,
// eliminating command buffer breaks between experts.

typedef struct {
    int32_t n_experts;
    int32_t in_dim;
    int32_t out_rows;
    int32_t block_size;
    int32_t type_size;
    int32_t per_expert_input;
    int32_t nsg;
} l26f_kargs_fused_moe_iq4nl;

kernel void kernel_l26f_fused_moe_iq4nl(
        constant l26f_kargs_fused_moe_iq4nl & args [[buffer(0)]],
        device const char     * weights     [[buffer(1)]],
        device const float    * input       [[buffer(2)]],
        device       float    * output      [[buffer(3)]],
        device const uint64_t * offsets     [[buffer(4)]],
        threadgroup float     * shmem       [[threadgroup(0)]],
        uint2  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]])
{
    const int NR0 = 2;
    const int NSG = args.nsg;
    const int e = (int)tgpig.y;
    if (e >= args.n_experts) return;

    const int n_blocks = args.in_dim / 32;
    const int first_row = ((int)tgpig.x * NSG + (int)sgitg) * NR0;

    device const block_iq4_nl * w = (device const block_iq4_nl *)
        (weights + offsets[e]);
    const uint64_t in_off = args.per_expert_input ? (uint64_t)e * args.in_dim : 0;
    device const float * y = input + in_off;

    const short ix = tiisg / 2;
    const short it = tiisg % 2;

    shmem[tiisg] = kvalues_iq4nl_f[tiisg % 16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[2] = {0.f};

    device const float * yb = y + ix * 32 + it * 8;

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    float4 qf1, qf2;

    for (int ib = ix; ib < n_blocks; ib += 16) {
        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        for (short row = 0; row < NR0; row++) {
            if (first_row + row >= args.out_rows) break;
            device const block_iq4_nl & xb = w[(uint64_t)(first_row + row) * n_blocks + ib];
            device const uint16_t * q4 = (device const uint16_t *)(xb.qs + 8*it);

            float4 acc1 = {0.f}, acc2 = {0.f};

            aux32[0] = q4[0] | (q4[1] << 16);
            aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
            aux32[0] &= 0x0f0f0f0f;
            qf1 = {shmem[q8[0]], shmem[q8[1]], shmem[q8[2]], shmem[q8[3]]};
            qf2 = {shmem[q8[4]], shmem[q8[5]], shmem[q8[6]], shmem[q8[7]]};
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

    for (int row = 0; row < NR0 && first_row + row < args.out_rows; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            output[(uint64_t)e * args.out_rows + first_row + row] = sum_all;
        }
    }
}

// ---- Fused MoE Gate+Up Matvec (IQ4_NL) ----
// Computes BOTH gate and up matvecs in one pass, reading input once.

typedef struct {
    int32_t n_experts;
    int32_t in_dim;
    int32_t out_rows;
    int32_t block_size;
    int32_t type_size;
} l26f_kargs_fused_moe_iq4nl_gate_up;

kernel void kernel_l26f_fused_moe_iq4nl_gate_up(
        constant l26f_kargs_fused_moe_iq4nl_gate_up & args [[buffer(0)]],
        device const char     * weights_gate  [[buffer(1)]],
        device const char     * weights_up    [[buffer(2)]],
        device const float    * input         [[buffer(3)]],
        device       float    * output_gate   [[buffer(4)]],
        device       float    * output_up     [[buffer(5)]],
        device const uint64_t * offsets_gate  [[buffer(6)]],
        device const uint64_t * offsets_up    [[buffer(7)]],
        threadgroup float     * shmem         [[threadgroup(0)]],
        uint2  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]])
{
    (void)sgitg;
    const int NR0 = 2;
    const int e = (int)tgpig.y;
    if (e >= args.n_experts) return;

    const int n_blocks = args.in_dim / 32;
    const int first_row = (int)tgpig.x * NR0;

    device const block_iq4_nl * w_gate = (device const block_iq4_nl *)
        (weights_gate + offsets_gate[e]);
    device const block_iq4_nl * w_up   = (device const block_iq4_nl *)
        (weights_up   + offsets_up[e]);
    device const float * y = input;

    const short ix = tiisg / 2;
    const short it = tiisg % 2;

    shmem[tiisg] = kvalues_iq4nl_f[tiisg % 16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf_gate[2] = {0.f};
    float sumf_up[2]   = {0.f};

    device const float * yb = y + ix * 32 + it * 8;

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    float4 qf1, qf2;

    for (int ib = ix; ib < n_blocks; ib += 16) {
        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        for (short row = 0; row < NR0; row++) {
            if (first_row + row >= args.out_rows) break;

            // Gate matvec
            {
                device const block_iq4_nl & xb = w_gate[(uint64_t)(first_row + row) * n_blocks + ib];
                device const uint16_t * q4 = (device const uint16_t *)(xb.qs + 8*it);
                float4 acc1 = {0.f}, acc2 = {0.f};
                aux32[0] = q4[0] | (q4[1] << 16);
                aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
                aux32[0] &= 0x0f0f0f0f;
                qf1 = {shmem[q8[0]], shmem[q8[1]], shmem[q8[2]], shmem[q8[3]]};
                qf2 = {shmem[q8[4]], shmem[q8[5]], shmem[q8[6]], shmem[q8[7]]};
                acc1 += yl[0] * qf1; acc2 += yl[1] * qf2;
                aux32[0] = q4[2] | (q4[3] << 16);
                aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
                aux32[0] &= 0x0f0f0f0f;
                qf1 = {shmem[q8[0]], shmem[q8[1]], shmem[q8[2]], shmem[q8[3]]};
                qf2 = {shmem[q8[4]], shmem[q8[5]], shmem[q8[6]], shmem[q8[7]]};
                acc1 += yl[2] * qf1; acc2 += yl[3] * qf2;
                acc1 += acc2;
                sumf_gate[row] += (float)xb.d * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
            }

            // Up matvec (same input, different weights)
            {
                device const block_iq4_nl & xb = w_up[(uint64_t)(first_row + row) * n_blocks + ib];
                device const uint16_t * q4 = (device const uint16_t *)(xb.qs + 8*it);
                float4 acc1 = {0.f}, acc2 = {0.f};
                aux32[0] = q4[0] | (q4[1] << 16);
                aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
                aux32[0] &= 0x0f0f0f0f;
                qf1 = {shmem[q8[0]], shmem[q8[1]], shmem[q8[2]], shmem[q8[3]]};
                qf2 = {shmem[q8[4]], shmem[q8[5]], shmem[q8[6]], shmem[q8[7]]};
                acc1 += yl[0] * qf1; acc2 += yl[1] * qf2;
                aux32[0] = q4[2] | (q4[3] << 16);
                aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
                aux32[0] &= 0x0f0f0f0f;
                qf1 = {shmem[q8[0]], shmem[q8[1]], shmem[q8[2]], shmem[q8[3]]};
                qf2 = {shmem[q8[4]], shmem[q8[5]], shmem[q8[6]], shmem[q8[7]]};
                acc1 += yl[2] * qf1; acc2 += yl[3] * qf2;
                acc1 += acc2;
                sumf_up[row] += (float)xb.d * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
            }
        }

        yb += 16 * 32;
    }

    for (int row = 0; row < NR0 && first_row + row < args.out_rows; ++row) {
        float gate_sum = simd_sum(sumf_gate[row]);
        float up_sum   = simd_sum(sumf_up[row]);
        if (tiisg == 0) {
            output_gate[(uint64_t)e * args.out_rows + first_row + row] = gate_sum;
            output_up  [(uint64_t)e * args.out_rows + first_row + row] = up_sum;
        }
    }
}

// ---- Fused MoE Expert Gate+Up+SwiGLU (IQ4_NL) ----
// Like kernel_l26f_fused_moe_iq4nl_gate_up but applies swiglu inline.
// Writes mid[e] = SiLU(gate[e]) * up[e] directly, eliminating separate swiglu dispatch.

kernel void kernel_l26f_fused_moe_iq4nl_gate_up_swiglu(
        constant l26f_kargs_fused_moe_iq4nl_gate_up & args [[buffer(0)]],
        device const char     * weights_gate  [[buffer(1)]],
        device const char     * weights_up    [[buffer(2)]],
        device const float    * input         [[buffer(3)]],
        device       float    * output_mid    [[buffer(4)]],
        device const uint64_t * offsets_gate  [[buffer(5)]],
        device const uint64_t * offsets_up    [[buffer(6)]],
        threadgroup float     * shmem         [[threadgroup(0)]],
        uint2  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]])
{
    (void)sgitg;
    const int NR0 = 2;
    const int e = (int)tgpig.y;
    if (e >= args.n_experts) return;

    const int n_blocks = args.in_dim / 32;
    const int first_row = (int)tgpig.x * NR0;

    device const block_iq4_nl * w_gate = (device const block_iq4_nl *)
        (weights_gate + offsets_gate[e]);
    device const block_iq4_nl * w_up   = (device const block_iq4_nl *)
        (weights_up   + offsets_up[e]);
    device const float * y = input;

    const short ix = tiisg / 2;
    const short it = tiisg % 2;

    shmem[tiisg] = kvalues_iq4nl_f[tiisg % 16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf_gate[2] = {0.f};
    float sumf_up[2]   = {0.f};

    device const float * yb = y + ix * 32 + it * 8;

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    float4 qf1, qf2;

    for (int ib = ix; ib < n_blocks; ib += 16) {
        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        for (short row = 0; row < NR0; row++) {
            if (first_row + row >= args.out_rows) break;

            {
                device const block_iq4_nl & xb = w_gate[(uint64_t)(first_row + row) * n_blocks + ib];
                device const uint16_t * q4 = (device const uint16_t *)(xb.qs + 8*it);
                float4 acc1 = {0.f}, acc2 = {0.f};
                aux32[0] = q4[0] | (q4[1] << 16);
                aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
                aux32[0] &= 0x0f0f0f0f;
                qf1 = {shmem[q8[0]], shmem[q8[1]], shmem[q8[2]], shmem[q8[3]]};
                qf2 = {shmem[q8[4]], shmem[q8[5]], shmem[q8[6]], shmem[q8[7]]};
                acc1 += yl[0] * qf1; acc2 += yl[1] * qf2;
                aux32[0] = q4[2] | (q4[3] << 16);
                aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
                aux32[0] &= 0x0f0f0f0f;
                qf1 = {shmem[q8[0]], shmem[q8[1]], shmem[q8[2]], shmem[q8[3]]};
                qf2 = {shmem[q8[4]], shmem[q8[5]], shmem[q8[6]], shmem[q8[7]]};
                acc1 += yl[2] * qf1; acc2 += yl[3] * qf2;
                acc1 += acc2;
                sumf_gate[row] += (float)xb.d * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
            }

            {
                device const block_iq4_nl & xb = w_up[(uint64_t)(first_row + row) * n_blocks + ib];
                device const uint16_t * q4 = (device const uint16_t *)(xb.qs + 8*it);
                float4 acc1 = {0.f}, acc2 = {0.f};
                aux32[0] = q4[0] | (q4[1] << 16);
                aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
                aux32[0] &= 0x0f0f0f0f;
                qf1 = {shmem[q8[0]], shmem[q8[1]], shmem[q8[2]], shmem[q8[3]]};
                qf2 = {shmem[q8[4]], shmem[q8[5]], shmem[q8[6]], shmem[q8[7]]};
                acc1 += yl[0] * qf1; acc2 += yl[1] * qf2;
                aux32[0] = q4[2] | (q4[3] << 16);
                aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
                aux32[0] &= 0x0f0f0f0f;
                qf1 = {shmem[q8[0]], shmem[q8[1]], shmem[q8[2]], shmem[q8[3]]};
                qf2 = {shmem[q8[4]], shmem[q8[5]], shmem[q8[6]], shmem[q8[7]]};
                acc1 += yl[2] * qf1; acc2 += yl[3] * qf2;
                acc1 += acc2;
                sumf_up[row] += (float)xb.d * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
            }
        }

        yb += 16 * 32;
    }

    for (int row = 0; row < NR0 && first_row + row < args.out_rows; ++row) {
        float gate_val = simd_sum(sumf_gate[row]);
        float up_val   = simd_sum(sumf_up[row]);
        if (tiisg == 0) {
            float sv = 1.0f / (1.0f + exp(-gate_val));
            output_mid[(uint64_t)e * args.out_rows + first_row + row] = gate_val * sv * up_val;
        }
    }
}

// ---- Fused MoE Expert Matvec (Q5_K) ----
// Same structure as IQ4_NL version but uses Q5_K SIMD-optimized matvec.
// Ported from ggml-metal's kernel_mul_mv_q5_K_f32_impl with expert offset indirection.

typedef struct {
    int32_t n_experts;
    int32_t in_dim;
    int32_t out_rows;
} l26f_kargs_fused_moe_q5k;

kernel void kernel_l26f_fused_moe_q5k(
        constant l26f_kargs_fused_moe_q5k & args [[buffer(0)]],
        device const char     * weights     [[buffer(1)]],
        device const float    * input       [[buffer(2)]],
        device       float    * output      [[buffer(3)]],
        device const uint64_t * offsets     [[buffer(4)]],
        uint2  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]])
{
    const short NSG = 1;
    const short nr0 = 1;
    const int e = (int)tgpig.y;
    if (e >= args.n_experts) return;

    const int nb = args.in_dim / QK_K;
    const int first_row = (int)tgpig.x * NSG + sgitg * nr0;
    if (first_row >= args.out_rows) return;

    const uint64_t row_bytes = (uint64_t)(args.in_dim / 256) * 176;
    device const block_q5_K * x = (device const block_q5_K *)
        (weights + offsets[e]) + (uint64_t)first_row * (args.in_dim / 256);
    device const float * yy = input + (uint64_t)e * args.in_dim;

    float sumf[1] = {0.f};

    float yl[16], yh[16];

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short tid = tiisg / 4;
    const short ix  = tiisg % 4;
    const short iq  = tid / 4;
    const short ir  = tid % 4;

    const short l0 = 8 * ir;
    const short q_offset = 32 * iq + l0;
    const short y_offset = 64 * iq + l0;

    const uint8_t hm1 = 1u << (2 * iq);
    const uint8_t hm2 = hm1 << 1;
    const uint8_t hm3 = hm1 << 4;
    const uint8_t hm4 = hm2 << 4;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    device const float * y1 = yy + ix * QK_K + y_offset;

    for (int i = ix; i < nb; i += 4) {
        device const uint8_t * q1 = x[i].qs + q_offset;
        device const uint8_t * qh = x[i].qh + l0;
        device const half * dh = &x[i].d;
        device const uint16_t * a = (device const uint16_t *)x[i].scales + iq;

        device const float * y2 = y1 + 128;
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short l = 0; l < 8; ++l) {
            yl[l+0] = y1[l +  0]; sumy[0] += yl[l+0];
            yl[l+8] = y1[l + 32]; sumy[1] += yl[l+8];
            yh[l+0] = y2[l +  0]; sumy[2] += yh[l+0];
            yh[l+8] = y2[l + 32]; sumy[3] += yh[l+8];
        }

        device const uint8_t * q2 = q1 + 64;

        sc16[0] = a[0] & kmask1;
        sc16[1] = a[2] & kmask1;
        sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
        sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);

        float4 acc1 = {0.f};
        float4 acc2 = {0.f};
        FOR_UNROLL (short l = 0; l < 8; ++l) {
            uint8_t h = qh[l];
            acc1[0] += yl[l+0] * (q1[l] & 0x0F);
            acc1[1] += yl[l+8] * (q1[l] & 0xF0);
            acc1[2] += yh[l+0] * (q2[l] & 0x0F);
            acc1[3] += yh[l+8] * (q2[l] & 0xF0);
            acc2[0] += h & hm1 ? yl[l+0] : 0.f;
            acc2[1] += h & hm2 ? yl[l+8] : 0.f;
            acc2[2] += h & hm3 ? yh[l+0] : 0.f;
            acc2[3] += h & hm4 ? yh[l+8] : 0.f;
        }

        sumf[0] += dh[0] * (sc8[0] * (acc1[0]      + 16.f*acc2[0]) +
                             sc8[1] * (acc1[1]/16.f + 16.f*acc2[1]) +
                             sc8[4] * (acc1[2]      + 16.f*acc2[2]) +
                             sc8[5] * (acc1[3]/16.f + 16.f*acc2[3])) -
                   dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

        y1 += 4 * QK_K;
    }

    {
        const float tot = simd_sum(sumf[0]);
        if (tiisg == 0) {
            output[(uint64_t)e * args.out_rows + first_row] = tot;
        }
    }
}

// ---- Fused MoE SwiGLU + Accumulate ----
//
// For each of K=8 experts:
//   mid[e][j] = gate[e][j] * sigmoid(gate[e][j]) * up[e][j]
// Then accumulate weighted experts:
//   out[j] += weight[e] * mid[e][j]
//
// But we also need the down matvec between swiglu and accumulate.
// So this kernel just does the swiglu part, storing mid[e] contiguously.

typedef struct {
    int32_t n_experts;
    int32_t n_elements;
} l26f_kargs_fused_swiglu;

kernel void kernel_l26f_fused_swiglu(
        constant l26f_kargs_fused_swiglu & args [[buffer(0)]],
        device const float * gate   [[buffer(1)]],
        device const float * up     [[buffer(2)]],
        device       float * mid    [[buffer(3)]],
        uint gid [[thread_position_in_grid]])
{
    const int total = args.n_experts * args.n_elements;
    if ((int)gid >= total) return;

    const int e = (int)gid / args.n_elements;
    const int j = (int)gid % args.n_elements;

    float g = gate[(uint64_t)e * args.n_elements + j];
    float u = up[(uint64_t)e * args.n_elements + j];
    float sig = 1.0f / (1.0f + exp(-g));
    mid[(uint64_t)e * args.n_elements + j] = g * sig * u;
}

// ---- Fused MoE Down + Weighted Accumulate ----
//
// After down matvec produces down_out[e][j], accumulate:
//   moe_out[j] = sum_e(weight[e] * down_out[e][j]) + shared_expert[j]
//
// We combine the accumulate with the shared expert addition.

typedef struct {
    int32_t n_experts;
    int32_t n_elements;
} l26f_kargs_fused_accum;

kernel void kernel_l26f_fused_accum(
        constant l26f_kargs_fused_accum & args [[buffer(0)]],
        device const float    * expert_out  [[buffer(1)]],
        device const float    * weights     [[buffer(2)]],
        device const float    * shared_out  [[buffer(3)]],
        device       float    * moe_out     [[buffer(4)]],
        uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= args.n_elements) return;

    float sum = shared_out[gid];
    for (int e = 0; e < args.n_experts; e++) {
        sum += weights[e] * expert_out[(uint64_t)e * args.n_elements + gid];
    }
    moe_out[gid] = sum;
}

kernel void kernel_l26f_fused_accum_residual(
        constant l26f_kargs_fused_accum & args [[buffer(0)]],
        device const float    * expert_out  [[buffer(1)]],
        device const float    * weights     [[buffer(2)]],
        device const float    * shared_out  [[buffer(3)]],
        device const float    * residual    [[buffer(4)]],
        device       float    * out         [[buffer(5)]],
        uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= args.n_elements) return;

    float sum = residual[gid] + shared_out[gid];
    for (int e = 0; e < args.n_experts; e++) {
        sum += weights[e] * expert_out[(uint64_t)e * args.n_elements + gid];
    }
    out[gid] = sum;
}

// ---- Gather Expert Offsets ----
//
// Read K selected expert indices, look up offsets from a 256-entry table,
// write K offsets to output buffer. Runs with K threads.

kernel void kernel_l26f_gather_offsets(
        device const int32_t   * sel_idx   [[buffer(0)]],
        device const uint64_t  * all_off   [[buffer(1)]],
        device       uint64_t  * out_off   [[buffer(2)]],
        constant     int32_t   & K         [[buffer(3)]],
        uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= K) return;
    out_off[gid] = all_off[sel_idx[gid]];
}

// ---- Fused Gather Expert Offsets (gate + up + down) ----
// Same as above but does all three offset tables in one dispatch.

kernel void kernel_l26f_gather_offsets_3(
        device const int32_t   * sel_idx       [[buffer(0)]],
        device const uint64_t  * all_off_gate  [[buffer(1)]],
        device const uint64_t  * all_off_up    [[buffer(2)]],
        device const uint64_t  * all_off_down  [[buffer(3)]],
        device       uint64_t  * out_off_gate  [[buffer(4)]],
        device       uint64_t  * out_off_up    [[buffer(5)]],
        device       uint64_t  * out_off_down  [[buffer(6)]],
        constant     int32_t   & K             [[buffer(7)]],
        uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= K) return;
    int idx = sel_idx[gid];
    out_off_gate[gid] = all_off_gate[idx];
    out_off_up[gid]   = all_off_up[idx];
    out_off_down[gid] = all_off_down[idx];
}

// ---- Shared Expert Gate+Up+Swiglu Fusion (IQ4_NL) ----
// Single dispatch: reads input once, computes gate and up projections,
// applies SiLU(gate)*up, writes result. 3 dispatches → 1.

typedef struct {
    int32_t in_dim;
    int32_t out_dim;
} l26f_kargs_shared_gate_up_swiglu;

kernel void kernel_l26f_shared_gate_up_swiglu_iq4nl(
        constant l26f_kargs_shared_gate_up_swiglu & args [[buffer(0)]],
        device const char     * weights_gate  [[buffer(1)]],
        device const char     * weights_up    [[buffer(2)]],
        device const float    * input         [[buffer(3)]],
        device       float    * output_mid    [[buffer(4)]],
        threadgroup float     * shmem         [[threadgroup(0)]],
        uint   tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]])
{
    const int NR0 = 2;
    const int n_blocks = args.in_dim / 32;
    const int first_row = (int)tgpig * NR0;

    device const block_iq4_nl * w_gate = (device const block_iq4_nl *) weights_gate;
    device const block_iq4_nl * w_up   = (device const block_iq4_nl *) weights_up;
    device const float * y = input;

    const short ix = tiisg / 2;
    const short it = tiisg % 2;

    shmem[tiisg] = kvalues_iq4nl_f[tiisg % 16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf_gate[2] = {0.f};
    float sumf_up[2]   = {0.f};

    device const float * yb = y + ix * 32 + it * 8;

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    float4 qf1, qf2;

    for (int ib = ix; ib < n_blocks; ib += 16) {
        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        for (short row = 0; row < NR0; row++) {
            if (first_row + row >= args.out_dim) break;

            {
                device const block_iq4_nl & xb = w_gate[(uint64_t)(first_row + row) * n_blocks + ib];
                device const uint16_t * q4 = (device const uint16_t *)(xb.qs + 8*it);
                float4 acc1 = {0.f}, acc2 = {0.f};
                aux32[0] = q4[0] | (q4[1] << 16);
                aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
                aux32[0] &= 0x0f0f0f0f;
                qf1 = {shmem[q8[0]], shmem[q8[1]], shmem[q8[2]], shmem[q8[3]]};
                qf2 = {shmem[q8[4]], shmem[q8[5]], shmem[q8[6]], shmem[q8[7]]};
                acc1 += yl[0] * qf1; acc2 += yl[1] * qf2;
                aux32[0] = q4[2] | (q4[3] << 16);
                aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
                aux32[0] &= 0x0f0f0f0f;
                qf1 = {shmem[q8[0]], shmem[q8[1]], shmem[q8[2]], shmem[q8[3]]};
                qf2 = {shmem[q8[4]], shmem[q8[5]], shmem[q8[6]], shmem[q8[7]]};
                acc1 += yl[2] * qf1; acc2 += yl[3] * qf2;
                acc1 += acc2;
                sumf_gate[row] += (float)xb.d * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
            }

            {
                device const block_iq4_nl & xb = w_up[(uint64_t)(first_row + row) * n_blocks + ib];
                device const uint16_t * q4 = (device const uint16_t *)(xb.qs + 8*it);
                float4 acc1 = {0.f}, acc2 = {0.f};
                aux32[0] = q4[0] | (q4[1] << 16);
                aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
                aux32[0] &= 0x0f0f0f0f;
                qf1 = {shmem[q8[0]], shmem[q8[1]], shmem[q8[2]], shmem[q8[3]]};
                qf2 = {shmem[q8[4]], shmem[q8[5]], shmem[q8[6]], shmem[q8[7]]};
                acc1 += yl[0] * qf1; acc2 += yl[1] * qf2;
                aux32[0] = q4[2] | (q4[3] << 16);
                aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
                aux32[0] &= 0x0f0f0f0f;
                qf1 = {shmem[q8[0]], shmem[q8[1]], shmem[q8[2]], shmem[q8[3]]};
                qf2 = {shmem[q8[4]], shmem[q8[5]], shmem[q8[6]], shmem[q8[7]]};
                acc1 += yl[2] * qf1; acc2 += yl[3] * qf2;
                acc1 += acc2;
                sumf_up[row] += (float)xb.d * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
            }
        }

        yb += 16 * 32;
    }

    for (int row = 0; row < NR0 && first_row + row < args.out_dim; ++row) {
        float gate_val = simd_sum(sumf_gate[row]);
        float up_val   = simd_sum(sumf_up[row]);
        if (tiisg == 0) {
            float sv = 1.0f / (1.0f + exp(-gate_val));
            output_mid[first_row + row] = gate_val * sv * up_val;
        }
    }
}

// ---- Batch MoE Accumulation ----
// Takes [T, K, N] expert outputs + [T, K] weights → [T, N] accumulated output
// Each thread handles one (token, element) pair.

typedef struct {
    int32_t n_tokens;
    int32_t n_experts;
    int32_t n_elements;
} l26f_kargs_batch_accum;

kernel void kernel_l26f_batch_accum(
        constant l26f_kargs_batch_accum & args [[buffer(0)]],
        device const float    * expert_out  [[buffer(1)]],
        device const float    * weights     [[buffer(2)]],
        device const float    * shared_out  [[buffer(3)]],
        device       float    * moe_out     [[buffer(4)]],
        uint gid [[thread_position_in_grid]])
{
    const int T = args.n_tokens;
    const int K = args.n_experts;
    const int N = args.n_elements;

    int tid = (int)gid / N;
    int eid = (int)gid % N;
    if (tid >= T) return;

    float sum = shared_out[tid * N + eid];
    for (int k = 0; k < K; k++) {
        float w = weights[tid * K + k];
        sum += w * expert_out[tid * K * N + k * N + eid];
    }
    moe_out[tid * N + eid] = sum;
}

// ---- Batch SwiGLU for mat-mat output ----
// Input: [T*K, M] float tensor. Output: element-wise gate * sigmoid(gate) * up
// gate starts at offset 0, up starts at offset T*K*M

kernel void kernel_l26f_batch_swiglu(
        device       float * mid     [[buffer(0)]],
        device const float * gate    [[buffer(1)]],
        device const float * up      [[buffer(2)]],
        constant     int32_t & n     [[buffer(3)]],
        uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= n) return;
    float g = gate[gid];
    mid[gid] = g * (1.0f / (1.0f + exp(-g))) * up[gid];
}

// ---- Expert weight gather: copy K selected experts to contiguous cache ----
//
// Copies K experts from a strided source buffer to a contiguous destination.
// Each expert is `expert_bytes` long. Source layout: expert e is at
// src[e * expert_stride]. Destination layout: expert i is at dst[i * expert_bytes].
//
// This enables the fused matmul kernels to read sequentially, improving cache
// locality and memory coalescing by ~10x.

typedef struct {
    uint64_t expert_bytes;
    uint64_t expert_stride;
    uint32_t n_experts;
} l26f_kargs_gather_experts;

kernel void kernel_l26f_gather_experts(
        constant l26f_kargs_gather_experts & args [[buffer(0)]],
        device const char * src [[buffer(1)]],
        device const int32_t * expert_ids [[buffer(2)]],
        device char * dst [[buffer(3)]],
        uint gid [[thread_position_in_grid]])
{
    const uint64_t total = (uint64_t)args.n_experts * args.expert_bytes;
    if (gid >= total) return;

    const uint32_t e = gid / args.expert_bytes;
    const uint64_t offset_in_expert = gid % args.expert_bytes;

    const int32_t src_eid = expert_ids[e];
    const uint64_t src_off = (uint64_t)src_eid * args.expert_stride + offset_in_expert;
    const uint64_t dst_off = (uint64_t)e * args.expert_bytes + offset_in_expert;

    dst[dst_off] = src[src_off];
}

// ---- Fused Q6_K LM-head matvec + per-threadgroup argmax ----
// Computes Q6_K matvec for the output head (vocab x hidden), but instead of
// writing all logits, tracks the maximum value and index within each
// threadgroup and writes per-threadgroup (max_val, max_idx) pairs to a small
// intermediate buffer.  The host then scans this buffer to get the global argmax.
//
// Dispatch: (vocab + 3) / 4 threadgroups, (32, 2, 1) threads per group.

kernel void kernel_l26f_logits_head_q6k_argmax(
    constant ds4_metal_args_mul_mv & args [[buffer(0)]],
    device const char   * src0 [[buffer(1)]],  // Q6_K weights
    device const char   * src1 [[buffer(2)]],  // input (normed hidden, 4096 floats)
    device       float  * dst_max [[buffer(3)]], // per-tg max values
    device       int32_t * dst_idx [[buffer(4)]], // per-tg max indices
    threadgroup float   * shmem [[threadgroup(0)]],
    uint   tgpig[[threadgroup_position_in_grid]],
    ushort tiisg[[thread_index_in_simdgroup]],
    ushort sgitg[[simdgroup_index_in_threadgroup]])
{
    const short NSG = 2;
    const short nr0 = 2;

    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;

    const int nb = args.ne00 / QK_K;

    // 1-D dispatch: tgpig directly indexes the output row block.
    const int first_tg_row = (int)tgpig * NSG * nr0;
    const int first_row = first_tg_row + (int)sgitg * nr0;

    const uint64_t offset0 = (uint64_t)first_row * args.nb01;
    const uint64_t offset1 = 0;

    device const block_q6_K * x = (device const block_q6_K *) (src0 + offset0);
    device const float      * yy = (device const float      *) (src1 + offset1);

    float sumf[nr0] = { 0.f };

    float yl[16];

    const short tid = tiisg / 2;
    const short ix  = tiisg % 2;
    const short ip  = tid / 8;
    const short il  = tid % 8;
    const short l0  = 4 * il;
    const short is  = 8 * ip + l0 / 16;

    const short y_offset   = 128 * ip + l0;
    const short q_offset_l =  64 * ip + l0;
    const short q_offset_h =  32 * ip + l0;

    for (int i = ix; i < nb; i += 2) {
        device const uint8_t * q1 = x[i].ql + q_offset_l;
        device const uint8_t * q2 = q1 + 32;
        device const uint8_t * qh = x[i].qh + q_offset_h;
        device const int8_t  * sc = x[i].scales + is;
        device const half    * dh = &x[i].d;

        device const float * y = yy + i * QK_K + y_offset;

        for (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = y[l +  0];
            yl[4*l + 1] = y[l + 32];
            yl[4*l + 2] = y[l + 64];
            yl[4*l + 3] = y[l + 96];
        }

        for (short row = 0; row < nr0; ++row) {
            float4 sums = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * ((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * ((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }

            sumf[row] += dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);

            q1 += args.nb01;
            q2 += args.nb01;
            qh += args.nb01;
            sc += args.nb01;
            dh += args.nb01 / 2;
        }
    }

    // Find the max within this simdgroup's rows
    float sg_max_val = -INFINITY;
    int32_t sg_max_idx = -1;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0 && sum_all > sg_max_val) {
            sg_max_val = sum_all;
            sg_max_idx = first_row + row;
        }
    }

    // Reduce across simdgroups via threadgroup memory
    threadgroup float   tg_max_vals[2];
    threadgroup int32_t tg_max_idxs[2];

    if (tiisg == 0) {
        tg_max_vals[sgitg] = sg_max_val;
        tg_max_idxs[sgitg] = sg_max_idx;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // simdgroup 0 thread 0 writes the threadgroup result
    if (sgitg == 0 && tiisg == 0) {
        float final_max = tg_max_vals[0];
        int32_t final_idx = tg_max_idxs[0];
        if (tg_max_vals[1] > final_max) {
            final_max = tg_max_vals[1];
            final_idx = tg_max_idxs[1];
        }
        dst_max[tgpig] = final_max;
        dst_idx[tgpig] = final_idx;
    }
}
