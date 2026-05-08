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
    int32_t nr0;
    int16_t r2, r3;
};

kernel void kernel_mul_mv_iq4_nl_f32(
    constant l26f_args_mul_mv_iq4_nl & args [[buffer(0)]],
    device const char               * src0 [[buffer(1)]],
    device const float              * src1 [[buffer(2)]],
    device       float              * dst  [[buffer(3)]],
    threadgroup float               * shmem [[threadgroup(0)]],
    uint3  tgpig[[threadgroup_position_in_grid]],
    ushort tiisg[[thread_index_in_simdgroup]],
    ushort sgitg[[simdgroup_index_in_threadgroup]])
{
    const int NR0 = 2;
    const int nb  = args.ne00 / 32;
    const int nbl = args.nb01 / args.nb00;

    const int first_row = ((int)tgpig.x * NR0);

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
            dst[first_row + row] = sum_all;
        }
    }
}

// ---- IQ4_NL small-batch matvec instantiations (reuse ds4 template) ----

typedef decltype(kernel_mul_mv_ext_q4_f32_disp<2, block_iq4_nl, 32, dequantize_iq4_nl_t4>) mul_mv_ext_iq4_nl_t;

template [[host_name("kernel_mul_mv_ext_iq4_nl_f32_r1_2")]] kernel mul_mv_ext_iq4_nl_t kernel_mul_mv_ext_q4_f32_disp<2, block_iq4_nl, 32, dequantize_iq4_nl_t4>;
template [[host_name("kernel_mul_mv_ext_iq4_nl_f32_r1_3")]] kernel mul_mv_ext_iq4_nl_t kernel_mul_mv_ext_q4_f32_disp<3, block_iq4_nl, 32, dequantize_iq4_nl_t4>;
template [[host_name("kernel_mul_mv_ext_iq4_nl_f32_r1_4")]] kernel mul_mv_ext_iq4_nl_t kernel_mul_mv_ext_q4_f32_disp<4, block_iq4_nl, 32, dequantize_iq4_nl_t4>;
template [[host_name("kernel_mul_mv_ext_iq4_nl_f32_r1_5")]] kernel mul_mv_ext_iq4_nl_t kernel_mul_mv_ext_q4_f32_disp<5, block_iq4_nl, 32, dequantize_iq4_nl_t4>;

// ---- Q5_K single-token matvec ----
kernel void kernel_mul_mv_q5_K_f32(
    constant ds4_metal_args_mul_mv & args [[buffer(0)]],
    device const block_q5_K  * src0 [[buffer(1)]],
    device const float       * src1 [[buffer(2)]],
    device       float       * dst  [[buffer(3)]],
    uint   tid[[thread_position_in_grid]])
{
    const int r0 = (int)tid;
    if (r0 >= args.ne01) return;

    const int in  = args.ne00;
    float sum = 0.0f;

    device const block_q5_K * blocks = src0 + r0 * (in / 256);
    for (int ib = 0; ib < (int)(in / 256); ib++) {
        for (short il = 0; il < 16; il++) {
            float4x4 reg;
            dequantize_q5_K(&blocks[ib], il, reg);
            int base = ib * 256 + il * 16;
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 4; k++) {
                    int idx = base + j * 4 + k;
                    if (idx < in) sum += reg[j][k] * src1[idx];
                }
            }
        }
    }
    dst[r0] = sum;
}

// ---- Q6_K single-token matvec ----
kernel void kernel_mul_mv_q6_K_f32(
    constant ds4_metal_args_mul_mv & args [[buffer(0)]],
    device const block_q6_K  * src0 [[buffer(1)]],
    device const float       * src1 [[buffer(2)]],
    device       float       * dst  [[buffer(3)]],
    uint   tid[[thread_position_in_grid]])
{
    const int r0 = (int)tid;
    if (r0 >= args.ne01) return;

    const int in = args.ne00;
    float sum = 0.0f;

    device const block_q6_K * blocks = src0 + r0 * (in / 256);
    for (int ib = 0; ib < (int)(in / 256); ib++) {
        for (short il = 0; il < 16; il++) {
            float4x4 reg;
            dequantize_q6_K(&blocks[ib], il, reg);
            int base = ib * 256 + il * 16;
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 4; k++) {
                    int idx = base + j * 4 + k;
                    if (idx < in) sum += reg[j][k] * src1[idx];
                }
            }
        }
    }
    dst[r0] = sum;
}
