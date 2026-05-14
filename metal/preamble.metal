#include <metal_stdlib>
using namespace metal;

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }
#define QK8_0 32
#define N_SIMDWIDTH 32
#define N_R0_Q8_0 2
#define N_SG_Q8_0 4
#define FC_MUL_MV 600
#define FC_MUL_MM 700
#define FC_BIN 1300
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
#define M_PI_F 3.14159265358979323846f

// Reads one byte per stride to warm model-backed pages without copying the
// model. This is outside inference and exists only to reduce first-use stalls.
kernel void kernel_touch_u8_stride(
        device const uchar    *src        [[buffer(0)]],
        device uchar          *dst        [[buffer(1)]],
        constant ulong        &stride     [[buffer(2)]],
        constant ulong        &bytes      [[buffer(3)]],
        constant ulong        &dst_offset [[buffer(4)]],
        uint gid [[thread_position_in_grid]]) {
    ulong off = (ulong)gid * stride;
    if (off >= bytes) return;
    dst[dst_offset + (ulong)gid] = src[off];
}

enum ds4_sort_order {
    DS4_SORT_ORDER_ASC,
    DS4_SORT_ORDER_DESC,
};

struct block_q8_0 {
    half d;
    int8_t qs[QK8_0];
};

// --- Ling-2.6-specific quantization block types ---
#define QK4_NL 32
#define QK_K  256

// F16 t4 dequantize (needed by ds4 dense.metal matvec-ext templates)
static void dequantize_f16_t4(device const half4 * src, short il, thread float4 & reg) { reg = float4(src[il]); }

// IQ4_NL: 4.5 bpw, 32 elements per block, 18 bytes
struct block_iq4_nl {
    half d;
    uint8_t qs[QK4_NL/2];
};

// Lookup table for IQ4_NL dequantization
constant float kvalues_iq4nl_f[16] = {
    -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
      1.0f,  13.0f,  25.0f,  38.0f,  53.0f,  69.0f,  89.0f, 113.0f
};

// Q5_K: 5.4 bpw, 256 elements per block, 176 bytes
struct block_q5_K {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};

// Q6_K: 6.6 bpw, 256 elements per block, 210 bytes
struct block_q6_K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t  scales[16];
    half d;
};

_Static_assert(sizeof(struct block_iq4_nl) == 18, "IQ4_NL block must be 18 bytes");
_Static_assert(sizeof(struct block_q5_K)   == 176, "Q5_K block must be 176 bytes");
_Static_assert(sizeof(struct block_q6_K)   == 210, "Q6_K block must be 210 bytes");
