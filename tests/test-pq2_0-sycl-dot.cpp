// Verifies vec_dot_pq2_0_q8_1 math against float reference.
// Stubs SYCL intrinsics for host execution.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdlib>

// ---- FP16 <-> FP32 ---------------------------------------------------------
static inline float half_to_float(uint16_t h) {
    const int32_t sign = (h >> 15) & 1;
    int32_t exp = (h >> 10) & 0x1F;
    int32_t frac = h & 0x3FF;
    if (exp == 0) {
        if (frac == 0) return sign ? -0.0f : 0.0f;
        exp = 1;
        while ((frac & 0x400) == 0) { frac <<= 1; exp--; }
        frac &= 0x3FF;
    }
    exp = exp - 15 + 127;
    int32_t f = (sign << 31) | (exp << 23) | (frac << 13);
    float result; memcpy(&result, &f, 4);
    return result;
}

static uint16_t float_to_half_bits(float f) {
    uint32_t fi; memcpy(&fi, &f, 4);
    int32_t sign = (fi >> 16) & 0x8000;
    int32_t exp = ((fi >> 23) & 0xFF) - 127 + 15;
    int32_t frac = (fi >> 13) & 0x3FF;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) exp = 31;
    return (uint16_t)(sign | (exp << 10) | frac);
}

// ---- stubs -----------------------------------------------------------------
static inline int dp4a(int a, int b, int c) {
    const int8_t* pa = (const int8_t*)&a; const int8_t* pb = (const int8_t*)&b;
    return c + pa[0]*pb[0] + pa[1]*pb[1] + pa[2]*pb[2] + pa[3]*pb[3];
}

struct sycl_float2 { float x, y; };
static inline sycl_float2 half2_to_float2(uint32_t h2) {
    return { half_to_float(h2 & 0xFFFF), half_to_float(h2 >> 16) };
}

// ---- types -----------------------------------------------------------------
struct block_q8_1 { uint32_t ds; int8_t qs[32]; };
struct block_pq2_0 { uint16_t d; uint8_t qs[32]; };

static inline int get_int_from_uint8(const uint8_t* x8, int i32) {
    const uint16_t* x16 = (const uint16_t*)(x8 + 4*i32);
    return x16[0] | (x16[1] << 16);
}

static inline int get_int_b4(const int8_t* qs, int j) {
    int v; memcpy(&v, qs + j*4, 4); return v;
}

// ---- kernel transcription (matches vecdotq.hpp after k*2 fix) --------------
static inline float kernel_pq2_0(const block_pq2_0* w, const block_q8_1* y) {
    int pq2[4][2];
    for (int k = 0; k < 4; ++k) {
        pq2[k][0] = get_int_from_uint8(w->qs, k * 2);
        pq2[k][1] = get_int_from_uint8(w->qs, k * 2 + 1);
    }
    int q8[4][8];
    for (int k = 0; k < 4; ++k)
        for (int j = 0; j < 8; ++j)
            q8[k][j] = get_int_b4(y[k].qs, j);

    float sum = 0.0f;
    const float d2 = half_to_float(w->d);
    for (int k = 0; k < 4; ++k) {
        int sumi = 0;
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 4; ++j) {
                const uint8_t q = (uint8_t)((uint32_t)pq2[k][i] >> (8 * j));
                int vi = ((q >> 0) & 0x3) | (((q >> 2) & 0x3) << 8) |
                         (((q >> 4) & 0x3) << 16) | (((q >> 6) & 0x3) << 24);
                sumi = dp4a(vi, q8[k][i * 4 + j], sumi);
            }
        }
        sycl_float2 ds8f = half2_to_float2(y[k].ds);
        sum += d2 * (sumi * ds8f.x - ds8f.y);
    }
    return sum;
}

// ---- float reference -------------------------------------------------------
static float ref_float_dot(const block_pq2_0* w, float wd2, const block_q8_1* y) {
    float total = 0.0f;
    for (int k = 0; k < 4; ++k) {
        sycl_float2 ds = half2_to_float2(y[k].ds);
        const float d1 = ds.x;
        for (int i = 0; i < 32; ++i) {
            const uint8_t b = w->qs[k * 8 + i / 4];
            const int trit = (int)((b >> ((i % 4) * 2)) & 3);
            total += ((float)trit - 1.0f) * wd2 * (float)y[k].qs[i] * d1;
        }
    }
    return total;
}

// ---- dequantize stub (matches dequantize.hpp dequantize_pq2_0) -------------
static void host_dequantize_pq2_0(const block_pq2_0* x, int iqs, float& vx, float& vy) {
    const float d = half_to_float(x->d);
    const int byte_idx = iqs / 4;
    const int shift = (iqs % 4) * 2;
    const uint8_t vui = x->qs[byte_idx];
    vx = ((float)((vui >> shift) & 3) - 1.0f) * d;
    vy = ((float)((vui >> (shift + 2)) & 3) - 1.0f) * d;
}

// ---- test helpers ----------------------------------------------------------
static void fill_q8_1(block_q8_1* y, float d1, int8_t val) {
    float sumq = 32.0f * val;
    uint16_t hd = float_to_half_bits(d1);
    uint16_t hs = float_to_half_bits(d1 * sumq);
    for (int k = 0; k < 4; ++k) {
        memset(y[k].qs, val, 32);
        y[k].ds = hd | (hs << 16);
    }
}

int main(void) {
    long checks = 0, fails = 0;
    double worst_rel = 0.0;
    unsigned seed = 42;
    auto rnd = [&]() { seed = seed * 1103515245u + 12345u; return (seed >> 16) & 0xFFFF; };

    // Test 1: ternary=1 (centered=0) -> expect 0
    {
        block_pq2_0 w; memset(&w, 0, sizeof(w));
        float wd = 0.5f;
        w.d = float_to_half_bits(wd);
        memset(w.qs, 0x55, 32);
        block_q8_1 y[4]; fill_q8_1(y, 0.5f, 1);

        const float got = kernel_pq2_0(&w, y);
        const float ref = ref_float_dot(&w, wd, y);
        printf("T1: kernel=%.4f ref=%.4f expect=0\n", got, ref);
        checks++;
        if (fabs(got) > 0.01 || fabs(ref) > 0.01 || fabs(got - ref) > 0.01) {
            printf("  FAIL\n"); fails++;
        }
    }

    // Test 2: ternary=2 (+1 centered) -> expect 128 * 0.5 * 0.5 = 32.0
    {
        block_pq2_0 w; memset(&w, 0, sizeof(w));
        float wd = 0.5f;
        w.d = float_to_half_bits(wd);
        memset(w.qs, 0xFF, 32);
        block_q8_1 y[4]; fill_q8_1(y, 0.5f, 1);

        const float got = kernel_pq2_0(&w, y);
        const float ref = ref_float_dot(&w, wd, y);
        printf("T2: kernel=%.4f ref=%.4f expect=64.0\n", got, ref);
        checks++;
        if (fabs(got - 64.0f) > 0.5 || fabs(got - ref) > 0.5) {
            printf("  FAIL\n"); fails++;
        }
    }

    // Test 3: ternary=0 (-1 centered) -> expect -32.0
    {
        block_pq2_0 w; memset(&w, 0, sizeof(w));
        float wd = 0.5f;
        w.d = float_to_half_bits(wd);
        memset(w.qs, 0x00, 32);
        block_q8_1 y[4]; fill_q8_1(y, 0.5f, 1);

        const float got = kernel_pq2_0(&w, y);
        const float ref = ref_float_dot(&w, wd, y);
        printf("T3: kernel=%.4f ref=%.4f expect=-32.0\n", got, ref);
        checks++;
        if (fabs(got - (-32.0f)) > 0.5 || fabs(got - ref) > 0.5) {
            printf("  FAIL\n"); fails++;
        }
    }

    // Test 4: alternating pattern
    {
        block_pq2_0 w; memset(&w, 0, sizeof(w));
        float wd = 1.0f;
        w.d = float_to_half_bits(wd);
        for (int i = 0; i < 32; ++i) {
            w.qs[i] = ((i%3)&3) | ((((i+1)%3)&3)<<2) | ((((i+2)%3)&3)<<4) | ((i%3)&3)<<6;
        }
        block_q8_1 y[4];
        float d1 = 1.0f;
        for (int k = 0; k < 4; ++k) {
            float sumq = 0;
            for (int i = 0; i < 32; ++i) {
                y[k].qs[i] = (int8_t)(i % 5 - 2);
                sumq += (float)y[k].qs[i];
            }
            y[k].ds = float_to_half_bits(d1) | ((uint32_t)float_to_half_bits(d1 * sumq) << 16);
        }

        const float got = kernel_pq2_0(&w, y);
        const float ref = ref_float_dot(&w, wd, y);
        printf("T4: kernel=%.4f ref=%.4f\n", got, ref);
        checks++;
        if (fabs(got - ref) > 0.5) { printf("  FAIL\n"); fails++; }
    }

    // Test 5: random data (1000 trials)
    {
        for (int t = 0; t < 1000; ++t) {
            block_pq2_0 w;
            for (int i = 0; i < 32; ++i) w.qs[i] = rnd() & 0xFF;
            float wd = 0.01f + (rnd() % 1000) / 10000.0f;
            w.d = float_to_half_bits(wd);

            block_q8_1 y[4];
            for (int k = 0; k < 4; ++k) {
                float d1 = 0.005f + (rnd() % 1000) / 20000.0f;
                float sumq = 0;
                for (int i = 0; i < 32; ++i) {
                    y[k].qs[i] = (int8_t)((int)(rnd() % 255) - 127);
                    sumq += (float)y[k].qs[i];
                }
                y[k].ds = float_to_half_bits(d1) | ((uint32_t)float_to_half_bits(d1 * sumq) << 16);
            }

            const float got = kernel_pq2_0(&w, y);
            const float ref = ref_float_dot(&w, wd, y);
            double rel = fabs(ref) > 1e-6 ? fabs((double)got - ref) / fabs(ref) : fabs((double)got - ref);
            if (rel > worst_rel) worst_rel = rel;
            if (rel > 0.1 && fails < 5)
                printf("T5[%d]: kernel=%.4f ref=%.4f rel=%.3e\n", t, got, ref, rel);
            if (rel > 0.1) fails++;
            checks++;
        }
        printf("T5: worst_rel=%.3e\n", worst_rel);
    }

    // Test 6: dequantize_pq2_0 -- all known patterns
    {
        block_pq2_0 w; memset(&w, 0, sizeof(w));
        float wd = 1.5f;
        w.d = float_to_half_bits(wd);
        // qs pattern: valid ternary values 0,1,2 only
        // byte 0 = 0x55 -> ternary 1,1,1,1 -> centered 0,0,0,0
        // byte 1 = 0x84 -> ternary 2,0,1,1 -> centered +1,-1,0,0
        w.qs[0] = 0x55; // ternary 1,1,1,1 -> centered 0,0,0,0
        w.qs[1] = 0x84; // ternary 2,0,1,1 -> centered +1,-1,0,0
        w.qs[2] = 0x00; // ternary 0,0,0,0 -> centered -1,-1,-1,-1
        for (int i = 3; i < 32; ++i) w.qs[i] = 0xAA; // ternary 2,0,2,0 -> centered +1,-1,+1,-1

        // Check all 64 values (128/2 = 64 calls to dequantize since qr=2)
        int dq_checks = 0, dq_fails = 0;
        for (int iq = 0; iq < 64; ++iq) {
            float vx, vy;
            host_dequantize_pq2_0(&w, iq, vx, vy);

            // Reference: read trits directly
            int byte_idx = iq / 4;
            int shift = (iq % 4) * 2;
            uint8_t vui = w.qs[byte_idx];
            float rx = ((float)((vui >> shift) & 3) - 1.0f) * wd;
            float ry = ((float)((vui >> (shift + 2)) & 3) - 1.0f) * wd;

            checks += 2;
            dq_checks += 2;
            if (fabs(vx - rx) > 1e-5) { printf("T6 dq[%d] vx=%.4f ref=%.4f FAIL\n", iq, vx, rx); fails++; dq_fails++; }
            if (fabs(vy - ry) > 1e-5) { printf("T6 dq[%d] vy=%.4f ref=%.4f FAIL\n", iq, vy, ry); fails++; dq_fails++; }
        }

        // Spot-check specific values
        // byte 0 = 0x55, iqs=0 -> ternary 1,1 -> centered 0,0 -> expect 0,0
        { float vx, vy; host_dequantize_pq2_0(&w, 0, vx, vy);
          checks += 2; dq_checks += 2;
          if (fabs(vx) > 1e-5 || fabs(vy) > 1e-5) { printf("T6[0] vx=%.4f vy=%.4f expect 0,0 FAIL\n", vx, vy); fails+=2; dq_fails+=2; } }
        // byte 1 = 0x84 = 10000100 -> ternary 0,1,0,2 (bits 0-7) -> centered -1,0,-1,+1
        // iqs=4: byte_idx=1, shift=0 -> reads bits 0-1 (0) and 2-3 (1) -> centered -1,0 -> -wd,0
        { float vx, vy; host_dequantize_pq2_0(&w, 4, vx, vy);
          checks += 2; dq_checks += 2;
          if (fabs(vx - (-wd)) > 1e-5 || fabs(vy) > 1e-5) { printf("T6[4] vx=%.4f vy=%.4f expect %.1f,0 FAIL\n", vx, vy, -wd); fails+=2; dq_fails+=2; } }
        // byte 2 = 0x00, iqs=8 -> ternary 0,0 -> centered -1,-1 -> expect -wd,-wd
        { float vx, vy; host_dequantize_pq2_0(&w, 8, vx, vy);
          checks += 2; dq_checks += 2;
          if (fabs(vx - (-wd)) > 1e-5 || fabs(vy - (-wd)) > 1e-5) { printf("T6[8] vx=%.4f vy=%.4f expect %.1f,%.1f FAIL\n", vx, vy, -wd, -wd); fails+=2; dq_fails+=2; } }

        printf("T6: dequantize %d checks, %d fails\n", dq_checks, dq_fails);
    }

    printf("\n%d checks, %ld fails\n", checks, fails);
    return fails != 0;
}
