// Verifies vec_dot_ptq1_0_q8_1 math against float reference.
// Stubs the fp16 conversions for host execution.
//
// The decode loop and the dot loop are transcribed from
// ggml/src/ggml-sycl/vecdotq.hpp, split at the kernel's own decode/dot boundary.
// They are cross-checked against independent paths:
//   - element-wise trit reference (different derivation of the codec mapping)
//   - round-trip through the CPU quantizer/dequantizer (ggml-quants.c)
//   - exact-scale integer dots (any mapping or scale-mixing error changes the sum)
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

// ---- types, layout matches ggml/src/ggml-common.h ---------------------------
#define QK8_1 32
#define QK_PTQ1_0 128
struct block_q8_1 { uint32_t ds; int8_t qs[QK8_1]; }; // ds = 2 packed fp16 halves (scale, scale*sumq)
struct block_ptq1_0 { uint8_t qs[24]; uint8_t qh[2]; uint16_t d; };
static_assert(sizeof(block_q8_1) == 36, "wrong q8_1 block size");
static_assert(sizeof(block_ptq1_0) == 28, "wrong ptq1_0 block size");

// ---- kernel transcription (matches vecdotq.hpp vec_dot_ptq1_0_q8_1) ---------
// decode loop, verbatim
static void kernel_decode(const block_ptq1_0 * bptq1_0, int8_t * q) {
    static const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
    static const size_t  stages[3] = {32, 16, 8};
    int o = 0;
    size_t j = 0;
    for (size_t st = 0; st < 3; ++st) {
        const size_t c = stages[st];
        for (; j + c <= sizeof(bptq1_0->qs); j += c) {
            for (size_t nn = 0; nn < 5; ++nn) {
                for (size_t m = 0; m < c; ++m) {
                    const uint8_t v = bptq1_0->qs[j + m] * pow3[nn];
                    const int16_t xi = ((uint16_t) v * 3) >> 8;
                    q[o++] = (int8_t) (xi - 1);
                }
            }
        }
    }
    for (size_t nn = 0; nn < 4; ++nn) {
        for (size_t h = 0; h < sizeof(bptq1_0->qh); ++h) {
            const uint8_t v = bptq1_0->qh[h] * pow3[nn];
            const int16_t xi = ((uint16_t) v * 3) >> 8;
            q[o++] = (int8_t) (xi - 1);
        }
    }
}

// dot loop, verbatim; d0 is the stubbed fp16->fp32 of bptq1_0->d
static float kernel_dot(const int8_t * q, const float d0, const block_q8_1 * bq8_1) {
    float sumf = 0.0f;
    for (int k = 0; k < 4; ++k) {
        const float d1 = half_to_float(bq8_1[k].ds & 0xFFFF); // ds8f.x(), low half
        int32_t sumi = 0;
        for (int b = 0; b < 32; ++b) {
            sumi += (int32_t) q[k * 32 + b] * (int32_t) bq8_1[k].qs[b];
        }
        sumf += d1 * (float) sumi;
    }
    return d0 * sumf;
}

static float vec_dot_ptq1_0_q8_1(const block_ptq1_0 * bptq1_0, const block_q8_1 * bq8_1) {
    int8_t q[QK_PTQ1_0];
    kernel_decode(bptq1_0, q);
    return kernel_dot(q, half_to_float(bptq1_0->d), bq8_1);
}

// ---- element-wise reference (independent mapping derivation) ----------------
static int ref_trit(const block_ptq1_0 * x, int e) {
    uint8_t b; int n;
    if (e < 80)       { b = x->qs[e & 15];              n = e >> 4; }
    else if (e < 120) { const int t = e - 80; b = x->qs[16 + (t & 7)]; n = t >> 3; }
    else              { const int t = e - 120; b = x->qh[t & 1];       n = t >> 1; }
    uint32_t v = b;
    for (int i = 0; i < 4; ++i) if (i < n) v = (v * 3) & 0xFF;
    return (int)((v * 3) >> 8) - 1;
}

// ---- CPU codec transcribed from ggml-quants.c --------------------------------
static void cpu_quantize(const float * x, block_ptq1_0 * y) {
    static const size_t stages[3] = {32, 16, 8};
    float amax = 0.0f;
    for (int j = 0; j < QK_PTQ1_0; ++j) amax = fmaxf(amax, fabsf(x[j]));
    const float d  = amax;
    const float id = d ? 1.0f/d : 0.0f;
    y->d = float_to_half_bits(d);
    size_t j = 0;
    for (size_t s = 0; s < 3; ++s) {
        const size_t c = stages[s];
        for (; j + c <= sizeof(y->qs); j += c) {
            for (size_t m = 0; m < c; ++m) {
                uint8_t q = 0;
                for (size_t n = 0; n < 5; ++n) {
                    int xi = (int) llroundf(x[m + n*c] * id) + 1;
                    q *= 3;
                    q += xi;
                }
                q = ((uint16_t)q * 256 + 242) / 243;
                y->qs[j + m] = q;
            }
            x += 5*c;
        }
    }
    for (size_t h = 0; h < sizeof(y->qh); ++h) {
        uint8_t q = 0;
        for (size_t m = 0; m < 4; ++m) {
            int xi = (int) llroundf(x[h + m*sizeof(y->qh)] * id) + 1;
            q *= 3;
            q += xi;
        }
        q *= 3;
        q = ((uint16_t)q * 256 + 242) / 243;
        y->qh[h] = q;
    }
}

static void cpu_dequant(const block_ptq1_0 * x, float * y) {
    static const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
    static const size_t  stages[3] = {32, 16, 8};
    const float d = half_to_float(x->d);
    size_t j = 0;
    for (size_t s = 0; s < 3; ++s) {
        const size_t c = stages[s];
        for (; j + c <= sizeof(x->qs); j += c)
            for (size_t n = 0; n < 5; ++n)
                for (size_t m = 0; m < c; ++m) {
                    uint8_t q = x->qs[j + m] * pow3[n];
                    int16_t xi = ((uint16_t) q * 3) >> 8;
                    *y++ = (float) (xi - 1) * d;
                }
    }
    for (size_t n = 0; n < 4; ++n)
        for (size_t h = 0; h < sizeof(x->qh); ++h) {
            uint8_t q = x->qh[h] * pow3[n];
            int16_t xi = ((uint16_t) q * 3) >> 8;
            *y++ = (float) (xi - 1) * d;
        }
}

// q8_1 quantizer, same as ggml quantize_row_q8_1_ref (for realistic activation rows)
static void quantize_row_q8_1(const float * x, block_q8_1 * y) {
    float amax = 0.0f;
    for (int i = 0; i < QK8_1; ++i) amax = fmaxf(amax, fabsf(x[i]));
    const float d  = amax / 7.0f;
    const float id = d ? 1.0f/d : 0.0f;
    float sumxs = 0.0f;
    for (int i = 0; i < QK8_1; ++i) {
        y->qs[i] = (int8_t) llroundf(x[i] * id);
        sumxs += (float) y->qs[i];
    }
    y->ds = (uint32_t) float_to_half_bits(d) | ((uint32_t) float_to_half_bits(d * sumxs) << 16);
}

// ---- test --------------------------------------------------------------------
static unsigned seed = 12345;
static uint8_t rnd_byte(void) {
    seed = seed * 1103515245u + 12345u;
    return (uint8_t)((seed >> 16) & 0xFF);
}
static float rnd_float(float lo, float hi) {
    return lo + (float)(rnd_byte() * 257 + rnd_byte()) / (257.0f * 257.0f) * (hi - lo);
}

static void rand_block(block_ptq1_0 * w, float d) {
    for (int i = 0; i < 24; ++i) w->qs[i] = rnd_byte();
    for (int i = 0; i < 2; ++i)  w->qh[i] = rnd_byte();
    w->d = float_to_half_bits(d);
}

int main(void) {
    long checks = 0, fails = 0;
    double worst_rel = 0.0;

    // T1: kernel decode vs element-wise reference, all 128 trits
    {
        int bad = 0;
        for (int t = 0; t < 5000; ++t) {
            block_ptq1_0 w; rand_block(&w, 0.25f);
            int8_t q[QK_PTQ1_0];
            kernel_decode(&w, q);
            for (int e = 0; e < QK_PTQ1_0; ++e) {
                ++checks;
                if (q[e] != ref_trit(&w, e)) { ++bad; ++fails; }
            }
        }
        // fixed patterns: constant trits, properly packed (5 trits per qs byte, 4 per qh byte)
        const uint8_t qs_byte[3] = {0x00, 0x80, 0xFF};
        const uint8_t qh_byte[3] = {0x00, 0x7F, 0xFD};
        for (int t = 0; t < 3; ++t) {
            block_ptq1_0 w;
            memset(w.qs, qs_byte[t], sizeof(w.qs));
            memset(w.qh, qh_byte[t], sizeof(w.qh));
            int8_t q[QK_PTQ1_0];
            kernel_decode(&w, q);
            for (int e = 0; e < QK_PTQ1_0; ++e) {
                ++checks;
                if (q[e] != (int8_t)(t - 1)) { ++bad; ++fails; }
            }
        }
        printf("T1: decode mapping, %d mismatches\n", bad);
    }

    // T2: round-trip through the CPU quantizer
    {
        int bad = 0;
        for (int t = 0; t < 2000; ++t) {
            float x[QK_PTQ1_0];
            for (int i = 0; i < QK_PTQ1_0; ++i) x[i] = rnd_float(-1.0f, 1.0f);
            block_ptq1_0 w;
            cpu_quantize(x, &w);
            int8_t q[QK_PTQ1_0];
            kernel_decode(&w, q);
            float ref[QK_PTQ1_0];
            cpu_dequant(&w, ref);
            const float d0 = half_to_float(w.d);
            for (int e = 0; e < QK_PTQ1_0; ++e) {
                ++checks;
                // same trit and same d conversion -> values must be bitwise equal
                if (q[e] * d0 != ref[e]) { ++bad; ++fails; }
            }
        }
        // already-ternary input must round-trip exactly
        for (int t = 0; t < 1000; ++t) {
            float x[QK_PTQ1_0];
            for (int i = 0; i < QK_PTQ1_0; ++i) {
                const int r = rnd_byte() % 3;
                x[i] = (float)(r - 1) * 0.5f; // d = 0.5, trit in {-1, 0, 1}
            }
            block_ptq1_0 w;
            cpu_quantize(x, &w);
            int8_t q[QK_PTQ1_0];
            kernel_decode(&w, q);
            for (int e = 0; e < QK_PTQ1_0; ++e) {
                ++checks;
                if (q[e] != (int8_t)((int)lroundf(x[e] / 0.5f))) { ++bad; ++fails; }
            }
        }
        printf("T2: quantizer round-trip, %d mismatches\n", bad);
    }

    // T3: exact integer dots with exactly representable scales
    {
        int bad = 0;
        const float d1s[4] = {1.0f, 2.0f, 4.0f, 8.0f};
        for (int t = 0; t < 2000; ++t) {
            block_ptq1_0 w; rand_block(&w, 1.0f); // d0 = 1.0, exact
            block_q8_1 y[4];
            for (int k = 0; k < 4; ++k) {
                memset(y[k].qs, 1, 32); // qs = 1 everywhere
                y[k].ds = float_to_half_bits(d1s[k]) | ((uint32_t) float_to_half_bits(d1s[k]) << 16);
            }
            // kernel = sum_k d1_k * sumi_k, every term exact in float
            const float got = vec_dot_ptq1_0_q8_1(&w, y);
            int32_t refi = 0;
            for (int k = 0; k < 4; ++k)
                for (int b = 0; b < 32; ++b)
                    refi += (int32_t) d1s[k] * ref_trit(&w, k * 32 + b);
            ++checks;
            if (got != (float) refi) {
                if (bad < 4) printf("T3[%d]: got=%.1f ref=%d\n", t, got, (int) refi);
                ++bad; ++fails;
            }
        }
        // all-zero trits (0x80/0x7F packing; 0x00 would decode to all -1)
        {
            block_ptq1_0 w;
            memset(w.qs, 0x80, sizeof(w.qs));
            memset(w.qh, 0x7F, sizeof(w.qh));
            w.d = float_to_half_bits(1.0f);
            block_q8_1 y[4];
            for (int k = 0; k < 4; ++k) { memset(y[k].qs, 5, 32); y[k].ds = float_to_half_bits(0.5f); }
            const float got = vec_dot_ptq1_0_q8_1(&w, y);
            ++checks;
            if (got != 0.0f) { ++bad; ++fails; }
        }
        printf("T3: exact-scale integer dots, %d mismatches\n", bad);
    }

    // T4: random data, float dot vs double reference over dequantized values
    {
        for (int t = 0; t < 2000; ++t) {
            block_ptq1_0 w; rand_block(&w, rnd_float(0.01f, 2.0f));
            block_q8_1 y[4];
            for (int k = 0; k < 4; ++k) {
                float x[QK8_1];
                for (int i = 0; i < QK8_1; ++i) x[i] = rnd_float(-1.0f, 1.0f);
                quantize_row_q8_1(x, &y[k]);
            }
            const float got = vec_dot_ptq1_0_q8_1(&w, y);
            const float d0 = half_to_float(w.d);
            double ref = 0.0, sumabs = 0.0;
            for (int k = 0; k < 4; ++k) {
                const double d1 = half_to_float(y[k].ds & 0xFFFF);
                for (int b = 0; b < QK8_1; ++b) {
                    const double term = (double) ref_trit(&w, k * 32 + b) * d0 * (double) y[k].qs[b] * d1;
                    ref += term;
                    sumabs += fabs(term);
                }
            }
            // error relative to sum |terms|, immune to cancellation
            const double rel = fabs(got - ref) / (sumabs > 1e-12 ? sumabs : 1.0);
            if (rel > worst_rel) worst_rel = rel;
            if (rel > 1e-6) {
                if (fails < 5) printf("T4[%d]: got=%.6f ref=%.6f sumabs=%.6f rel=%.3e\n", t, got, ref, sumabs, rel);
                ++fails;
            }
            ++checks;
        }
        printf("T4: random float dots, worst_err_sumabs=%.3e\n", worst_rel);
    }

    // T5: full-row emulation of the MMVQ split, 4 blocks of 128, one kernel call per block
    {
        int bad = 0;
        for (int t = 0; t < 1000; ++t) {
            float x[4 * QK_PTQ1_0];
            for (int i = 0; i < 4 * QK_PTQ1_0; ++i) x[i] = rnd_float(-1.0f, 1.0f);
            block_ptq1_0 w[4];
            for (int b = 0; b < 4; ++b) cpu_quantize(x + b * QK_PTQ1_0, &w[b]);
            block_q8_1 y[16];
            for (int k = 0; k < 16; ++k) {
                float xk[QK8_1];
                for (int i = 0; i < QK8_1; ++i) xk[i] = rnd_float(-1.0f, 1.0f);
                quantize_row_q8_1(xk, &y[k]);
            }
            // template sum: each thread takes one full block, qi/vdr = 1
            float got = 0.0f;
            for (int b = 0; b < 4; ++b) got += vec_dot_ptq1_0_q8_1(&w[b], &y[b * 4]);
            double ref = 0.0, sumabs = 0.0;
            for (int b = 0; b < 4; ++b)
                for (int k = 0; k < 4; ++k) {
                    const double d1 = half_to_float(y[b * 4 + k].ds & 0xFFFF);
                    for (int i = 0; i < QK8_1; ++i) {
                        const double term = (double) ref_trit(&w[b], k * 32 + i) * half_to_float(w[b].d)
                                           * (double) y[b * 4 + k].qs[i] * d1;
                        ref += term;
                        sumabs += fabs(term);
                    }
                }
            const double rel = fabs(got - ref) / (sumabs > 1e-12 ? sumabs : 1.0);
            ++checks;
            if (rel > 1e-6) {
                if (bad < 4) printf("T5[%d]: got=%.6f ref=%.6f sumabs=%.6f rel=%.3e\n", t, got, ref, sumabs, rel);
                ++bad; ++fails;
            }
        }
        printf("T5: full-row block split, %d mismatches\n", bad);
    }

    printf("\n%ld checks, %ld fails\n", checks, fails);
    return fails != 0;
}
