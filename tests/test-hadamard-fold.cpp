// Pins the load-time hadamard fold used in tensor split mode for ternary
// (PQ2_0/PTQ1_0) weights: W'' = W * H * diag(s), with an extra grouped->
// tiled scatter for ssm_out. H is the normalized block-diagonal Sylvester
// (Walsh-Hadamard) matrix. The production fold runs this per weight column
// in llama_model_base::fold_hadamard_weight (src/llama-model.cpp); the math
// components are transcribed here and checked against an independent
// dense-matrix reference, so a regression in the formula (e.g. a stray
// permutation, swapped FWHT/signs order, wrong scatter direction) fails.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cstring>

static int fails = 0;

static void report(bool ok, const char * what) {
    printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) {
        ++fails;
    }
}

static uint32_t popc(uint32_t x) {
    uint32_t c = 0;
    while (x) { x &= x - 1; ++c; }
    return c;
}

// --- transcribed from fold_hadamard_weight (src/llama-model.cpp) ---
static void fwht_block(float * v, uint32_t n) {
    for (uint32_t h = n / 2; h > 0; h >>= 1) {
        for (uint32_t i = 0; i < n; i += 2 * h) {
            for (uint32_t j = i; j < i + h; ++j) {
                const float a = v[j];
                const float b = v[j + h];
                v[j]     = a + b;
                v[j + h] = a - b;
            }
        }
    }
    const float scale = 1.0f / sqrtf((float) n);
    for (uint32_t i = 0; i < n; ++i) {
        v[i] *= scale;
    }
}

// one fold step on a column: FWHT per block, signs, optional scatter
static void fold_column(const float * w, float * out, int64_t n_feat, uint32_t block_size,
                        const int8_t * signs, const int32_t * perm_g) {
    std::vector<float> col(w, w + n_feat);
    for (int64_t b0 = 0; b0 < n_feat; b0 += block_size) {
        fwht_block(col.data() + b0, block_size);
    }
    if (signs) {
        for (int64_t k = 0; k < n_feat; ++k) {
            col[k] *= (float) signs[k];
        }
    }
    if (perm_g) {
        for (int64_t r = 0; r < n_feat; ++r) {
            out[r] = col[perm_g[r]];
        }
    } else {
        std::memcpy(out, col.data(), (size_t) n_feat * sizeof(float));
    }
}

// grouped->tiled index map for ssm_out, transcribed from the same function
static std::vector<int32_t> make_perm_g(int64_t n_feat, int64_t n_v, int64_t n_k) {
    const int64_t n_rep = n_v / n_k;
    const int64_t n_hd  = n_feat / n_v;
    std::vector<int32_t> perm_g((size_t) n_feat);
    for (int64_t r = 0; r < n_feat; ++r) {
        const int64_t h = r % n_hd;
        const int64_t n = (r / n_hd) % n_k;
        const int64_t v = r / (n_hd * n_k);
        perm_g[r] = (int32_t) (h + n_hd * v + n_hd * n_rep * n);
    }
    return perm_g;
}

// --- independent dense reference: H[m][j] = (-1)^popcount(m & j) / sqrt(n) ---
static void dense_sylvester_row(const float * w, float * out, uint32_t n) {
    const double scale = 1.0 / sqrt((double) n);
    for (uint32_t m = 0; m < n; ++m) {
        double acc = 0.0;
        for (uint32_t j = 0; j < n; ++j) {
            const double sgn = (popc(m & j) & 1) ? -1.0 : 1.0;
            acc += sgn * (double) w[j];
        }
        out[m] = (float) (acc * scale);
    }
}

// reference fold step computed with dense matrices: (W @ H_block) * s, scatter
static void fold_column_ref(const float * w, float * out, int64_t n_feat, uint32_t block_size,
                            const int8_t * signs, const int32_t * perm_g) {
    std::vector<float> tmp((size_t) n_feat);
    for (int64_t b0 = 0; b0 < n_feat; b0 += block_size) {
        dense_sylvester_row(w + b0, tmp.data() + b0, block_size);
    }
    for (int64_t k = 0; k < n_feat; ++k) {
        tmp[k] *= signs ? (float) signs[k] : 1.0f;
    }
    if (perm_g) {
        for (int64_t r = 0; r < n_feat; ++r) {
            out[r] = tmp[perm_g[r]];
        }
    } else {
        std::memcpy(out, tmp.data(), (size_t) n_feat * sizeof(float));
    }
}

static uint32_t rng_state = 0x12345678u;
static uint32_t rng(void) {
    rng_state = rng_state * 1103515245u + 12345u;
    return (rng_state >> 8) & 0xFFFF;
}
static float frand(void) {
    return ((float) rng() / 32768.0f) - 1.0f;
}

static double max_diff(const float * a, const float * b, int64_t n) {
    double d = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double e = std::fabs((double) a[i] - (double) b[i]);
        if (e > d) {
            d = e;
        }
    }
    return d;
}

// check 1: the in-place butterfly is the normalized Sylvester matrix
static void test_fwht_is_sylvester(void) {
    const uint32_t sizes[4] = {4, 8, 16, 1024};
    for (uint32_t si = 0; si < 4; ++si) {
        const uint32_t n = sizes[si];
        const uint32_t n_vec = (n <= 16) ? n : 32; // basis vectors: full column coverage
        for (uint32_t t = 0; t < n_vec; ++t) {
            std::vector<float> v(n, 0.0f), got(n), ref(n);
            if (n <= 16) {
                v[t] = 1.0f;
            } else {
                for (uint32_t i = 0; i < n; ++i) {
                    v[i] = frand();
                }
            }
            std::vector<float> wv = v;
            fwht_block(wv.data(), n);
            dense_sylvester_row(v.data(), ref.data(), n);
            std::swap(got, wv);
            const double d = max_diff(got.data(), ref.data(), (int64_t) n);
            if (d > (n <= 16 ? 1e-5 : 1e-3)) {
                report(false, "fwht matches dense sylvester (see above)");
                return;
            }
        }
        char what[64];
        snprintf(what, sizeof(what), "fwht butterfly == dense sylvester, block n=%u", n);
        report(true, what);
    }
}

// check 2: the fold pipeline (FWHT then signs) == W @ H @ diag(s), small case
static void test_pipeline_no_perm(void) {
    const int64_t K = 8, N = 3;
    const uint32_t block = 4;
    std::vector<float> W((size_t) N * (size_t) K), got((size_t) K), ref((size_t) K);
    std::vector<int8_t> signs((size_t) K);
    for (auto & x : W) {
        x = frand();
    }
    for (auto & s : signs) {
        s = (rng() & 1) ? (int8_t) 1 : (int8_t) -1;
    }
    double worst = 0.0;
    for (int64_t r = 0; r < N; ++r) {
        fold_column(W.data() + r * K, got.data(), K, block, signs.data(), nullptr);
        fold_column_ref(W.data() + r * K, ref.data(), K, block, signs.data(), nullptr);
        const double d = max_diff(got.data(), ref.data(), K);
        if (d > worst) {
            worst = d;
        }
    }
    printf("  pipeline (no perm) worst |fold - W*H*diag(s)| = %.3e\n", worst);
    report(worst < 1e-5, "pipeline (no perm) == W @ H @ diag(s)");
}

// check 3: the ssm_out grouped->tiled map, real 6144/48/16 geometry
static void test_perm(void) {
    const int64_t n_feat = 6144; // 128 heads x 48 time_step_rank
    const int64_t n_v = 48, n_k = 16; // n_rep = 3, n_hd = 128
    const std::vector<int32_t> perm_g = make_perm_g(n_feat, n_v, n_k);
    std::vector<int32_t> seen((size_t) n_feat, 0);
    bool bijection = true;
    for (int64_t r = 0; r < n_feat; ++r) {
        const int32_t l = perm_g[r];
        if (l < 0 || l >= n_feat || ++seen[l] > 1) {
            bijection = false;
            break;
        }
    }
    report(bijection, "perm_g is a bijection on 6144");
    report(perm_g[128] == 384 && perm_g[384] == 1152 && perm_g[2048] == 128,
           "perm_g spot values (128->384, 384->1152, 2048->128)");
    report(perm_g[perm_g[128]] != 128, "perm_g is NOT self-inverse (scatter, not gather)");
    // block alignment facts: why ffn_down/attn_output need the fold at all
    printf("  2-device per-block extents: ssm_out 6144/2 = %lld (%s), "
           "ffn_down 17408/2 = %lld (block rem %lld), attn_output 5120/2 = %lld (block rem %lld)\n",
           (long long) (6144 / 2), (6144 / 2) % 1024 == 0 ? "aligned" : "straddling",
           (long long) (17408 / 2), (long long) (17408 / 2 % 1024),
           (long long) (5120 / 2),  (long long) (5120 / 2 % 1024));
}

// check 4: full ssm_out pipeline at real scale == dense reference
static void test_pipeline_with_perm(void) {
    const int64_t K = 6144, N = 2;
    const uint32_t block = 1024;
    const std::vector<int32_t> perm_g = make_perm_g(K, 48, 16);
    std::vector<float> W((size_t) N * (size_t) K), got((size_t) K), ref((size_t) K);
    std::vector<int8_t> signs((size_t) K);
    for (auto & x : W) {
        x = frand();
    }
    for (auto & s : signs) {
        s = (rng() & 1) ? (int8_t) 1 : (int8_t) -1;
    }
    double worst = 0.0;
    for (int64_t r = 0; r < N; ++r) {
        fold_column(W.data() + r * K, got.data(), K, block, signs.data(), perm_g.data());
        fold_column_ref(W.data() + r * K, ref.data(), K, block, signs.data(), perm_g.data());
        const double d = max_diff(got.data(), ref.data(), K);
        if (d > worst) {
            worst = d;
        }
    }
    printf("  pipeline (ssm_out, 6144) worst |fold - W*H*diag(s)*P| = %.3e\n", worst);
    report(worst < 1e-3, "pipeline (ssm_out) == W @ H @ diag(s) @ P, real 6144/1024 geometry");
}

int main(void) {
    printf("hadamard fold formula (tensor-mode ternary weights):\n");
    test_fwht_is_sylvester();
    test_pipeline_no_perm();
    test_perm();
    test_pipeline_with_perm();
    printf("  %s\n", fails == 0 ? "FOLD FORMULA OK" : "FOLD FORMULA BROKEN");
    return fails != 0;
}
