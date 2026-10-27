// tests for llama_context_resize (runtime KV cache grow/shrink):
//
//  (a) a grow preserves the cached K/V: decode a prefix on a small context, grow it,
//      keep decoding, and compare the final logits against a single-pass reference
//      decode on a context created at the grown size
//  (b) an empty cache can be grown; the size is rounded up to a multiple of 256
//  (c) resizing to the current size is a no-op; an empty cache can be shrunk
//  (d) a shrink that would drop used cells fails and leaves the cache byte-exact:
//      the continuation must match a control continuation without the resize call
//      (guards the cells-first ordering: the rows of the dropped cells must not be
//      destroyed by the failed attempt)
//  (e) a shrink that only drops empty cells succeeds and the kept rows survive the
//      buffer copy byte-exact
//
// All cases run on the plain kv cache path (full attention), the one the server
// grow uses. (a) compares two different ubatch splits and allows a small fp
// difference; (d) and (e) use a same-split control, so the resize is the only
// difference between the two paths.

#include "llama.h"
#include "arg.h"
#include "common.h"
#include "log.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#define TEST_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "%s:%d: assertion failed: %s\n", __FILE__, __LINE__, #cond); \
            abort(); \
        } \
    } while (0)

// a fixed prompt tiled to n_tokens, so every case decodes the same sequence
static std::vector<llama_token> make_tokens(const llama_model * model, int n_tokens) {
    const std::string prompt = "The quick brown fox jumps over the lazy dog.";
    const auto * vocab = llama_model_get_vocab(model);
    const auto tok = common_tokenize(vocab, prompt, /*add_special=*/ true);
    TEST_ASSERT(!tok.empty());

    std::vector<llama_token> tokens;
    tokens.reserve(n_tokens);
    for (int i = 0; i < n_tokens; i++) {
        tokens.push_back(tok[i % (int) tok.size()]);
    }

    return tokens;
}

// decode tokens[start..end) at absolute positions start..end-1, return the
// logits of the last position of the batch
static std::vector<float> decode_range(llama_context * ctx, const std::vector<llama_token> & tokens, size_t start, size_t end) {
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));

    llama_batch batch = llama_batch_init((int32_t) (end - start), 0, 1);
    for (size_t i = start; i < end; i++) {
        common_batch_add(batch, tokens[i], (llama_pos) i, { 0 }, true);
    }

    if (llama_decode(ctx, batch)) {
        llama_batch_free(batch);
        throw std::runtime_error("llama_decode failed");
    }

    const float * logits = llama_get_logits_ith(ctx, (int32_t) batch.n_tokens - 1);

    std::vector<float> ret(logits, logits + n_vocab);

    llama_batch_free(batch);

    return ret;
}

static int32_t argmax(const std::vector<float> & v) {
    size_t i = 0;
    for (size_t j = 1; j < v.size(); j++) {
        if (v[j] > v[i]) {
            i = j;
        }
    }
    return (int32_t) i;
}

// the resized context must agree with a context that was created at the target
// size from the start; the rows are copied byte-exact, so only kernel-level fp
// differences between the two decode paths are allowed
static void assert_same_output(const char * what, const std::vector<float> & a, const std::vector<float> & b, float tol) {
    TEST_ASSERT(a.size() == b.size());

    float max_diff = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
    }

    LOG_INF("%s: max |logit diff| = %.3g, argmax %d vs %d\n", what, max_diff, argmax(a), argmax(b));

    TEST_ASSERT(max_diff < tol);
    TEST_ASSERT(argmax(a) == argmax(b));
}

static void test_grow_preserves_kv(llama_model * model) {
    common_params params;

    auto cparams = common_context_params_to_llama(params);
    cparams.n_batch   = 1024;
    cparams.n_ubatch  = 128;
    cparams.n_seq_max = 1;
    cparams.n_ctx     = 512;

    const std::vector<llama_token> tokens = make_tokens(model, 544);

    // two decodes: 400 on the small context, then a grow and 144 more
    llama_context_ptr ctx(llama_init_from_model(model, cparams));
    TEST_ASSERT(ctx != nullptr);

    decode_range(ctx.get(), tokens, 0, 400);

    TEST_ASSERT(llama_context_resize(ctx.get(), 1536));
    TEST_ASSERT(llama_n_ctx(ctx.get()) == 1536);

    std::vector<float> logits_grown = decode_range(ctx.get(), tokens, 400, 544);

    // reference: same tokens in a single pass on a context that started at 1536
    cparams.n_ctx = 1536;
    llama_context_ptr ref(llama_init_from_model(model, cparams));
    TEST_ASSERT(ref != nullptr);
    std::vector<float> logits_ref = decode_range(ref.get(), tokens, 0, 544);

    assert_same_output("grow preserves kv", logits_grown, logits_ref, 1e-4f);
}

static void test_grow_empty(llama_model * model) {
    common_params params;

    auto cparams = common_context_params_to_llama(params);
    cparams.n_batch   = 1024;
    cparams.n_ubatch  = 128;
    cparams.n_seq_max = 1;
    cparams.n_ctx     = 512;

    const std::vector<llama_token> tokens = make_tokens(model, 300);

    llama_context_ptr ctx(llama_init_from_model(model, cparams));
    TEST_ASSERT(ctx != nullptr);

    // 1500 is rounded up to the next multiple of 256
    TEST_ASSERT(llama_context_resize(ctx.get(), 1500));
    TEST_ASSERT(llama_n_ctx(ctx.get()) == 1536);

    decode_range(ctx.get(), tokens, 0, 300); // the grown empty cache is usable
}

static void test_noop_and_shrink_empty(llama_model * model) {
    common_params params;

    auto cparams = common_context_params_to_llama(params);
    cparams.n_batch   = 1024;
    cparams.n_ubatch  = 128;
    cparams.n_seq_max = 1;
    cparams.n_ctx     = 512;

    llama_context_ptr ctx(llama_init_from_model(model, cparams));
    TEST_ASSERT(ctx != nullptr);

    TEST_ASSERT(llama_context_resize(ctx.get(), 512)); // same size: no-op
    TEST_ASSERT(llama_n_ctx(ctx.get()) == 512);

    TEST_ASSERT(llama_context_resize(ctx.get(), 256)); // all cells empty: ok
    TEST_ASSERT(llama_n_ctx(ctx.get()) == 256);
}

static void test_shrink_used_fails(llama_model * model) {
    common_params params;

    auto cparams = common_context_params_to_llama(params);
    cparams.n_batch   = 1024;
    cparams.n_ubatch  = 128;
    cparams.n_seq_max = 1;
    cparams.n_ctx     = 1024;

    const std::vector<llama_token> tokens = make_tokens(model, 900);

    // the failed shrink must not touch the cached rows of the kept cells
    llama_context_ptr ctx(llama_init_from_model(model, cparams));
    TEST_ASSERT(ctx != nullptr);

    decode_range(ctx.get(), tokens, 0, 800);

    TEST_ASSERT(!llama_context_resize(ctx.get(), 512)); // cells 512..799 are used
    TEST_ASSERT(llama_n_ctx(ctx.get()) == 1024);

    std::vector<float> logits_shrunk = decode_range(ctx.get(), tokens, 800, 900);

    // control: the same continuation without the resize call; the ubatch split
    // matches, so the only difference between the two is the failed resize
    cparams.n_ctx = 1024;
    llama_context_ptr cont(llama_init_from_model(model, cparams));
    TEST_ASSERT(cont != nullptr);
    decode_range(cont.get(), tokens, 0, 800);
    std::vector<float> logits_cont = decode_range(cont.get(), tokens, 800, 900);

    assert_same_output("failed shrink keeps kv intact", logits_shrunk, logits_cont, 1e-4f);
}

static void test_shrink_empty_tail(llama_model * model) {
    common_params params;

    auto cparams = common_context_params_to_llama(params);
    cparams.n_batch   = 1024;
    cparams.n_ubatch  = 128;
    cparams.n_seq_max = 1;
    cparams.n_ctx     = 1024;

    const std::vector<llama_token> tokens = make_tokens(model, 470);

    llama_context_ptr ctx(llama_init_from_model(model, cparams));
    TEST_ASSERT(ctx != nullptr);

    decode_range(ctx.get(), tokens, 0, 320);

    // only empty cells are dropped: the shrink must succeed
    TEST_ASSERT(llama_context_resize(ctx.get(), 512));
    TEST_ASSERT(llama_n_ctx(ctx.get()) == 512);

    std::vector<float> logits_shrunk = decode_range(ctx.get(), tokens, 320, 470);

    // control: the same continuation on a context that was never resized; the
    // rows 0..319 must survive the shrink byte-exact
    cparams.n_ctx = 1024;
    llama_context_ptr cont(llama_init_from_model(model, cparams));
    TEST_ASSERT(cont != nullptr);
    decode_range(cont.get(), tokens, 0, 320);
    std::vector<float> logits_cont = decode_range(cont.get(), tokens, 320, 470);

    assert_same_output("shrink of empty tail", logits_shrunk, logits_cont, 1e-4f);
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    auto mparams = common_model_params_to_llama(params);

    llama_model_ptr model(llama_model_load_from_file(params.model.path.c_str(), mparams));
    if (!model) {
        LOG_ERR("%s: failed to load model '%s'\n", __func__, params.model.path.c_str());
        return 1;
    }

    test_grow_preserves_kv(model.get());
    test_grow_empty(model.get());
    test_noop_and_shrink_empty(model.get());
    test_shrink_used_fails(model.get());
    test_shrink_empty_tail(model.get());

    LOG_INF("%s: OK\n", __func__);

    return 0;
}
