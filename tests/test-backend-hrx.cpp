#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <ggml-hrx.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

static ggml_context_ptr make_context() {
    ggml_init_params params = {
        /* .mem_size   = */ 256 * ggml_tensor_overhead() + ggml_graph_overhead_custom(96, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    return ggml_context_ptr(ggml_init(params));
}

static void expect_eq(const std::vector<float> & actual, const std::vector<float> & expected, const char * label) {
    GGML_ASSERT(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::fprintf(stderr, "%s[%zu]: got %.9g expected %.9g\n", label, i, actual[i], expected[i]);
            std::abort();
        }
    }
}

static void expect_eq_i32(const std::vector<int32_t> & actual,
                          const std::vector<int32_t> & expected,
                          const char *                 label) {
    GGML_ASSERT(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::fprintf(stderr, "%s[%zu]: got %d expected %d\n", label, i, actual[i], expected[i]);
            std::abort();
        }
    }
}

static void expect_near(const std::vector<float> & actual,
                        const std::vector<float> & expected,
                        float                      tolerance,
                        const char *               label) {
    GGML_ASSERT(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(actual[i] - expected[i]) > tolerance) {
            std::fprintf(stderr, "%s[%zu]: got %.9g expected %.9g tolerance %.9g\n", label, i, actual[i], expected[i],
                         tolerance);
            std::abort();
        }
    }
}

static bool run_scale_support_case(ggml_backend_dev_t dev) {
    bool scale_f32_supported = false;
    {
        ggml_context_ptr ctx = make_context();
        ggml_tensor *    src = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 16);
        ggml_tensor *    out = ggml_scale_bias(ctx.get(), src, 1.25f, -0.75f);
        scale_f32_supported  = ggml_backend_dev_supports_op(dev, out);
    }

    {
        ggml_context_ptr ctx = make_context();
        ggml_tensor *    src = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F16, 16);
        ggml_tensor *    out = ggml_scale_bias(ctx.get(), src, 1.25f, -0.75f);
        GGML_ASSERT(!ggml_backend_dev_supports_op(dev, out));
    }
    return scale_f32_supported;
}

static bool run_clamp_support_case(ggml_backend_dev_t dev) {
    bool clamp_f32_supported = false;
    {
        ggml_context_ptr ctx = make_context();
        ggml_tensor *    src = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 16);
        ggml_tensor *    out = ggml_clamp(ctx.get(), src, -1.5f, 2.0f);
        clamp_f32_supported  = ggml_backend_dev_supports_op(dev, out);
    }

    {
        ggml_context_ptr ctx = make_context();
        ggml_tensor *    src = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F16, 16);
        ggml_tensor *    out = ggml_clamp(ctx.get(), src, -1.5f, 2.0f);
        GGML_ASSERT(!ggml_backend_dev_supports_op(dev, out));
    }
    return clamp_f32_supported;
}

static bool run_add_support_case(ggml_backend_dev_t dev) {
    bool add_f32_supported = false;
    {
        ggml_context_ptr ctx = make_context();
        ggml_tensor *    lhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 16);
        ggml_tensor *    rhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 16);
        ggml_tensor *    out = ggml_add(ctx.get(), lhs, rhs);
        add_f32_supported    = ggml_backend_dev_supports_op(dev, out);
    }

    {
        ggml_context_ptr ctx = make_context();
        ggml_tensor *    lhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_BF16, 16);
        ggml_tensor *    rhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_BF16, 16);
        ggml_tensor *    out = ggml_add(ctx.get(), lhs, rhs);
        GGML_ASSERT(!ggml_backend_dev_supports_op(dev, out));
    }
    return add_f32_supported;
}

static bool run_mul_support_case(ggml_backend_dev_t dev) {
    bool mul_f32_supported = false;
    {
        ggml_context_ptr ctx = make_context();
        ggml_tensor *    lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 128, 32);
        ggml_tensor *    rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 128, 1);
        ggml_tensor *    out = ggml_mul(ctx.get(), lhs, rhs);
        mul_f32_supported    = ggml_backend_dev_supports_op(dev, out);
    }

    {
        ggml_context_ptr ctx = make_context();
        ggml_tensor *    lhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_BF16, 16);
        ggml_tensor *    rhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_BF16, 16);
        ggml_tensor *    out = ggml_mul(ctx.get(), lhs, rhs);
        GGML_ASSERT(!ggml_backend_dev_supports_op(dev, out));
    }

    return mul_f32_supported;
}

static bool run_div_support_case(ggml_backend_dev_t dev) {
    bool div_f32_supported = false;
    {
        ggml_context_ptr ctx = make_context();
        ggml_tensor *    lhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 8);
        ggml_tensor *    rhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        ggml_tensor *    out = ggml_div(ctx.get(), lhs, rhs);
        div_f32_supported    = ggml_backend_dev_supports_op(dev, out);
    }

    {
        ggml_context_ptr ctx = make_context();
        ggml_tensor *    lhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_BF16, 16);
        ggml_tensor *    rhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_BF16, 1);
        ggml_tensor *    out = ggml_div(ctx.get(), lhs, rhs);
        GGML_ASSERT(!ggml_backend_dev_supports_op(dev, out));
    }
    return div_f32_supported;
}

static bool run_sum_rows_support_case(ggml_backend_dev_t dev) {
    bool sum_rows_f32_supported = false;
    {
        ggml_context_ptr ctx   = make_context();
        ggml_tensor *    src   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 8, 1);
        ggml_tensor *    out   = ggml_sum_rows(ctx.get(), src);
        sum_rows_f32_supported = ggml_backend_dev_supports_op(dev, out);
    }

    {
        ggml_context_ptr ctx = make_context();
        ggml_tensor *    src = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F16, 8);
        ggml_tensor *    out = ggml_sum_rows(ctx.get(), src);
        GGML_ASSERT(!ggml_backend_dev_supports_op(dev, out));
    }
    return sum_rows_f32_supported;
}

static bool run_soft_max_support_case(ggml_backend_dev_t dev) {
    bool soft_max_f32_supported = false;
    {
        ggml_context_ptr ctx   = make_context();
        ggml_tensor *    src   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 128, 1);
        ggml_tensor *    out   = ggml_soft_max(ctx.get(), src);
        soft_max_f32_supported = ggml_backend_dev_supports_op(dev, out);
    }

    {
        ggml_context_ptr ctx = make_context();
        ggml_tensor *    src = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1025);
        ggml_tensor *    out = ggml_soft_max(ctx.get(), src);
        GGML_ASSERT(!ggml_backend_dev_supports_op(dev, out));
    }
    return soft_max_f32_supported;
}

static bool run_argsort_support_case(ggml_backend_dev_t dev) {
    bool argsort_f32_i32_supported = false;
    {
        ggml_context_ptr ctx      = make_context();
        ggml_tensor *    src      = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 128, 1);
        ggml_tensor *    out      = ggml_argsort(ctx.get(), src, GGML_SORT_ORDER_DESC);
        argsort_f32_i32_supported = ggml_backend_dev_supports_op(dev, out);
    }

    {
        ggml_context_ptr ctx = make_context();
        ggml_tensor *    src = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 257);
        ggml_tensor *    out = ggml_argsort(ctx.get(), src, GGML_SORT_ORDER_DESC);
        GGML_ASSERT(!ggml_backend_dev_supports_op(dev, out));
    }
    return argsort_f32_i32_supported;
}

static bool run_swiglu_support_case(ggml_backend_dev_t dev) {
    bool swiglu_f32_supported = false;
    {
        ggml_context_ptr ctx  = make_context();
        ggml_tensor *    gate = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 33, 2, 67);
        ggml_tensor *    up   = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 33, 2, 67);
        ggml_tensor *    out  = ggml_swiglu_split(ctx.get(), gate, up);
        swiglu_f32_supported  = ggml_backend_dev_supports_op(dev, out);
    }

    {
        ggml_context_ptr ctx  = make_context();
        ggml_tensor *    gate = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, 33, 2, 67);
        ggml_tensor *    up   = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, 33, 2, 67);
        ggml_tensor *    out  = ggml_swiglu_split(ctx.get(), gate, up);
        GGML_ASSERT(!ggml_backend_dev_supports_op(dev, out));
    }
    return swiglu_f32_supported;
}

static void run_scale_case(ggml_backend_t backend, ggml_backend_dev_t dev, int64_t n) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor *    src = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    ggml_tensor *    out = ggml_scale_bias(ctx.get(), src, 1.25f, -0.75f);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> src_data(n);
    std::vector<float> expected(n);
    for (int64_t i = 0; i < n; ++i) {
        src_data[i] = static_cast<float>(i % 13) - 6.0f;
        expected[i] = src_data[i] * 1.25f - 0.75f;
    }

    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> actual(n, -1.0f);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    expect_near(actual, expected, 1e-6f, "scale");
}

static void run_clamp_case(ggml_backend_t backend, ggml_backend_dev_t dev, int64_t n) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor *    src = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    ggml_tensor *    out = ggml_clamp(ctx.get(), src, -1.5f, 2.0f);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> src_data(n);
    std::vector<float> expected(n);
    for (int64_t i = 0; i < n; ++i) {
        src_data[i] = static_cast<float>(i % 17) - 8.0f;
        expected[i] = src_data[i] < -1.5f ? -1.5f : (src_data[i] > 2.0f ? 2.0f : src_data[i]);
    }

    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> actual(n, -1.0f);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    expect_near(actual, expected, 1e-6f, "clamp");
}

static void run_add_case(ggml_backend_t backend, ggml_backend_dev_t dev, int64_t n) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor *    lhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    ggml_tensor *    rhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    ggml_tensor *    out = ggml_add(ctx.get(), lhs, rhs);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_data(n);
    std::vector<float> rhs_data(n);
    std::vector<float> expected(n);
    for (int64_t i = 0; i < n; ++i) {
        lhs_data[i] = static_cast<float>(i % 19) - 9.0f;
        rhs_data[i] = static_cast<float>(i % 7) * 0.5f - 1.0f;
        expected[i] = lhs_data[i] + rhs_data[i];
    }

    ggml_backend_tensor_set(lhs, lhs_data.data(), 0, lhs_data.size() * sizeof(float));
    ggml_backend_tensor_set(rhs, rhs_data.data(), 0, rhs_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> actual(n, -1.0f);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    expect_near(actual, expected, 1e-6f, "add");
}

static void run_add_transient_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    const char *      previous = std::getenv("GGML_HRX_LOOM_FORCE_ROUTE");
    const std::string saved    = previous ? previous : "";
    setenv("GGML_HRX_LOOM_FORCE_ROUTE", "add_f32_transient_demo", 1);

    run_add_case(backend, dev, 257);

    if (previous) {
        setenv("GGML_HRX_LOOM_FORCE_ROUTE", saved.c_str(), 1);
    } else {
        unsetenv("GGML_HRX_LOOM_FORCE_ROUTE");
    }
}

static void run_add_transient_pair_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    const char *      previous = std::getenv("GGML_HRX_LOOM_FORCE_ROUTE");
    const std::string saved    = previous ? previous : "";
    setenv("GGML_HRX_LOOM_FORCE_ROUTE", "add_f32_transient_demo", 1);

    const int64_t    n    = 257;
    ggml_context_ptr ctx  = make_context();
    ggml_tensor *    lhs0 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    ggml_tensor *    rhs0 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    ggml_tensor *    out0 = ggml_add(ctx.get(), lhs0, rhs0);
    ggml_tensor *    lhs1 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    ggml_tensor *    rhs1 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    ggml_tensor *    out1 = ggml_add(ctx.get(), lhs1, rhs1);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out0));
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out1));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out0);
    ggml_build_forward_expand(graph, out1);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs0_data(n);
    std::vector<float> rhs0_data(n);
    std::vector<float> lhs1_data(n);
    std::vector<float> rhs1_data(n);
    std::vector<float> expected0(n);
    std::vector<float> expected1(n);
    for (int64_t i = 0; i < n; ++i) {
        lhs0_data[i] = static_cast<float>(i % 19) - 9.0f;
        rhs0_data[i] = static_cast<float>(i % 7) * 0.5f - 1.0f;
        lhs1_data[i] = static_cast<float>(i % 23) * 0.25f - 2.0f;
        rhs1_data[i] = static_cast<float>(i % 11) - 5.0f;
        expected0[i] = lhs0_data[i] + rhs0_data[i];
        expected1[i] = lhs1_data[i] + rhs1_data[i];
    }

    ggml_backend_tensor_set(lhs0, lhs0_data.data(), 0, lhs0_data.size() * sizeof(float));
    ggml_backend_tensor_set(rhs0, rhs0_data.data(), 0, rhs0_data.size() * sizeof(float));
    ggml_backend_tensor_set(lhs1, lhs1_data.data(), 0, lhs1_data.size() * sizeof(float));
    ggml_backend_tensor_set(rhs1, rhs1_data.data(), 0, rhs1_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> actual0(n, -1.0f);
    std::vector<float> actual1(n, -1.0f);
    ggml_backend_tensor_get(out0, actual0.data(), 0, actual0.size() * sizeof(float));
    ggml_backend_tensor_get(out1, actual1.data(), 0, actual1.size() * sizeof(float));
    expect_near(actual0, expected0, 1e-6f, "add_transient_pair_0");
    expect_near(actual1, expected1, 1e-6f, "add_transient_pair_1");

    if (previous) {
        setenv("GGML_HRX_LOOM_FORCE_ROUTE", saved.c_str(), 1);
    } else {
        unsetenv("GGML_HRX_LOOM_FORCE_ROUTE");
    }
}

static void run_graph_transient_case(ggml_backend_t backend) {
    const char *      previous = std::getenv("GGML_HRX_LOOM_FORCE_ROUTE");
    const std::string saved    = previous ? previous : "";
    unsetenv("GGML_HRX_LOOM_FORCE_ROUTE");

    const int64_t    n     = 509;
    ggml_context_ptr ctx   = make_context();
    ggml_tensor *    x     = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    ggml_tensor *    scale = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    ggml_tensor *    tmp   = ggml_mul(ctx.get(), x, scale);
    ggml_tensor *    bias  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    ggml_tensor *    out   = ggml_add(ctx.get(), tmp, bias);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> x_data(n);
    std::vector<float> scale_data(n);
    std::vector<float> bias_data(n);
    std::vector<float> expected(n);
    for (int64_t i = 0; i < n; ++i) {
        x_data[i]     = static_cast<float>(i % 17) - 8.0f;
        scale_data[i] = static_cast<float>(i % 5) * 0.25f + 0.5f;
        bias_data[i]  = static_cast<float>(i % 11) - 5.0f;
        expected[i]   = x_data[i] * scale_data[i] + bias_data[i];
    }

    ggml_backend_tensor_set(x, x_data.data(), 0, x_data.size() * sizeof(float));
    ggml_backend_tensor_set(scale, scale_data.data(), 0, scale_data.size() * sizeof(float));
    ggml_backend_tensor_set(bias, bias_data.data(), 0, bias_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> actual(n, -1.0f);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    expect_near(actual, expected, 1e-6f, "graph_transient");

    if (previous) {
        setenv("GGML_HRX_LOOM_FORCE_ROUTE", saved.c_str(), 1);
    }
}

static void run_mul_row_broadcast_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    const int64_t    ncols = 128;
    const int64_t    nrows = 32;
    ggml_context_ptr ctx   = make_context();
    ggml_tensor *    lhs   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ncols, nrows);
    ggml_tensor *    rhs   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ncols, 1);
    ggml_tensor *    out   = ggml_mul(ctx.get(), lhs, rhs);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_data(ncols * nrows);
    std::vector<float> rhs_data(ncols);
    std::vector<float> expected(ncols * nrows);
    for (int64_t i = 0; i < ncols * nrows; ++i) {
        lhs_data[i] = static_cast<float>(i % 23) - 11.0f;
        expected[i] = lhs_data[i] * (static_cast<float>(i % ncols) * 0.25f - 3.0f);
    }
    for (int64_t i = 0; i < ncols; ++i) {
        rhs_data[i] = static_cast<float>(i) * 0.25f - 3.0f;
    }

    ggml_backend_tensor_set(lhs, lhs_data.data(), 0, lhs_data.size() * sizeof(float));
    ggml_backend_tensor_set(rhs, rhs_data.data(), 0, rhs_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> actual(expected.size(), -1.0f);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    expect_near(actual, expected, 1e-6f, "mul");
}

static void run_rms_norm_mul_case(ggml_backend_t backend) {
    const float   eps          = 1e-6f;
    const int64_t ncols        = 64;
    const int64_t n1           = 3;
    const int64_t n2           = 2;
    const int64_t n3           = 2;
    const int64_t ne[4]        = { ncols, n1, n2, n3 };
    const int64_t weight_ne[4] = { ncols, 1, 1, 1 };
    const int64_t nrows        = n1 * n2 * n3;

    ggml_context_ptr ctx    = make_context();
    ggml_tensor *    src    = ggml_new_tensor(ctx.get(), GGML_TYPE_F32, 4, ne);
    ggml_tensor *    weight = ggml_new_tensor(ctx.get(), GGML_TYPE_F32, 4, weight_ne);
    ggml_tensor *    out    = ggml_mul(ctx.get(), ggml_rms_norm(ctx.get(), src, eps), weight);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> src_data(ncols * nrows);
    std::vector<float> weight_data(ncols);
    std::vector<float> expected(src_data.size());
    for (int64_t i = 0; i < ncols * nrows; ++i) {
        src_data[i] = static_cast<float>((i % 31) - 15) * 0.125f;
    }
    for (int64_t col = 0; col < ncols; ++col) {
        weight_data[col] = 0.5f + static_cast<float>(col % 11) * 0.0625f;
    }
    for (int64_t row = 0; row < nrows; ++row) {
        float sum = 0.0f;
        for (int64_t col = 0; col < ncols; ++col) {
            const float value = src_data[row * ncols + col];
            sum += value * value;
        }
        const float scale = 1.0f / std::sqrt(sum / static_cast<float>(ncols) + eps);
        for (int64_t col = 0; col < ncols; ++col) {
            expected[row * ncols + col] = src_data[row * ncols + col] * scale * weight_data[col];
        }
    }

    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> actual(expected.size(), -1.0f);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    expect_near(actual, expected, 1e-5f, "rms_norm_mul");
}

static void run_div_scalar_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor *    lhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 8);
    ggml_tensor *    rhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
    ggml_tensor *    out = ggml_div(ctx.get(), lhs, rhs);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    const std::vector<float> lhs_data = { 2.0f, -4.0f, 6.0f, -8.0f, 10.0f, -12.0f, 14.0f, -16.0f };
    const std::vector<float> rhs_data = { 2.0f };
    std::vector<float>       expected(lhs_data.size());
    for (size_t i = 0; i < lhs_data.size(); ++i) {
        expected[i] = lhs_data[i] / rhs_data[0];
    }

    ggml_backend_tensor_set(lhs, lhs_data.data(), 0, lhs_data.size() * sizeof(float));
    ggml_backend_tensor_set(rhs, rhs_data.data(), 0, rhs_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> actual(expected.size(), -1.0f);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    expect_near(actual, expected, 1e-6f, "div");
}

static void run_sum_rows_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    const int64_t    ncols = 8;
    const int64_t    nrows = 3;
    ggml_context_ptr ctx   = make_context();
    ggml_tensor *    src   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ncols, nrows);
    ggml_tensor *    out   = ggml_sum_rows(ctx.get(), src);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> src_data(ncols * nrows);
    std::vector<float> expected(nrows, 0.0f);
    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t col = 0; col < ncols; ++col) {
            const float value           = static_cast<float>((row + 1) * (col - 3));
            src_data[row * ncols + col] = value;
            expected[row] += value;
        }
    }

    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> actual(expected.size(), -1.0f);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    expect_near(actual, expected, 1e-6f, "sum_rows");
}

static void run_soft_max_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    const int64_t    ncols = 128;
    const int64_t    nrows = 2;
    ggml_context_ptr ctx   = make_context();
    ggml_tensor *    src   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ncols, nrows);
    ggml_tensor *    out   = ggml_soft_max(ctx.get(), src);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> src_data(ncols * nrows);
    std::vector<float> expected(ncols * nrows);
    for (int64_t row = 0; row < nrows; ++row) {
        float maximum = -INFINITY;
        for (int64_t col = 0; col < ncols; ++col) {
            const float value           = static_cast<float>((col % 17) - 8) * 0.25f + static_cast<float>(row);
            src_data[row * ncols + col] = value;
            maximum                     = std::max(maximum, value);
        }
        float sum = 0.0f;
        for (int64_t col = 0; col < ncols; ++col) {
            const float value           = std::exp(src_data[row * ncols + col] - maximum);
            expected[row * ncols + col] = value;
            sum += value;
        }
        for (int64_t col = 0; col < ncols; ++col) {
            expected[row * ncols + col] /= sum;
        }
    }

    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> actual(expected.size(), -1.0f);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    expect_near(actual, expected, 1e-6f, "soft_max");
}

static void run_argsort_case(ggml_backend_t backend, ggml_backend_dev_t dev, ggml_sort_order order) {
    const int64_t    ncols = 128;
    const int64_t    nrows = 2;
    ggml_context_ptr ctx   = make_context();
    ggml_tensor *    src   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ncols, nrows);
    ggml_tensor *    out   = ggml_argsort(ctx.get(), src, order);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float>   src_data(ncols * nrows);
    std::vector<int32_t> expected(ncols * nrows);
    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t col = 0; col < ncols; ++col) {
            src_data[row * ncols + col] = static_cast<float>((col * 37 + row * 11) % 131);
            expected[row * ncols + col] = static_cast<int32_t>(col);
        }
        std::sort(expected.begin() + row * ncols, expected.begin() + (row + 1) * ncols, [&](int32_t lhs, int32_t rhs) {
            const float lhs_value = src_data[row * ncols + lhs];
            const float rhs_value = src_data[row * ncols + rhs];
            return order == GGML_SORT_ORDER_ASC ? lhs_value < rhs_value : lhs_value > rhs_value;
        });
    }

    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<int32_t> actual(expected.size(), -1);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(int32_t));
    expect_eq_i32(actual, expected, "argsort");
}

static void run_swiglu_case(ggml_backend_t     backend,
                            ggml_backend_dev_t dev,
                            int64_t            output_size,
                            int64_t            route_count,
                            int64_t            token_count) {
    const int64_t    element_count = output_size * route_count * token_count;
    ggml_context_ptr ctx           = make_context();
    ggml_tensor *    gate = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, output_size, route_count, token_count);
    ggml_tensor *    up   = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, output_size, route_count, token_count);
    ggml_tensor *    out  = ggml_swiglu_split(ctx.get(), gate, up);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> gate_data(element_count);
    std::vector<float> up_data(element_count);
    std::vector<float> expected(element_count);
    for (int64_t i = 0; i < element_count; ++i) {
        gate_data[i] = static_cast<float>((i % 29) - 14) * 0.125f;
        up_data[i]   = static_cast<float>((i % 17) - 8) * 0.25f;
        expected[i]  = gate_data[i] / (1.0f + std::exp(-gate_data[i])) * up_data[i];
    }

    ggml_backend_tensor_set(gate, gate_data.data(), 0, gate_data.size() * sizeof(float));
    ggml_backend_tensor_set(up, up_data.data(), 0, up_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> actual(element_count, -1.0f);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    expect_near(actual, expected, 1e-5f, "swiglu");
}

static void run_qwen3_moe_router_projection_case(ggml_backend_t backend) {
    const char *      previous = std::getenv("GGML_HRX_LOOM_FORCE_ROUTE");
    const std::string saved    = previous ? previous : "";
    setenv("GGML_HRX_LOOM_FORCE_ROUTE", "qwen3_moe_router_projection_f32_four_row_wave32_bringup_workaround", 1);

    const int64_t hidden_size  = 2048;
    const int64_t expert_count = 128;
    const int64_t token_count  = 2;

    ggml_context_ptr ctx    = make_context();
    ggml_tensor *    weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, hidden_size, expert_count);
    ggml_tensor *    input  = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, hidden_size, token_count);
    ggml_tensor *    out    = ggml_mul_mat(ctx.get(), weight, input);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> weight_data(hidden_size * expert_count, 0.015625f);
    std::vector<float> input_data(hidden_size * token_count, 0.00390625f);
    const float        expected_value = static_cast<float>(hidden_size) * 0.015625f * 0.00390625f;
    std::vector<float> expected(expert_count * token_count, expected_value);

    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(float));
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> actual(expected.size(), -1.0f);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    expect_near(actual, expected, 1e-4f, "qwen3_moe_router_projection");

    if (previous) {
        setenv("GGML_HRX_LOOM_FORCE_ROUTE", saved.c_str(), 1);
    } else {
        unsetenv("GGML_HRX_LOOM_FORCE_ROUTE");
    }
}

static void run_qwen3_moe_router_top8_case(ggml_backend_t backend) {
    const char *      previous = std::getenv("GGML_HRX_LOOM_FORCE_ROUTE");
    const std::string saved    = previous ? previous : "";
    setenv("GGML_HRX_LOOM_FORCE_ROUTE", "qwen3_moe_router_top8_f32_stride_bringup_workaround", 1);

    const int64_t expert_count = 128;
    const int64_t route_count  = 8;
    const int64_t token_count  = 2;

    ggml_context_ptr ctx        = make_context();
    ggml_tensor *    logits     = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, expert_count, token_count);
    ggml_tensor *    probs      = ggml_soft_max(ctx.get(), logits);
    ggml_tensor *    route_ids  = ggml_argsort_top_k(ctx.get(), probs, route_count);
    ggml_tensor *    probs_3d   = ggml_reshape_3d(ctx.get(), probs, 1, expert_count, token_count);
    ggml_tensor *    weights_3d = ggml_get_rows(ctx.get(), probs_3d, route_ids);
    ggml_tensor *    weights    = ggml_reshape_2d(ctx.get(), weights_3d, route_count, token_count);
    ggml_tensor *    sum        = ggml_sum_rows(ctx.get(), weights);
    sum                         = ggml_clamp(ctx.get(), sum, 6.103515625e-5f, INFINITY);
    weights                     = ggml_div(ctx.get(), weights, sum);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, weights);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> logits_data(expert_count * token_count);
    for (int64_t token = 0; token < token_count; ++token) {
        for (int64_t expert = 0; expert < expert_count; ++expert) {
            logits_data[token * expert_count + expert] = static_cast<float>(expert % route_count);
        }
    }

    std::vector<int32_t> expected_ids(route_count * token_count);
    std::vector<float>   expected_weights(route_count * token_count, 0.125f);
    for (int64_t token = 0; token < token_count; ++token) {
        for (int64_t route = 0; route < route_count; ++route) {
            expected_ids[token * route_count + route] = static_cast<int32_t>(7 + 8 * route);
        }
    }

    ggml_backend_tensor_set(logits, logits_data.data(), 0, logits_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<int32_t> actual_ids(expected_ids.size(), -1);
    std::vector<float>   actual_weights(expected_weights.size(), -1.0f);
    const int64_t        route_id_stride = route_ids->nb[1] / ggml_type_size(route_ids->type);
    std::vector<int32_t> actual_id_storage(route_id_stride * token_count, -1);
    ggml_backend_tensor_get(route_ids->view_src, actual_id_storage.data(), route_ids->view_offs,
                            actual_id_storage.size() * sizeof(int32_t));
    for (int64_t token = 0; token < token_count; ++token) {
        for (int64_t route = 0; route < route_count; ++route) {
            actual_ids[token * route_count + route] = actual_id_storage[token * route_id_stride + route];
        }
    }
    ggml_backend_tensor_get(weights, actual_weights.data(), 0, actual_weights.size() * sizeof(float));
    expect_eq_i32(actual_ids, expected_ids, "qwen3_moe_router_top8_ids");
    expect_near(actual_weights, expected_weights, 1e-5f, "qwen3_moe_router_top8_weights");

    if (previous) {
        setenv("GGML_HRX_LOOM_FORCE_ROUTE", saved.c_str(), 1);
    } else {
        unsetenv("GGML_HRX_LOOM_FORCE_ROUTE");
    }
}

static void run_qwen3_moe_gate_up_case(ggml_backend_t backend) {
    const char *      previous = std::getenv("GGML_HRX_LOOM_FORCE_ROUTE");
    const std::string saved    = previous ? previous : "";
    setenv("GGML_HRX_LOOM_FORCE_ROUTE", "qwen3_moe_routed_gate_up_swiglu_q4k_f16_wmma", 1);

    const int64_t input_size   = 2048;
    const int64_t output_size  = 768;
    const int64_t expert_count = 128;
    const int64_t route_count  = 8;
    const int64_t token_count  = 1;

    ggml_context_ptr ctx         = make_context();
    ggml_tensor *    gate_weight = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q4_K, input_size, output_size, expert_count);
    ggml_tensor *    up_weight   = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q4_K, input_size, output_size, expert_count);
    ggml_tensor *    input       = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, input_size, 1, token_count);
    ggml_tensor *    route_ids   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, route_count, token_count);
    ggml_tensor *    gate        = ggml_mul_mat_id(ctx.get(), gate_weight, input, route_ids);
    ggml_tensor *    up          = ggml_mul_mat_id(ctx.get(), up_weight, input, route_ids);
    ggml_tensor *    out         = ggml_swiglu_split(ctx.get(), gate, up);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<uint8_t> gate_weight_data(ggml_nbytes(gate_weight), 0);
    std::vector<uint8_t> up_weight_data(ggml_nbytes(up_weight), 0);
    std::vector<float>   input_data(input_size * token_count, 0.0f);
    std::vector<int32_t> route_id_data(route_count * token_count);
    for (int64_t i = 0; i < route_count * token_count; ++i) {
        route_id_data[i] = static_cast<int32_t>(i % expert_count);
    }

    ggml_backend_tensor_set(gate_weight, gate_weight_data.data(), 0, gate_weight_data.size());
    ggml_backend_tensor_set(up_weight, up_weight_data.data(), 0, up_weight_data.size());
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    ggml_backend_tensor_set(route_ids, route_id_data.data(), 0, route_id_data.size() * sizeof(int32_t));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    const int64_t      element_count = output_size * route_count * token_count;
    std::vector<float> actual(element_count, -1.0f);
    std::vector<float> expected(element_count, 0.0f);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    expect_near(actual, expected, 1e-6f, "qwen3_moe_gate_up");

    if (previous) {
        setenv("GGML_HRX_LOOM_FORCE_ROUTE", saved.c_str(), 1);
    } else {
        unsetenv("GGML_HRX_LOOM_FORCE_ROUTE");
    }
}

static void run_qwen3_moe_routed_down_case(ggml_backend_t backend,
                                           ggml_type      weight_type,
                                           const char *   route_id,
                                           const char *   label) {
    const char *      previous = std::getenv("GGML_HRX_LOOM_FORCE_ROUTE");
    const std::string saved    = previous ? previous : "";
    setenv("GGML_HRX_LOOM_FORCE_ROUTE", route_id, 1);

    const int64_t input_size   = 768;
    const int64_t output_size  = 2048;
    const int64_t expert_count = 128;
    const int64_t route_count  = 8;
    const int64_t token_count  = 17;

    ggml_context_ptr ctx           = make_context();
    ggml_tensor *    down_weight   = ggml_new_tensor_3d(ctx.get(), weight_type, input_size, output_size, expert_count);
    ggml_tensor *    input         = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, input_size, route_count, token_count);
    ggml_tensor *    route_ids     = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, route_count, token_count);
    ggml_tensor *    down          = ggml_mul_mat_id(ctx.get(), down_weight, input, route_ids);
    ggml_tensor *    route_weights = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 1, route_count, token_count);
    ggml_tensor *    weighted_down = ggml_mul(ctx.get(), down, route_weights);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 64, false);
    ggml_build_forward_expand(graph, weighted_down);

    ggml_tensor * route_outputs[8] = {};
    for (int64_t i = 0; i < route_count; ++i) {
        route_outputs[i] = ggml_view_2d(ctx.get(), weighted_down, output_size, token_count, weighted_down->nb[2],
                                        i * weighted_down->nb[1]);
        ggml_build_forward_expand(graph, route_outputs[i]);
    }

    ggml_tensor * moe_out = route_outputs[0];
    for (int64_t i = 1; i < route_count; ++i) {
        moe_out = ggml_add(ctx.get(), moe_out, route_outputs[i]);
        ggml_build_forward_expand(graph, moe_out);
    }

    ggml_tensor * residual = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, output_size, token_count);
    ggml_tensor * out      = ggml_add(ctx.get(), moe_out, residual);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<uint8_t> down_weight_data(ggml_nbytes(down_weight), 0);
    std::vector<float>   input_data(input_size * route_count * token_count, 0.0f);
    std::vector<int32_t> route_id_data(route_count * token_count);
    std::vector<float>   route_weight_data(route_count * token_count, 0.125f);
    std::vector<float>   residual_data(output_size * token_count);
    for (int64_t i = 0; i < route_count * token_count; ++i) {
        route_id_data[i] = static_cast<int32_t>(i % expert_count);
    }
    for (int64_t i = 0; i < output_size * token_count; ++i) {
        residual_data[i] = static_cast<float>((i % 31) - 15) * 0.03125f;
    }

    ggml_backend_tensor_set(down_weight, down_weight_data.data(), 0, down_weight_data.size());
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    ggml_backend_tensor_set(route_ids, route_id_data.data(), 0, route_id_data.size() * sizeof(int32_t));
    ggml_backend_tensor_set(route_weights, route_weight_data.data(), 0, route_weight_data.size() * sizeof(float));
    ggml_backend_tensor_set(residual, residual_data.data(), 0, residual_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> actual(residual_data.size(), -1.0f);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    expect_near(actual, residual_data, 1e-6f, label);

    if (previous) {
        setenv("GGML_HRX_LOOM_FORCE_ROUTE", saved.c_str(), 1);
    } else {
        unsetenv("GGML_HRX_LOOM_FORCE_ROUTE");
    }
}

static void run_qwen3_moe_routed_down_case(ggml_backend_t backend) {
    run_qwen3_moe_routed_down_case(backend, GGML_TYPE_Q4_K, "qwen3_moe_routed_down_q4k_f16_wmma_residual",
                                   "qwen3_moe_routed_down_q4k");
    run_qwen3_moe_routed_down_case(backend, GGML_TYPE_Q6_K, "qwen3_moe_routed_down_q6k_f16_wmma_residual",
                                   "qwen3_moe_routed_down_q6k");
}

static void run_qwen3_moe_routed_down_next_rmsnorm_case(ggml_backend_t backend,
                                                        ggml_type      weight_type,
                                                        const char *   route_id,
                                                        const char *   label) {
    const char *      previous = std::getenv("GGML_HRX_LOOM_FORCE_ROUTE");
    const std::string saved    = previous ? previous : "";
    setenv("GGML_HRX_LOOM_FORCE_ROUTE", route_id, 1);

    const int64_t input_size   = 768;
    const int64_t output_size  = 2048;
    const int64_t expert_count = 128;
    const int64_t route_count  = 8;
    const int64_t token_count  = 17;
    const float   eps          = 1.0e-6f;

    ggml_context_ptr ctx           = make_context();
    ggml_tensor *    down_weight   = ggml_new_tensor_3d(ctx.get(), weight_type, input_size, output_size, expert_count);
    ggml_tensor *    input         = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, input_size, route_count, token_count);
    ggml_tensor *    route_ids     = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, route_count, token_count);
    ggml_tensor *    down          = ggml_mul_mat_id(ctx.get(), down_weight, input, route_ids);
    ggml_tensor *    route_weights = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 1, route_count, token_count);
    ggml_tensor *    weighted_down = ggml_mul(ctx.get(), down, route_weights);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 96, false);
    ggml_build_forward_expand(graph, weighted_down);

    ggml_tensor * route_outputs[8] = {};
    for (int64_t i = 0; i < route_count; ++i) {
        route_outputs[i] = ggml_view_2d(ctx.get(), weighted_down, output_size, token_count, weighted_down->nb[2],
                                        i * weighted_down->nb[1]);
        ggml_build_forward_expand(graph, route_outputs[i]);
    }

    ggml_tensor * moe_out = route_outputs[0];
    for (int64_t i = 1; i < route_count; ++i) {
        moe_out = ggml_add(ctx.get(), moe_out, route_outputs[i]);
        ggml_build_forward_expand(graph, moe_out);
    }

    ggml_tensor * residual              = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, output_size, token_count);
    ggml_tensor * out                   = ggml_add(ctx.get(), moe_out, residual);
    ggml_tensor * next_norm_weight      = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, output_size);
    ggml_tensor * rms                   = ggml_rms_norm(ctx.get(), out, eps);
    ggml_tensor * next_projection_input = ggml_mul(ctx.get(), rms, next_norm_weight);
    ggml_build_forward_expand(graph, out);
    ggml_build_forward_expand(graph, next_projection_input);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<uint8_t> down_weight_data(ggml_nbytes(down_weight), 0);
    std::vector<float>   input_data(input_size * route_count * token_count, 0.0f);
    std::vector<int32_t> route_id_data(route_count * token_count);
    std::vector<float>   route_weight_data(route_count * token_count, 0.125f);
    std::vector<float>   residual_data(output_size * token_count);
    std::vector<float>   next_norm_weight_data(output_size);
    std::vector<float>   expected_next(output_size * token_count);
    for (int64_t i = 0; i < route_count * token_count; ++i) {
        route_id_data[i] = static_cast<int32_t>(i % expert_count);
    }
    for (int64_t i = 0; i < output_size; ++i) {
        next_norm_weight_data[i] = 0.75f + static_cast<float>(i % 19) * 0.015625f;
    }
    for (int64_t i = 0; i < output_size * token_count; ++i) {
        residual_data[i] = static_cast<float>((i % 31) - 15) * 0.03125f;
    }
    for (int64_t token = 0; token < token_count; ++token) {
        float sum_squares = 0.0f;
        for (int64_t channel = 0; channel < output_size; ++channel) {
            const float value = residual_data[token * output_size + channel];
            sum_squares += value * value;
        }
        const float scale = 1.0f / std::sqrt(sum_squares / static_cast<float>(output_size) + eps);
        for (int64_t channel = 0; channel < output_size; ++channel) {
            expected_next[token * output_size + channel] =
                residual_data[token * output_size + channel] * scale * next_norm_weight_data[channel];
        }
    }

    ggml_backend_tensor_set(down_weight, down_weight_data.data(), 0, down_weight_data.size());
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    ggml_backend_tensor_set(route_ids, route_id_data.data(), 0, route_id_data.size() * sizeof(int32_t));
    ggml_backend_tensor_set(route_weights, route_weight_data.data(), 0, route_weight_data.size() * sizeof(float));
    ggml_backend_tensor_set(residual, residual_data.data(), 0, residual_data.size() * sizeof(float));
    ggml_backend_tensor_set(next_norm_weight, next_norm_weight_data.data(), 0,
                            next_norm_weight_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> actual_out(residual_data.size(), -1.0f);
    ggml_backend_tensor_get(out, actual_out.data(), 0, actual_out.size() * sizeof(float));
    expect_near(actual_out, residual_data, 1e-6f, label);

    std::vector<float> actual_next(expected_next.size(), -1.0f);
    ggml_backend_tensor_get(next_projection_input, actual_next.data(), 0, actual_next.size() * sizeof(float));
    expect_near(actual_next, expected_next, 1e-3f, label);

    if (previous) {
        setenv("GGML_HRX_LOOM_FORCE_ROUTE", saved.c_str(), 1);
    } else {
        unsetenv("GGML_HRX_LOOM_FORCE_ROUTE");
    }
}

static void run_qwen3_moe_routed_down_next_rmsnorm_case(ggml_backend_t backend) {
    run_qwen3_moe_routed_down_next_rmsnorm_case(backend, GGML_TYPE_Q4_K,
                                                "qwen3_moe_routed_down_q4k_f16_wmma_next_rmsnorm",
                                                "qwen3_moe_routed_down_q4k_next_rmsnorm");
    run_qwen3_moe_routed_down_next_rmsnorm_case(backend, GGML_TYPE_Q6_K,
                                                "qwen3_moe_routed_down_q6k_f16_wmma_next_rmsnorm",
                                                "qwen3_moe_routed_down_q6k_next_rmsnorm");
}

}  // namespace

int main() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("HRX0");
    if (!dev) {
        std::fprintf(stderr, "HRX0 not available; skipping test-backend-hrx\n");
        return 0;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    GGML_ASSERT(buft != nullptr);
    {
        ggml_context_ptr        standalone_ctx = make_context();
        ggml_tensor *           standalone     = ggml_new_tensor_1d(standalone_ctx.get(), GGML_TYPE_F32, 4);
        ggml_backend_buffer_ptr standalone_buffer(ggml_backend_alloc_ctx_tensors_from_buft(standalone_ctx.get(), buft));
        GGML_ASSERT(standalone_buffer != nullptr);

        const std::vector<float> standalone_input = { 10.0f, 11.0f, 12.0f, 13.0f };
        ggml_backend_tensor_set(standalone, standalone_input.data(), 0, standalone_input.size() * sizeof(float));

        std::vector<float> standalone_output(standalone_input.size(), -1.0f);
        ggml_backend_tensor_get(standalone, standalone_output.data(), 0, standalone_output.size() * sizeof(float));
        expect_eq(standalone_output, standalone_input, "standalone_output");
    }

    ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
    GGML_ASSERT(backend != nullptr);

    if (const char * test_only = std::getenv("GGML_HRX_TEST_ONLY")) {
        if (std::string(test_only) == "scale") {
            if (run_scale_support_case(dev)) {
                run_scale_case(backend.get(), dev, 1);
                run_scale_case(backend.get(), dev, 256);
                run_scale_case(backend.get(), dev, 257);
            }
            ggml_backend_synchronize(backend.get());
            return 0;
        }
        if (std::string(test_only) == "clamp") {
            if (run_clamp_support_case(dev)) {
                run_clamp_case(backend.get(), dev, 1);
                run_clamp_case(backend.get(), dev, 512);
            }
            ggml_backend_synchronize(backend.get());
            return 0;
        }
        if (std::string(test_only) == "add") {
            if (run_add_support_case(dev)) {
                run_add_case(backend.get(), dev, 1);
                run_add_case(backend.get(), dev, 257);
                run_add_case(backend.get(), dev, 2048);
            }
            ggml_backend_synchronize(backend.get());
            return 0;
        }
        if (std::string(test_only) == "add_transient") {
            if (run_add_support_case(dev)) {
                run_add_transient_case(backend.get(), dev);
            }
            ggml_backend_synchronize(backend.get());
            return 0;
        }
        if (std::string(test_only) == "add_transient_pair") {
            if (run_add_support_case(dev)) {
                run_add_transient_pair_case(backend.get(), dev);
            }
            ggml_backend_synchronize(backend.get());
            return 0;
        }
        if (std::string(test_only) == "graph_transient") {
            run_graph_transient_case(backend.get());
            ggml_backend_synchronize(backend.get());
            return 0;
        }
        if (std::string(test_only) == "mul") {
            if (run_mul_support_case(dev)) {
                run_mul_row_broadcast_case(backend.get(), dev);
            }
            ggml_backend_synchronize(backend.get());
            return 0;
        }
        if (std::string(test_only) == "rms_norm_mul") {
            run_rms_norm_mul_case(backend.get());
            ggml_backend_synchronize(backend.get());
            return 0;
        }
        if (std::string(test_only) == "div") {
            if (run_div_support_case(dev)) {
                run_div_scalar_case(backend.get(), dev);
            }
            ggml_backend_synchronize(backend.get());
            return 0;
        }
        if (std::string(test_only) == "sum_rows") {
            if (run_sum_rows_support_case(dev)) {
                run_sum_rows_case(backend.get(), dev);
            }
            ggml_backend_synchronize(backend.get());
            return 0;
        }
        if (std::string(test_only) == "soft_max") {
            if (run_soft_max_support_case(dev)) {
                run_soft_max_case(backend.get(), dev);
            }
            ggml_backend_synchronize(backend.get());
            return 0;
        }
        if (std::string(test_only) == "argsort") {
            if (run_argsort_support_case(dev)) {
                run_argsort_case(backend.get(), dev, GGML_SORT_ORDER_ASC);
                run_argsort_case(backend.get(), dev, GGML_SORT_ORDER_DESC);
            }
            ggml_backend_synchronize(backend.get());
            return 0;
        }
        if (std::string(test_only) == "swiglu") {
            if (run_swiglu_support_case(dev)) {
                run_swiglu_case(backend.get(), dev, 33, 2, 67);
                run_swiglu_case(backend.get(), dev, 768, 8, 1);
            }
            ggml_backend_synchronize(backend.get());
            return 0;
        }
        if (std::string(test_only) == "qwen3_moe_gate_up") {
            run_qwen3_moe_gate_up_case(backend.get());
            ggml_backend_synchronize(backend.get());
            return 0;
        }
        if (std::string(test_only) == "qwen3_moe_router") {
            run_qwen3_moe_router_projection_case(backend.get());
            run_qwen3_moe_router_top8_case(backend.get());
            ggml_backend_synchronize(backend.get());
            return 0;
        }
        if (std::string(test_only) == "qwen3_moe_routed_down") {
            run_qwen3_moe_routed_down_case(backend.get());
            ggml_backend_synchronize(backend.get());
            return 0;
        }
        if (std::string(test_only) == "qwen3_moe_routed_down_next_rmsnorm") {
            run_qwen3_moe_routed_down_next_rmsnorm_case(backend.get());
            ggml_backend_synchronize(backend.get());
            return 0;
        }
        std::fprintf(stderr, "unknown GGML_HRX_TEST_ONLY=%s\n", test_only);
        return 1;
    }

    if (run_scale_support_case(dev)) {
        run_scale_case(backend.get(), dev, 1);
        run_scale_case(backend.get(), dev, 256);
        run_scale_case(backend.get(), dev, 257);
    }
    if (run_clamp_support_case(dev)) {
        run_clamp_case(backend.get(), dev, 1);
        run_clamp_case(backend.get(), dev, 512);
    }
    if (run_add_support_case(dev)) {
        run_add_case(backend.get(), dev, 1);
        run_add_case(backend.get(), dev, 257);
        run_add_case(backend.get(), dev, 2048);
        run_add_transient_case(backend.get(), dev);
        run_add_transient_pair_case(backend.get(), dev);
        run_graph_transient_case(backend.get());
    }
    if (run_mul_support_case(dev)) {
        run_mul_row_broadcast_case(backend.get(), dev);
    }
    if (run_div_support_case(dev)) {
        run_div_scalar_case(backend.get(), dev);
    }
    if (run_sum_rows_support_case(dev)) {
        run_sum_rows_case(backend.get(), dev);
    }
    if (run_soft_max_support_case(dev)) {
        run_soft_max_case(backend.get(), dev);
    }
    if (run_argsort_support_case(dev)) {
        run_argsort_case(backend.get(), dev, GGML_SORT_ORDER_ASC);
        run_argsort_case(backend.get(), dev, GGML_SORT_ORDER_DESC);
    }
    if (run_swiglu_support_case(dev)) {
        run_swiglu_case(backend.get(), dev, 33, 2, 67);
        run_swiglu_case(backend.get(), dev, 768, 8, 1);
    }

    ggml_backend_synchronize(backend.get());
    return 0;
}
