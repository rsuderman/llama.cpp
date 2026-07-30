#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <ggml-hrx.h>

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
            std::fprintf(stderr, "%s[%zu]: got %.9g expected %.9g\n",
                label, i, actual[i], expected[i]);
            std::abort();
        }
    }
}

static void expect_eq_i32(const std::vector<int32_t> & actual, const std::vector<int32_t> & expected, const char * label) {
    GGML_ASSERT(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::fprintf(stderr, "%s[%zu]: got %d expected %d\n",
                label, i, actual[i], expected[i]);
            std::abort();
        }
    }
}

static void expect_near(
        const std::vector<float> & actual,
        const std::vector<float> & expected,
        float tolerance,
        const char * label) {
    GGML_ASSERT(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(actual[i] - expected[i]) > tolerance) {
            std::fprintf(stderr, "%s[%zu]: got %.9g expected %.9g tolerance %.9g\n",
                label, i, actual[i], expected[i], tolerance);
            std::abort();
        }
    }
}

static bool run_scale_support_case(ggml_backend_dev_t dev) {
    bool scale_f32_supported = false;
    {
        ggml_context_ptr ctx = make_context();
        ggml_tensor * src = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 16);
        ggml_tensor * out = ggml_scale_bias(ctx.get(), src, 1.25f, -0.75f);
        scale_f32_supported = ggml_backend_dev_supports_op(dev, out);
    }

    {
        ggml_context_ptr ctx = make_context();
        ggml_tensor * src = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F16, 16);
        ggml_tensor * out = ggml_scale_bias(ctx.get(), src, 1.25f, -0.75f);
        GGML_ASSERT(!ggml_backend_dev_supports_op(dev, out));
    }
    return scale_f32_supported;
}

static bool run_clamp_support_case(ggml_backend_dev_t dev) {
    bool clamp_f32_supported = false;
    {
        ggml_context_ptr ctx = make_context();
        ggml_tensor * src = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 16);
        ggml_tensor * out = ggml_clamp(ctx.get(), src, -1.5f, 2.0f);
        clamp_f32_supported = ggml_backend_dev_supports_op(dev, out);
    }

    {
        ggml_context_ptr ctx = make_context();
        ggml_tensor * src = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F16, 16);
        ggml_tensor * out = ggml_clamp(ctx.get(), src, -1.5f, 2.0f);
        GGML_ASSERT(!ggml_backend_dev_supports_op(dev, out));
    }
    return clamp_f32_supported;
}

static bool run_add_support_case(ggml_backend_dev_t dev) {
    bool add_f32_supported = false;
    {
        ggml_context_ptr ctx = make_context();
        ggml_tensor * lhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 16);
        ggml_tensor * rhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 16);
        ggml_tensor * out = ggml_add(ctx.get(), lhs, rhs);
        add_f32_supported = ggml_backend_dev_supports_op(dev, out);
    }

    {
        ggml_context_ptr ctx = make_context();
        ggml_tensor * lhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F16, 16);
        ggml_tensor * rhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F16, 16);
        ggml_tensor * out = ggml_add(ctx.get(), lhs, rhs);
        GGML_ASSERT(!ggml_backend_dev_supports_op(dev, out));
    }
    return add_f32_supported;
}

static bool run_mul_support_case(ggml_backend_dev_t dev) {
    bool mul_f32_supported = false;
    {
        ggml_context_ptr ctx = make_context();
        ggml_tensor * lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 128, 32);
        ggml_tensor * rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 128, 1);
        ggml_tensor * out = ggml_mul(ctx.get(), lhs, rhs);
        mul_f32_supported = ggml_backend_dev_supports_op(dev, out);
    }

    {
        ggml_context_ptr ctx = make_context();
        ggml_tensor * lhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F16, 16);
        ggml_tensor * rhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F16, 16);
        ggml_tensor * out = ggml_mul(ctx.get(), lhs, rhs);
        GGML_ASSERT(!ggml_backend_dev_supports_op(dev, out));
    }

    {
        ggml_context_ptr ctx = make_context();
        ggml_tensor * base = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, 4);
        ggml_tensor * view = ggml_view_2d(ctx.get(), base, 2, 2, 4 * sizeof(float), 0);
        ggml_tensor * rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 2, 2);
        ggml_tensor * out = ggml_mul(ctx.get(), view, rhs);
        GGML_ASSERT(!ggml_backend_dev_supports_op(dev, out));
    }
    return mul_f32_supported;
}

static bool run_div_support_case(ggml_backend_dev_t dev) {
    bool div_f32_supported = false;
    {
        ggml_context_ptr ctx = make_context();
        ggml_tensor * lhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 8);
        ggml_tensor * rhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        ggml_tensor * out = ggml_div(ctx.get(), lhs, rhs);
        div_f32_supported = ggml_backend_dev_supports_op(dev, out);
    }

    {
        ggml_context_ptr ctx = make_context();
        ggml_tensor * lhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F16, 16);
        ggml_tensor * rhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F16, 1);
        ggml_tensor * out = ggml_div(ctx.get(), lhs, rhs);
        GGML_ASSERT(!ggml_backend_dev_supports_op(dev, out));
    }
    return div_f32_supported;
}

static bool run_sum_rows_support_case(ggml_backend_dev_t dev) {
    bool sum_rows_f32_supported = false;
    {
        ggml_context_ptr ctx = make_context();
        ggml_tensor * src = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 8, 1);
        ggml_tensor * out = ggml_sum_rows(ctx.get(), src);
        sum_rows_f32_supported = ggml_backend_dev_supports_op(dev, out);
    }

    {
        ggml_context_ptr ctx = make_context();
        ggml_tensor * src = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F16, 8);
        ggml_tensor * out = ggml_sum_rows(ctx.get(), src);
        GGML_ASSERT(!ggml_backend_dev_supports_op(dev, out));
    }
    return sum_rows_f32_supported;
}

static bool run_soft_max_support_case(ggml_backend_dev_t dev) {
    bool soft_max_f32_supported = false;
    {
        ggml_context_ptr ctx = make_context();
        ggml_tensor * src = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 128, 1);
        ggml_tensor * out = ggml_soft_max(ctx.get(), src);
        soft_max_f32_supported = ggml_backend_dev_supports_op(dev, out);
    }

    {
        ggml_context_ptr ctx = make_context();
        ggml_tensor * src = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 257);
        ggml_tensor * out = ggml_soft_max(ctx.get(), src);
        GGML_ASSERT(!ggml_backend_dev_supports_op(dev, out));
    }
    return soft_max_f32_supported;
}

static bool run_argsort_support_case(ggml_backend_dev_t dev) {
    bool argsort_f32_i32_supported = false;
    {
        ggml_context_ptr ctx = make_context();
        ggml_tensor * src = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 128, 1);
        ggml_tensor * out = ggml_argsort(ctx.get(), src, GGML_SORT_ORDER_DESC);
        argsort_f32_i32_supported = ggml_backend_dev_supports_op(dev, out);
    }

    {
        ggml_context_ptr ctx = make_context();
        ggml_tensor * src = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 257);
        ggml_tensor * out = ggml_argsort(ctx.get(), src, GGML_SORT_ORDER_DESC);
        GGML_ASSERT(!ggml_backend_dev_supports_op(dev, out));
    }
    return argsort_f32_i32_supported;
}

static void run_scale_case(ggml_backend_t backend, ggml_backend_dev_t dev, int64_t n) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    ggml_tensor * out = ggml_scale_bias(ctx.get(), src, 1.25f, -0.75f);
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
    ggml_tensor * src = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    ggml_tensor * out = ggml_clamp(ctx.get(), src, -1.5f, 2.0f);
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
    ggml_tensor * lhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    ggml_tensor * rhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    ggml_tensor * out = ggml_add(ctx.get(), lhs, rhs);
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

static void run_mul_row_broadcast_case(ggml_backend_t backend, ggml_backend_dev_t dev) {
    const int64_t ncols = 128;
    const int64_t nrows = 32;
    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ncols, nrows);
    ggml_tensor * rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ncols, 1);
    ggml_tensor * out = ggml_mul(ctx.get(), lhs, rhs);
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
    const float eps = 1e-6f;
    const int64_t ncols = 64;
    const int64_t n1 = 3;
    const int64_t n2 = 2;
    const int64_t n3 = 2;
    const int64_t ne[4] = { ncols, n1, n2, n3 };
    const int64_t weight_ne[4] = { ncols, 1, 1, 1 };
    const int64_t nrows = n1 * n2 * n3;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor(ctx.get(), GGML_TYPE_F32, 4, ne);
    ggml_tensor * weight = ggml_new_tensor(ctx.get(), GGML_TYPE_F32, 4, weight_ne);
    ggml_tensor * out = ggml_mul(ctx.get(), ggml_rms_norm(ctx.get(), src, eps), weight);

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
    ggml_tensor * lhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 8);
    ggml_tensor * rhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
    ggml_tensor * out = ggml_div(ctx.get(), lhs, rhs);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    const std::vector<float> lhs_data = { 2.0f, -4.0f, 6.0f, -8.0f, 10.0f, -12.0f, 14.0f, -16.0f };
    const std::vector<float> rhs_data = { 2.0f };
    std::vector<float> expected(lhs_data.size());
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
    const int64_t ncols = 8;
    const int64_t nrows = 3;
    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ncols, nrows);
    ggml_tensor * out = ggml_sum_rows(ctx.get(), src);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> src_data(ncols * nrows);
    std::vector<float> expected(nrows, 0.0f);
    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t col = 0; col < ncols; ++col) {
            const float value = static_cast<float>((row + 1) * (col - 3));
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
    const int64_t ncols = 128;
    const int64_t nrows = 2;
    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ncols, nrows);
    ggml_tensor * out = ggml_soft_max(ctx.get(), src);
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
            const float value = static_cast<float>((col % 17) - 8) * 0.25f + static_cast<float>(row);
            src_data[row * ncols + col] = value;
            maximum = std::max(maximum, value);
        }
        float sum = 0.0f;
        for (int64_t col = 0; col < ncols; ++col) {
            const float value = std::exp(src_data[row * ncols + col] - maximum);
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
    const int64_t ncols = 128;
    const int64_t nrows = 2;
    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ncols, nrows);
    ggml_tensor * out = ggml_argsort(ctx.get(), src, order);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> src_data(ncols * nrows);
    std::vector<int32_t> expected(ncols * nrows);
    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t col = 0; col < ncols; ++col) {
            src_data[row * ncols + col] = static_cast<float>((col * 37 + row * 11) % 131);
            expected[row * ncols + col] = static_cast<int32_t>(col);
        }
        std::sort(
            expected.begin() + row * ncols,
            expected.begin() + (row + 1) * ncols,
            [&](int32_t lhs, int32_t rhs) {
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

} // namespace

int main() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("HRX0");
    if (!dev) {
        std::fprintf(stderr, "HRX0 not available; skipping test-backend-hrx\n");
        return 0;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    GGML_ASSERT(buft != nullptr);
    {
        ggml_context_ptr standalone_ctx = make_context();
        ggml_tensor * standalone = ggml_new_tensor_1d(standalone_ctx.get(), GGML_TYPE_F32, 4);
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

    ggml_backend_synchronize(backend.get());
    return 0;
}
