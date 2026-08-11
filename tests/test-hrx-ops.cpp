#include "backend-context.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-hrx.h"
#include "ggml.h"
#include "runtime/graph-executor.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define REQUIRE(condition)                                                                           \
    do {                                                                                             \
        if (!(condition)) {                                                                          \
            std::fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, #condition); \
            std::abort();                                                                            \
        }                                                                                            \
    } while (false)

static constexpr float kQwenRmsNormEps = 0.000001f;

static std::vector<float> make_input(int64_t hidden_size, int64_t token_count) {
    std::vector<float> data(hidden_size * token_count);
    for (int64_t i = 0; i < static_cast<int64_t>(data.size()); ++i) {
        data[i] = static_cast<float>((i % 29) - 14) * 0.125f;
    }
    return data;
}

static std::vector<float> make_weight(int64_t hidden_size) {
    std::vector<float> data(hidden_size);
    for (int64_t i = 0; i < hidden_size; ++i) {
        data[i] = 0.5f + static_cast<float>(i % 17) * 0.03125f;
    }
    return data;
}

static std::vector<float> rmsnorm_mul_reference(const std::vector<float> & input,
                                                const std::vector<float> & weight,
                                                int64_t                    hidden_size,
                                                int64_t                    token_count) {
    std::vector<float> output(input.size());
    for (int64_t token = 0; token < token_count; ++token) {
        float sum_squares = 0.0f;
        for (int64_t column = 0; column < hidden_size; ++column) {
            const float value = input[token * hidden_size + column];
            sum_squares += value * value;
        }
        const float scale = 1.0f / std::sqrt(sum_squares / static_cast<float>(hidden_size) + kQwenRmsNormEps);
        for (int64_t column = 0; column < hidden_size; ++column) {
            output[token * hidden_size + column] = input[token * hidden_size + column] * scale * weight[column];
        }
    }
    return output;
}

static ggml_tensor * build_rmsnorm_mul_graph(ggml_context * ctx,
                                             ggml_tensor *  input,
                                             ggml_tensor *  weight,
                                             float          eps = kQwenRmsNormEps) {
    ggml_tensor * rms = ggml_rms_norm(ctx, input, eps);
    REQUIRE(rms != nullptr);
    ggml_tensor * output = ggml_mul(ctx, rms, weight);
    REQUIRE(output != nullptr);
    return output;
}

static void run_rmsnorm_support_checks() {
    ggml_backend_hrx_device_context device_context  = {};
    ggml_backend_hrx_context        backend_context = {};
    device_context.architecture                     = "gfx1151";
    backend_context.device                          = &device_context;
    const ggml::hrx::GraphExecutor executor(backend_context);

    ggml_init_params params = {};
    params.mem_size         = 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 1);
    ggml_tensor * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
    REQUIRE(input != nullptr);
    REQUIRE(weight != nullptr);
    ggml_tensor * output = build_rmsnorm_mul_graph(ctx, input, weight);
    ggml_cgraph * graph  = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);
    const ggml::hrx::GraphSupportResult support = executor.can_execute(*graph);
    REQUIRE(support.supported);
    REQUIRE(support.status.success());

    ggml_tensor * wrong_eps_output = build_rmsnorm_mul_graph(ctx, input, weight, 1.0e-5f);
    ggml_cgraph * wrong_eps_graph  = ggml_new_graph(ctx);
    REQUIRE(wrong_eps_graph != nullptr);
    ggml_build_forward_expand(wrong_eps_graph, wrong_eps_output);
    const ggml::hrx::GraphSupportResult wrong_eps_support = executor.can_execute(*wrong_eps_graph);
    REQUIRE(!wrong_eps_support.supported);

    ggml_tensor * wrong_type_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 256, 1);
    ggml_tensor * wrong_type_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 256);
    REQUIRE(wrong_type_input != nullptr);
    REQUIRE(wrong_type_weight != nullptr);
    ggml_tensor * wrong_type_output = build_rmsnorm_mul_graph(ctx, wrong_type_input, wrong_type_weight);
    ggml_cgraph * wrong_type_graph  = ggml_new_graph(ctx);
    REQUIRE(wrong_type_graph != nullptr);
    ggml_build_forward_expand(wrong_type_graph, wrong_type_output);
    const ggml::hrx::GraphSupportResult wrong_type_support = executor.can_execute(*wrong_type_graph);
    REQUIRE(!wrong_type_support.supported);

    ggml_tensor * wrong_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
    REQUIRE(wrong_weight != nullptr);
    ggml_tensor * wrong_weight_output = build_rmsnorm_mul_graph(ctx, input, wrong_weight);
    ggml_cgraph * wrong_weight_graph  = ggml_new_graph(ctx);
    REQUIRE(wrong_weight_graph != nullptr);
    ggml_build_forward_expand(wrong_weight_graph, wrong_weight_output);
    const ggml::hrx::GraphSupportResult wrong_weight_support = executor.can_execute(*wrong_weight_graph);
    REQUIRE(!wrong_weight_support.supported);

    ggml_free(ctx);
}

static void run_rmsnorm_mul_case(int64_t hidden_size, int64_t token_count) {
    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = static_cast<size_t>(hidden_size * token_count * sizeof(float) * 8 + 1024 * 1024);
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, token_count);
    ggml_tensor * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
    REQUIRE(input != nullptr);
    REQUIRE(weight != nullptr);
    ggml_tensor * output = build_rmsnorm_mul_graph(ctx, input, weight);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);

    const std::vector<float> input_data  = make_input(hidden_size, token_count);
    const std::vector<float> weight_data = make_weight(hidden_size);
    const std::vector<float> expected    = rmsnorm_mul_reference(input_data, weight_data, hidden_size, token_count);

    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    for (size_t i = 0; i < actual.size(); ++i) {
        const float diff = std::fabs(actual[i] - expected[i]);
        REQUIRE(diff <= 5.0e-4f);
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

int main() {
    run_rmsnorm_support_checks();

    if (ggml_backend_hrx_get_device_count() == 0) {
        std::fprintf(stderr, "test skipped: no HRX devices available\n");
        return 0;
    }

    run_rmsnorm_mul_case(256, 1);
    run_rmsnorm_mul_case(256, 4);
    run_rmsnorm_mul_case(2048, 1);
    return 0;
}
