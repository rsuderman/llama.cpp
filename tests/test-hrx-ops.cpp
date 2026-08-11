#include "backend-context.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-hrx.h"
#include "ggml.h"
#include "runtime/graph-executor.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#define REQUIRE(condition)                                                                           \
    do {                                                                                             \
        if (!(condition)) {                                                                          \
            std::fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, #condition); \
            std::abort();                                                                            \
        }                                                                                            \
    } while (false)

static constexpr float   kQwenRmsNormEps    = 0.000001f;
static constexpr int64_t kQwenFlashHeadSize = 128;

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

static std::vector<float> make_router_input(int64_t hidden_size, int64_t token_count) {
    std::vector<float> data(hidden_size * token_count);
    for (int64_t i = 0; i < static_cast<int64_t>(data.size()); ++i) {
        data[i] = static_cast<float>((i % 41) - 20) * 0.01f;
    }
    return data;
}

static std::vector<float> make_router_weight(int64_t hidden_size, int64_t expert_count) {
    std::vector<float> data(hidden_size * expert_count);
    for (int64_t expert = 0; expert < expert_count; ++expert) {
        for (int64_t column = 0; column < hidden_size; ++column) {
            data[expert * hidden_size + column] = static_cast<float>(((expert + column) % 31) - 15) * 0.0025f;
        }
    }
    return data;
}

static std::vector<float> make_flash_query(int64_t token_count) {
    std::vector<float> data(kQwenFlashHeadSize * token_count);
    for (int64_t i = 0; i < static_cast<int64_t>(data.size()); ++i) {
        data[i] = static_cast<float>((i % 37) - 18) * 0.01f;
    }
    return data;
}

static std::vector<ggml_fp16_t> make_flash_key_value(int64_t token_count, int offset) {
    std::vector<ggml_fp16_t> data(kQwenFlashHeadSize * token_count);
    for (int64_t i = 0; i < static_cast<int64_t>(data.size()); ++i) {
        const float value = static_cast<float>(((i + offset) % 31) - 15) * 0.015f;
        data[i]           = ggml_fp32_to_fp16(value);
    }
    return data;
}

static std::vector<ggml_fp16_t> make_flash_mask(int64_t query_token_count, int64_t key_value_token_count) {
    std::vector<ggml_fp16_t> data(query_token_count * key_value_token_count);
    for (int64_t query = 0; query < query_token_count; ++query) {
        for (int64_t key = 0; key < key_value_token_count; ++key) {
            const float value                         = key <= query + 1 ? 0.0f : -10000.0f;
            data[query * key_value_token_count + key] = ggml_fp32_to_fp16(value);
        }
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

static std::vector<float> router_projection_reference(const std::vector<float> & input,
                                                      const std::vector<float> & weight,
                                                      int64_t                    hidden_size,
                                                      int64_t                    expert_count,
                                                      int64_t                    token_count) {
    std::vector<float> output(expert_count * token_count);
    for (int64_t token = 0; token < token_count; ++token) {
        for (int64_t expert = 0; expert < expert_count; ++expert) {
            float sum = 0.0f;
            for (int64_t column = 0; column < hidden_size; ++column) {
                sum += input[token * hidden_size + column] * weight[expert * hidden_size + column];
            }
            output[token * expert_count + expert] = sum;
        }
    }
    return output;
}

static std::vector<float> flash_attention_reference(const std::vector<float> &       query,
                                                    const std::vector<ggml_fp16_t> & key,
                                                    const std::vector<ggml_fp16_t> & value,
                                                    const std::vector<ggml_fp16_t> & mask,
                                                    int64_t                          query_token_count,
                                                    int64_t                          key_value_token_count) {
    std::vector<float> output(query_token_count * kQwenFlashHeadSize);
    const float        scale = 1.0f / std::sqrt(static_cast<float>(kQwenFlashHeadSize));
    for (int64_t query_token = 0; query_token < query_token_count; ++query_token) {
        std::vector<float> scores(key_value_token_count);
        float              max_score = -std::numeric_limits<float>::infinity();
        for (int64_t key_token = 0; key_token < key_value_token_count; ++key_token) {
            float dot = 0.0f;
            for (int64_t channel = 0; channel < kQwenFlashHeadSize; ++channel) {
                dot += query[query_token * kQwenFlashHeadSize + channel] *
                       ggml_fp16_to_fp32(key[key_token * kQwenFlashHeadSize + channel]);
            }
            const float score = dot * scale + ggml_fp16_to_fp32(mask[query_token * key_value_token_count + key_token]);
            scores[key_token] = score;
            max_score         = std::max(max_score, score);
        }

        float sum = 0.0f;
        for (float & score : scores) {
            score = std::exp(score - max_score);
            sum += score;
        }
        for (int64_t channel = 0; channel < kQwenFlashHeadSize; ++channel) {
            float weighted_sum = 0.0f;
            for (int64_t key_token = 0; key_token < key_value_token_count; ++key_token) {
                const float probability = scores[key_token] / sum;
                weighted_sum += probability * ggml_fp16_to_fp32(value[key_token * kQwenFlashHeadSize + channel]);
            }
            output[query_token * kQwenFlashHeadSize + channel] = weighted_sum;
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

static ggml_tensor * build_qwen_flash_attention_graph(ggml_context * ctx,
                                                      ggml_tensor *  query,
                                                      ggml_tensor *  key,
                                                      ggml_tensor *  value,
                                                      ggml_tensor *  mask) {
    ggml_tensor * output = ggml_flash_attn_ext(ctx, query, key, value, mask,
                                               1.0f / std::sqrt(static_cast<float>(kQwenFlashHeadSize)), 0.0f, 0.0f);
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

static void run_router_projection_case(int64_t token_count) {
    static constexpr int64_t kHiddenSize  = 2048;
    static constexpr int64_t kExpertCount = 128;

    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = static_cast<size_t>(
        (kHiddenSize * token_count + kHiddenSize * kExpertCount + kExpertCount * token_count) * sizeof(float) * 4 +
        1024 * 1024);
    params.no_alloc    = true;
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHiddenSize, kExpertCount);
    ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHiddenSize, token_count);
    REQUIRE(weight != nullptr);
    REQUIRE(input != nullptr);
    ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
    REQUIRE(output != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);

    const std::vector<float> input_data  = make_router_input(kHiddenSize, token_count);
    const std::vector<float> weight_data = make_router_weight(kHiddenSize, kExpertCount);
    const std::vector<float> expected =
        router_projection_reference(input_data, weight_data, kHiddenSize, kExpertCount, token_count);

    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    for (size_t i = 0; i < actual.size(); ++i) {
        const float diff = std::fabs(actual[i] - expected[i]);
        REQUIRE(diff <= 1.0e-2f);
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static void run_qwen_flash_attention_case() {
    static constexpr int64_t kQueryTokenCount    = 2;
    static constexpr int64_t kKeyValueTokenCount = 4;

    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = 2 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * query = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kQwenFlashHeadSize, kQueryTokenCount, 1);
    ggml_tensor * key   = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kQwenFlashHeadSize, kKeyValueTokenCount, 1);
    ggml_tensor * value = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kQwenFlashHeadSize, kKeyValueTokenCount, 1);
    ggml_tensor * mask  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kKeyValueTokenCount, kQueryTokenCount);
    REQUIRE(query != nullptr);
    REQUIRE(key != nullptr);
    REQUIRE(value != nullptr);
    REQUIRE(mask != nullptr);
    ggml_tensor * output = build_qwen_flash_attention_graph(ctx, query, key, value, mask);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);

    const std::vector<float>       query_data = make_flash_query(kQueryTokenCount);
    const std::vector<ggml_fp16_t> key_data   = make_flash_key_value(kKeyValueTokenCount, 3);
    const std::vector<ggml_fp16_t> value_data = make_flash_key_value(kKeyValueTokenCount, 11);
    const std::vector<ggml_fp16_t> mask_data  = make_flash_mask(kQueryTokenCount, kKeyValueTokenCount);
    const std::vector<float>       expected =
        flash_attention_reference(query_data, key_data, value_data, mask_data, kQueryTokenCount, kKeyValueTokenCount);

    ggml_backend_tensor_set(query, query_data.data(), 0, query_data.size() * sizeof(float));
    ggml_backend_tensor_set(key, key_data.data(), 0, key_data.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(value, value_data.data(), 0, value_data.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));

    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    for (size_t i = 0; i < actual.size(); ++i) {
        const float diff = std::fabs(actual[i] - expected[i]);
        REQUIRE(diff <= 5.0e-2f);
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
    run_router_projection_case(4);
    run_qwen_flash_attention_case();
    return 0;
}
