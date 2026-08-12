#include "dispatch-qwen-matmul.h"

#include "ggml.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kQwenDenseLinearQ4KF16WmmaKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_q4k_f16_wmma");
static constexpr KernelCatalogRef kQwenDenseLinearQ6KF16WmmaKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_q6k_f16_wmma");
static constexpr KernelCatalogRef kGgmlLinearQ6KQ8_1X4Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_linear_q6k_q8_1_x4");
static constexpr KernelCatalogRef kQwenRouterProjectionF32FourRowWave32Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_router_projection_f32_four_row_wave32");

static constexpr int64_t kQwenHiddenSize        = 2048;
static constexpr int64_t kQwenRouterExpertCount = 128;
static constexpr int64_t kQwenVocabularyCount   = 151936;

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool is_2d(const Value & value) {
    return value.ne[0] > 0 && value.ne[1] > 0 && value.ne[2] == 1 && value.ne[3] == 1;
}

static bool is_supported_token_count(int64_t token_count) {
    return token_count >= 1 && token_count <= 2048;
}

static bool is_supported_dense_input_size(int64_t input_size) {
    return input_size >= 256 && input_size <= 32768 && input_size % 256 == 0;
}

static bool is_supported_dense_output_size(int64_t output_size) {
    return output_size >= 1 && output_size <= 262144;
}

static std::string to_config_value(int64_t value) {
    return std::to_string(value);
}

struct QwenMatmulMatch {
    const Value *    input       = nullptr;
    const Value *    weight      = nullptr;
    const Value *    output      = nullptr;
    KernelCatalogRef kernel      = {};
    ValueId          input_value = {};
    size_t           input_bytes = 0;
    int64_t          input_size  = 0;
    int64_t          output_size = 0;
    int64_t          token_count = 0;
    bool             dense       = false;
    bool             router      = false;

    bool matched() const {
        return input != nullptr && weight != nullptr && output != nullptr && kernel.id != kUncatalogedKernelId;
    }
};

enum class QwenMatmulRoute {
    DenseQ4K,
    DenseQ6K,
    RouterF32,
};

static size_t q8_1_x4_byte_count(int64_t token_count, int64_t input_size) {
    if (token_count <= 0 || input_size <= 0) {
        return 0;
    }
    return static_cast<size_t>(token_count) * ggml_row_size(GGML_TYPE_Q8_1, input_size);
}

static QwenMatmulMatch match_qwen_matmul(const Graph & graph, const GraphNode * node, QwenMatmulRoute route) {
    QwenMatmulMatch match;
    if (node == nullptr || node->op != GGML_OP_MUL_MAT || node->inputs.size() != 2) {
        return match;
    }

    const Value * weight = graph_value(graph, node->inputs[0]);
    const Value * input  = graph_value(graph, node->inputs[1]);
    const Value * output = graph_value(graph, node->output);
    if (weight == nullptr || input == nullptr || output == nullptr) {
        return {};
    }
    if (!is_2d(*weight) || !is_2d(*input) || !is_2d(*output)) {
        return {};
    }
    if (!weight->contiguous || !input->contiguous || !output->contiguous) {
        return {};
    }
    if (input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32) {
        return {};
    }

    const int64_t input_size  = weight->ne[0];
    const int64_t output_size = weight->ne[1];
    const int64_t token_count = input->ne[1];
    if (input->ne[0] != input_size || output->ne[0] != output_size || output->ne[1] != token_count) {
        return {};
    }
    if (!is_supported_token_count(token_count)) {
        return {};
    }

    if ((route == QwenMatmulRoute::DenseQ4K && weight->type == GGML_TYPE_Q4_K) ||
        (route == QwenMatmulRoute::DenseQ6K && weight->type == GGML_TYPE_Q6_K)) {
        if (!is_supported_dense_input_size(input_size) || !is_supported_dense_output_size(output_size)) {
            return {};
        }
        match.input       = input;
        match.weight      = weight;
        match.output      = output;
        match.input_value = input->id;
        match.input_bytes = input->byte_count;
        match.kernel =
            weight->type == GGML_TYPE_Q4_K ? kQwenDenseLinearQ4KF16WmmaKernel : kQwenDenseLinearQ6KF16WmmaKernel;
        match.input_size  = input_size;
        match.output_size = output_size;
        match.token_count = token_count;
        match.dense       = true;
        return match;
    }

    if (route == QwenMatmulRoute::RouterF32 && weight->type == GGML_TYPE_F32 && input_size == kQwenHiddenSize &&
        output_size == kQwenRouterExpertCount) {
        match.input       = input;
        match.weight      = weight;
        match.output      = output;
        match.input_value = input->id;
        match.input_bytes = input->byte_count;
        match.kernel      = kQwenRouterProjectionF32FourRowWave32Kernel;
        match.input_size  = input_size;
        match.output_size = output_size;
        match.token_count = token_count;
        match.router      = true;
        return match;
    }

    return {};
}

static QwenMatmulMatch match_qwen_q6k_q8_matmul(const Graph & graph, const GraphNode * node, const CommandPlan & plan) {
    QwenMatmulMatch match;
    if (node == nullptr || node->op != GGML_OP_MUL_MAT || node->inputs.size() != 2) {
        return match;
    }

    const Value * weight = graph_value(graph, node->inputs[0]);
    const Value * input  = graph_value(graph, node->inputs[1]);
    const Value * output = graph_value(graph, node->output);
    if (weight == nullptr || input == nullptr || output == nullptr) {
        return {};
    }
    if (!is_2d(*weight) || !is_2d(*input) || !is_2d(*output)) {
        return {};
    }
    if (!weight->contiguous || !input->contiguous || !output->contiguous) {
        return {};
    }
    if (weight->type != GGML_TYPE_Q6_K || input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32) {
        return {};
    }

    const int64_t input_size  = weight->ne[0];
    const int64_t output_size = weight->ne[1];
    const int64_t token_count = input->ne[1];
    if (input->ne[0] != input_size || output->ne[0] != output_size || output->ne[1] != token_count) {
        return {};
    }
    if (token_count != 1 || input_size != kQwenHiddenSize || output_size != kQwenVocabularyCount ||
        !is_supported_token_count(token_count) || !is_supported_dense_input_size(input_size) ||
        !is_supported_dense_output_size(output_size)) {
        return {};
    }

    const size_t                      q8_byte_count = q8_1_x4_byte_count(token_count, input_size);
    const CommandPlanAlternateValue * alternate = find_alternate_value(plan, input->id, GGML_TYPE_Q8_1, q8_byte_count);
    if (alternate == nullptr) {
        return {};
    }

    match.input       = input;
    match.weight      = weight;
    match.output      = output;
    match.input_value = alternate->alternate_value;
    match.input_bytes = alternate->byte_count;
    match.kernel      = kGgmlLinearQ6KQ8_1X4Kernel;
    match.input_size  = input_size;
    match.output_size = output_size;
    match.token_count = token_count;
    return match;
}

}  // namespace

static void build_qwen_matmul_dispatch(const QwenMatmulMatch & match,
                                       DispatchMatch &         dispatch_match,
                                       size_t                  root_index) {
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(match.kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    if (match.kernel.id == kGgmlLinearQ6KQ8_1X4Kernel.id) {
        dispatch.kernel.integer_parameters.emplace("input_size", match.input_size);
        dispatch.kernel.integer_parameters.emplace("output_size", match.output_size);
    } else {
        dispatch.kernel.compile_parameters.emplace("qwen3_moe.workload.token_capacity",
                                                   to_config_value(match.token_count));
    }
    if (match.dense) {
        dispatch.kernel.compile_parameters.emplace("qwen3_moe.dense_quantized.input_size",
                                                   to_config_value(match.input_size));
        dispatch.kernel.compile_parameters.emplace("qwen3_moe.dense_quantized.output_size",
                                                   to_config_value(match.output_size));
        dispatch.kernel.compile_parameters.emplace("qwen3_moe.dense_quantized.output_accumulation", "0");
    } else if (match.router) {
        dispatch.kernel.compile_parameters.emplace("qwen3_moe.model.hidden_size", to_config_value(match.input_size));
        dispatch.kernel.compile_parameters.emplace("qwen3_moe.router.expert_count", to_config_value(match.output_size));
    }
    dispatch.bindings.push_back({ match.input_value, 0, match.input_bytes });
    dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });

    dispatch_match.covered_nodes.push_back(root_index);
    dispatch_match.dispatches.push_back(std::move(dispatch));
}

static bool match_qwen_dense_q4k_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const QwenMatmulMatch match = match_qwen_matmul(context.graph, context.root_node, QwenMatmulRoute::DenseQ4K);
    if (!match.matched()) {
        return false;
    }
    build_qwen_matmul_dispatch(match, dispatch_match, context.root_index);
    return true;
}

static bool match_qwen_dense_q6k_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const QwenMatmulMatch match = match_qwen_matmul(context.graph, context.root_node, QwenMatmulRoute::DenseQ6K);
    if (!match.matched()) {
        return false;
    }
    build_qwen_matmul_dispatch(match, dispatch_match, context.root_index);
    return true;
}

static bool match_qwen_router_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const QwenMatmulMatch match = match_qwen_matmul(context.graph, context.root_node, QwenMatmulRoute::RouterF32);
    if (!match.matched()) {
        return false;
    }
    build_qwen_matmul_dispatch(match, dispatch_match, context.root_index);
    return true;
}

static bool match_qwen_q6k_q8_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const QwenMatmulMatch match = match_qwen_q6k_q8_matmul(context.graph, context.root_node, context.plan);
    if (!match.matched()) {
        return false;
    }
    build_qwen_matmul_dispatch(match, dispatch_match, context.root_index);
    return true;
}

void register_qwen_matmul_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "qwen.matmul.q6k_q8_1_x4",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::SingleOp,
        200,
        DispatchSource::Qwen,
        match_qwen_q6k_q8_dispatch,
    });
    registry.add({
        "qwen.matmul.dense_q4k_f16_wmma",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::SingleOp,
        100,
        DispatchSource::Qwen,
        match_qwen_dense_q4k_dispatch,
    });
    registry.add({
        "qwen.matmul.dense_q6k_f16_wmma",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::SingleOp,
        100,
        DispatchSource::Qwen,
        match_qwen_dense_q6k_dispatch,
    });
    registry.add({
        "qwen.matmul.router_projection_f32_four_row_wave32",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::SingleOp,
        90,
        DispatchSource::Qwen,
        match_qwen_router_f32_dispatch,
    });
}

}  // namespace ggml::hrx
