#include "dispatch-rmsnorm.h"

#include "ggml.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kQwenRmsNormF32Kernel = GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_rmsnorm_f32");
static constexpr KernelCatalogRef kQwenRmsNormF32QuantizeQ8_1X4Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_rmsnorm_f32_quantize_q8_1_x4");
static constexpr float   kQwenRmsNormEpsilon  = 0.000001f;
static constexpr int64_t kQwenHiddenSize      = 2048;
static constexpr int64_t kQwenVocabularyCount = 151936;

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool same_shape(const Value & lhs, const Value & rhs) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (lhs.ne[i] != rhs.ne[i]) {
            return false;
        }
    }
    return true;
}

static bool is_qwen_rms_norm_epsilon(float eps) {
    return std::fabs(eps - kQwenRmsNormEpsilon) <= 1.0e-12f;
}

static bool is_supported_hidden_size(int64_t hidden_size) {
    return hidden_size >= 128 && hidden_size <= 32768 && hidden_size % 128 == 0;
}

static bool is_supported_token_count(int64_t token_count) {
    return token_count >= 1 && token_count <= 2048;
}

static size_t q8_1_x4_byte_count(int64_t token_count, int64_t hidden_size) {
    if (token_count <= 0 || hidden_size <= 0) {
        return 0;
    }
    return static_cast<size_t>(token_count) * ggml_row_size(GGML_TYPE_Q8_1, hidden_size);
}

static bool is_weight_shape(const Value & weight, int64_t hidden_size) {
    if (weight.ne[0] != hidden_size) {
        return false;
    }
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (weight.ne[i] != 1) {
            return false;
        }
    }
    return true;
}

struct RmsNormMatch {
    const GraphNode * rms_node       = nullptr;
    const GraphNode * mul_node       = nullptr;
    const Value *     input          = nullptr;
    const Value *     weight         = nullptr;
    const Value *     output         = nullptr;
    size_t            rms_node_index = 0;
    size_t            mul_node_index = 0;
    int64_t           hidden_size    = 0;
    int64_t           token_count    = 0;
    int64_t           q8_group_count = 0;

    bool matched() const {
        return rms_node != nullptr && mul_node != nullptr && input != nullptr && weight != nullptr && output != nullptr;
    }
};

static RmsNormMatch match_qwen_rmsnorm_f32(const Graph & graph, const GraphNode * node, size_t node_index) {
    RmsNormMatch match;
    if (node == nullptr || node->op != GGML_OP_RMS_NORM || node->inputs.size() != 1 || !graph.has_index()) {
        return match;
    }

    const RmsNormParams * rms_params = op_params_as<RmsNormParams>(node->params);
    if (rms_params == nullptr || !is_qwen_rms_norm_epsilon(rms_params->eps)) {
        return {};
    }
    const std::vector<const GraphNode *> & consumers = graph.index().consumers(node->output);
    if (consumers.size() != 1) {
        return {};
    }
    const GraphNode * mul_node = consumers.front();
    size_t            mul_node_index;
    if (mul_node == nullptr || mul_node->op != GGML_OP_MUL || mul_node->inputs.size() != 2 ||
        !graph.index().node_index(mul_node, mul_node_index)) {
        return {};
    }

    const Value * weight = nullptr;
    for (ValueId input : mul_node->inputs) {
        if (input != node->output) {
            weight = graph_value(graph, input);
        }
    }
    const Value * input  = graph_value(graph, node->inputs[0]);
    const Value * rms    = graph_value(graph, node->output);
    const Value * output = graph_value(graph, mul_node->output);
    if (input == nullptr || rms == nullptr || weight == nullptr || output == nullptr) {
        return {};
    }
    if (input->type != GGML_TYPE_F32 || rms->type != GGML_TYPE_F32 || weight->type != GGML_TYPE_F32 ||
        output->type != GGML_TYPE_F32) {
        return {};
    }
    if (!input->contiguous || !rms->contiguous || !weight->contiguous || !output->contiguous) {
        return {};
    }
    if (!same_shape(*input, *rms) || !same_shape(*input, *output)) {
        return {};
    }

    const int64_t hidden_size = output->ne[0];
    if (!is_supported_hidden_size(hidden_size) || !is_weight_shape(*weight, hidden_size)) {
        return {};
    }
    if (hidden_size == 0 || output->element_count <= 0 || output->element_count % hidden_size != 0) {
        return {};
    }
    const int64_t token_count = output->element_count / hidden_size;
    if (!is_supported_token_count(token_count)) {
        return {};
    }

    match.rms_node       = node;
    match.mul_node       = mul_node;
    match.input          = input;
    match.weight         = weight;
    match.output         = output;
    match.rms_node_index = node_index;
    match.mul_node_index = mul_node_index;
    match.hidden_size    = hidden_size;
    match.token_count    = token_count;
    match.q8_group_count = token_count * ((hidden_size + 127) / 128);
    return match;
}

static bool has_qwen_q6k_q8_consumer(const Graph & graph, const RmsNormMatch & match) {
    if (!graph.has_index()) {
        return false;
    }
    const std::vector<const GraphNode *> & consumers = graph.index().consumers(match.output->id);
    if (consumers.size() != 1) {
        return false;
    }
    const GraphNode * consumer = consumers.front();
    if (consumer == nullptr || consumer->op != GGML_OP_MUL_MAT || consumer->inputs.size() != 2) {
        return false;
    }
    const Value * weight = graph_value(graph, consumer->inputs[0]);
    const Value * input  = graph_value(graph, consumer->inputs[1]);
    const Value * output = graph_value(graph, consumer->output);
    if (weight == nullptr || input == nullptr || output == nullptr) {
        return false;
    }
    return input->id == match.output->id && weight->type == GGML_TYPE_Q6_K && output->type == GGML_TYPE_F32 &&
           weight->contiguous && input->contiguous && output->contiguous && match.hidden_size == kQwenHiddenSize &&
           match.token_count == 1 && weight->ne[0] == match.hidden_size && weight->ne[1] == kQwenVocabularyCount &&
           output->ne[0] == kQwenVocabularyCount && output->ne[1] == match.token_count;
}

static std::string to_config_value(int64_t value) {
    return std::to_string(value);
}

}  // namespace

static bool match_qwen_rmsnorm_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const std::vector<GraphNode> & nodes = context.graph.nodes();
    if (context.root_index >= nodes.size()) {
        return false;
    }
    const RmsNormMatch rms_match =
        match_qwen_rmsnorm_f32(context.graph, &nodes[context.root_index], context.root_index);
    if (!rms_match.matched() || rms_match.rms_node_index >= context.covered_nodes.size() ||
        rms_match.mul_node_index >= context.covered_nodes.size() || context.covered_nodes[rms_match.rms_node_index] ||
        context.covered_nodes[rms_match.mul_node_index]) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kQwenRmsNormF32Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", rms_match.token_count);
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.model.hidden_size", to_config_value(rms_match.hidden_size));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.model.rms_epsilon", "0.000001");
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.workload.token_capacity",
                                               to_config_value(rms_match.token_count));
    dispatch.kernel.compile_parameters.emplace("ggml.quantize_q8_1_x4.group_capacity",
                                               to_config_value(rms_match.q8_group_count));
    dispatch.bindings.push_back({ rms_match.input->id, 0, rms_match.input->byte_count });
    dispatch.bindings.push_back({ rms_match.weight->id, 0, rms_match.weight->byte_count });
    dispatch.bindings.push_back({ rms_match.output->id, 0, rms_match.output->byte_count });

    match.covered_nodes.push_back(rms_match.rms_node_index);
    match.covered_nodes.push_back(rms_match.mul_node_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_qwen_rmsnorm_f32_quantize_q8_1_x4_dispatch(const DispatchMatchContext & context,
                                                             DispatchMatch &              match) {
    const std::vector<GraphNode> & nodes = context.graph.nodes();
    if (context.root_index >= nodes.size()) {
        return false;
    }
    const RmsNormMatch rms_match =
        match_qwen_rmsnorm_f32(context.graph, &nodes[context.root_index], context.root_index);
    if (!rms_match.matched() || rms_match.rms_node_index >= context.covered_nodes.size() ||
        rms_match.mul_node_index >= context.covered_nodes.size() || context.covered_nodes[rms_match.rms_node_index] ||
        context.covered_nodes[rms_match.mul_node_index] || !has_qwen_q6k_q8_consumer(context.graph, rms_match)) {
        return false;
    }

    const size_t q8_byte_count = q8_1_x4_byte_count(rms_match.token_count, rms_match.hidden_size);
    if (q8_byte_count == 0) {
        return false;
    }

    const ValueId q8_value = context.next_plan_value;

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kQwenRmsNormF32QuantizeQ8_1X4Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", rms_match.token_count);
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.model.hidden_size", to_config_value(rms_match.hidden_size));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.model.rms_epsilon", "0.000001");
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.workload.token_capacity",
                                               to_config_value(rms_match.token_count));
    dispatch.kernel.compile_parameters.emplace("ggml.quantize_q8_1_x4.group_capacity",
                                               to_config_value(rms_match.q8_group_count));
    dispatch.bindings.push_back({ rms_match.input->id, 0, rms_match.input->byte_count });
    dispatch.bindings.push_back({ rms_match.weight->id, 0, rms_match.weight->byte_count });
    dispatch.bindings.push_back({ rms_match.output->id, 0, rms_match.output->byte_count });
    dispatch.bindings.push_back({ q8_value, 0, q8_byte_count });

    match.covered_nodes.push_back(rms_match.rms_node_index);
    match.covered_nodes.push_back(rms_match.mul_node_index);
    match.dispatches.push_back(std::move(dispatch));
    match.transients.push_back({ q8_value, "qwen.rmsnorm.q8_1_x4", q8_byte_count, 256 });
    match.metadata.append_alternate_value(
        { rms_match.output->id, q8_value, GGML_TYPE_Q8_1, q8_byte_count, "qwen.rmsnorm.q8_1_x4" }, match.status);
    return match.status.success();
}

void register_qwen_rmsnorm_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "qwen.rmsnorm_f32_quantize_q8_1_x4",
        GGML_OP_RMS_NORM,
        DispatchMatchKind::Fused,
        1100,
        DispatchSource::Qwen,
        match_qwen_rmsnorm_f32_quantize_q8_1_x4_dispatch,
    });
    registry.add({
        "qwen.rmsnorm_f32.mul_weight",
        GGML_OP_RMS_NORM,
        DispatchMatchKind::Fused,
        1000,
        DispatchSource::Qwen,
        match_qwen_rmsnorm_f32_dispatch,
    });
}

}  // namespace ggml::hrx
