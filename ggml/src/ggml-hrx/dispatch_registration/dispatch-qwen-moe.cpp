#include "dispatch-qwen-moe.h"

#include "ggml.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kQwenRoutedGateUpSwiGLUQ4KF16WmmaKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_routed_gate_up_swiglu_q4k_f16_wmma");

static constexpr int64_t      kQwenMoeInputSize              = 2048;
static constexpr int64_t      kQwenMoeOutputSize             = 768;
static constexpr int64_t      kQwenMoeExpertCount            = 128;
static constexpr int64_t      kQwenMoeRouteCount             = 8;
static constexpr size_t       kQwenMoePlanTransientAlignment = 256;
static constexpr const char * kQwenMoeF16GateUpOutputName    = "qwen.moe.gate_up_swiglu_f16";

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool is_shape(const Value & value, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    return value.ne[0] == ne0 && value.ne[1] == ne1 && value.ne[2] == ne2 && value.ne[3] == ne3;
}

static bool same_shape(const Value & lhs, const Value & rhs) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (lhs.ne[i] != rhs.ne[i]) {
            return false;
        }
    }
    return true;
}

static bool is_supported_token_count(int64_t token_count) {
    return token_count >= 1 && token_count <= 2048;
}

static bool is_qwen_routed_weight(const Value & value) {
    return value.type == GGML_TYPE_Q4_K && value.contiguous &&
           is_shape(value, kQwenMoeInputSize, kQwenMoeOutputSize, kQwenMoeExpertCount, 1);
}

static bool is_qwen_routed_projection_output(const Value & value, int64_t token_count) {
    return value.type == GGML_TYPE_F32 && value.contiguous &&
           is_shape(value, kQwenMoeOutputSize, kQwenMoeRouteCount, token_count, 1);
}

static const GraphNode * find_consumer_with_op(const Graph & graph, ValueId value, ggml_op op) {
    for (const GraphNode * consumer : graph.index().consumers(value)) {
        if (consumer != nullptr && consumer->op == op) {
            return consumer;
        }
    }
    return nullptr;
}

static const GraphNode * producer_with_op(const Graph & graph, ValueId value, ggml_op op) {
    const GraphNode * producer = graph.index().producer(value);
    return producer != nullptr && producer->op == op ? producer : nullptr;
}

static bool node_index(const Graph & graph, const GraphNode * node, size_t & index) {
    return node != nullptr && graph.index().node_index(node, index);
}

static bool append_covered_node(const DispatchMatchContext & context, const GraphNode * node, DispatchMatch & match) {
    size_t index = 0;
    if (!node_index(context.graph, node, index) || index >= context.covered_nodes.size() ||
        context.covered_nodes[index]) {
        return false;
    }
    for (const size_t covered : match.covered_nodes) {
        if (covered == index) {
            return true;
        }
    }
    match.covered_nodes.push_back(index);
    return true;
}

static std::string to_config_value(int64_t value) {
    return std::to_string(value);
}

static size_t expert_table_size(int64_t token_count) {
    return static_cast<size_t>(kQwenMoeExpertCount + kQwenMoeExpertCount * token_count) * sizeof(int32_t);
}

static size_t partition_table_size(int64_t token_count) {
    const int64_t assignment_count           = token_count * kQwenMoeRouteCount;
    const int64_t assignment_partition_count = (assignment_count + 31) / 32;
    return static_cast<size_t>(1 + assignment_partition_count + kQwenMoeExpertCount) * sizeof(int32_t);
}

static size_t f16_gate_up_output_size(int64_t token_count) {
    return static_cast<size_t>(token_count * kQwenMoeRouteCount * kQwenMoeOutputSize) * sizeof(ggml_fp16_t);
}

struct RoutedGateUpMatch {
    const Value *                        gate_weight              = nullptr;
    const Value *                        up_weight                = nullptr;
    const Value *                        input                    = nullptr;
    const Value *                        route_ids                = nullptr;
    const Value *                        gate_output              = nullptr;
    const Value *                        up_output                = nullptr;
    const Value *                        glu_output               = nullptr;
    const CommandPlanGeneratedResource * expert_table_resource    = nullptr;
    const CommandPlanGeneratedResource * partition_table_resource = nullptr;
    const GraphNode *                    gate_node                = nullptr;
    const GraphNode *                    up_node                  = nullptr;
    const GraphNode *                    glu_node                 = nullptr;
    int64_t                              token_count              = 0;

    bool matched() const {
        return gate_weight != nullptr && up_weight != nullptr && input != nullptr && route_ids != nullptr &&
               gate_output != nullptr && up_output != nullptr && glu_output != nullptr &&
               expert_table_resource != nullptr && partition_table_resource != nullptr && gate_node != nullptr &&
               up_node != nullptr && glu_node != nullptr && token_count > 0;
    }
};

static bool resource_matches_qwen_router(const CommandPlanGeneratedResource & resource,
                                         ValueId                              route_ids,
                                         int64_t                              token_count,
                                         size_t                               expected_byte_count) {
    QwenMoeRoutingResourceMetadata metadata;
    return resource.source_value == route_ids && resource.generated_value.value >= 0 &&
           resource.byte_count == expected_byte_count && resource.metadata.read(metadata) &&
           metadata.token_count == token_count && metadata.route_count == kQwenMoeRouteCount &&
           metadata.expert_count == kQwenMoeExpertCount && metadata.route_stride >= kQwenMoeRouteCount;
}

static bool match_same_route_projection(const Graph &     graph,
                                        const GraphNode & node,
                                        ValueId           expected_input,
                                        ValueId           expected_route_ids,
                                        int64_t           token_count,
                                        const Value *&    weight,
                                        const Value *&    output) {
    if (node.op != GGML_OP_MUL_MAT_ID || node.inputs.size() != 3 || node.inputs[1] != expected_input ||
        node.inputs[2] != expected_route_ids) {
        return false;
    }
    weight = graph_value(graph, node.inputs[0]);
    output = graph_value(graph, node.output);
    return weight != nullptr && output != nullptr && is_qwen_routed_weight(*weight) &&
           is_qwen_routed_projection_output(*output, token_count);
}

static RoutedGateUpMatch match_qwen_routed_gate_up_swiglu(const DispatchMatchContext & context) {
    RoutedGateUpMatch match;
    const GraphNode * root = context.root_node;
    if (root == nullptr || root->op != GGML_OP_MUL_MAT_ID || root->inputs.size() != 3 || !context.graph.has_index()) {
        return match;
    }

    const Value * root_weight = graph_value(context.graph, root->inputs[0]);
    const Value * input       = graph_value(context.graph, root->inputs[1]);
    const Value * route_ids   = graph_value(context.graph, root->inputs[2]);
    const Value * root_output = graph_value(context.graph, root->output);
    if (root_weight == nullptr || input == nullptr || route_ids == nullptr || root_output == nullptr ||
        !is_qwen_routed_weight(*root_weight) || input->type != GGML_TYPE_F32 || !input->contiguous ||
        !is_shape(*input, kQwenMoeInputSize, 1, input->ne[2], 1) || route_ids->type != GGML_TYPE_I32 ||
        !is_shape(*route_ids, kQwenMoeRouteCount, input->ne[2], 1, 1)) {
        return {};
    }

    const int64_t token_count = input->ne[2];
    if (!is_supported_token_count(token_count) || !is_qwen_routed_projection_output(*root_output, token_count)) {
        return {};
    }

    const CommandPlanGeneratedResource * expert_table_resource =
        context.plan.metadata.find_generated_resource(route_ids->id, GeneratedResourceRole::QwenMoeExpertTable);
    const CommandPlanGeneratedResource * partition_table_resource =
        context.plan.metadata.find_generated_resource(route_ids->id, GeneratedResourceRole::QwenMoePartitionTable);
    if (expert_table_resource == nullptr || partition_table_resource == nullptr ||
        !resource_matches_qwen_router(*expert_table_resource, route_ids->id, token_count,
                                      expert_table_size(token_count)) ||
        !resource_matches_qwen_router(*partition_table_resource, route_ids->id, token_count,
                                      partition_table_size(token_count))) {
        return {};
    }

    const GraphNode * glu_node = find_consumer_with_op(context.graph, root->output, GGML_OP_GLU);
    if (glu_node == nullptr || glu_node->inputs.size() != 2) {
        return {};
    }
    const GluParams * glu_params = op_params_as<GluParams>(glu_node->params);
    if (glu_params == nullptr || glu_params->op != GGML_GLU_OP_SWIGLU) {
        return {};
    }

    const GraphNode * gate_node = producer_with_op(context.graph, glu_node->inputs[0], GGML_OP_MUL_MAT_ID);
    const GraphNode * up_node   = producer_with_op(context.graph, glu_node->inputs[1], GGML_OP_MUL_MAT_ID);
    if (gate_node == nullptr || up_node == nullptr || gate_node == up_node || (gate_node != root && up_node != root)) {
        return {};
    }

    const Value * gate_weight = nullptr;
    const Value * gate_output = nullptr;
    const Value * up_weight   = nullptr;
    const Value * up_output   = nullptr;
    if (!match_same_route_projection(context.graph, *gate_node, input->id, route_ids->id, token_count, gate_weight,
                                     gate_output) ||
        !match_same_route_projection(context.graph, *up_node, input->id, route_ids->id, token_count, up_weight,
                                     up_output)) {
        return {};
    }
    if (!same_shape(*gate_output, *up_output)) {
        return {};
    }

    const Value * glu_output = graph_value(context.graph, glu_node->output);
    if (glu_output == nullptr || glu_output->kind != ValueKind::Transient ||
        !is_qwen_routed_projection_output(*glu_output, token_count)) {
        return {};
    }

    match.gate_weight              = gate_weight;
    match.up_weight                = up_weight;
    match.input                    = input;
    match.route_ids                = route_ids;
    match.gate_output              = gate_output;
    match.up_output                = up_output;
    match.glu_output               = glu_output;
    match.expert_table_resource    = expert_table_resource;
    match.partition_table_resource = partition_table_resource;
    match.gate_node                = gate_node;
    match.up_node                  = up_node;
    match.glu_node                 = glu_node;
    match.token_count              = token_count;
    return match;
}

static bool match_qwen_routed_gate_up_swiglu_q4k_f16_wmma_dispatch(const DispatchMatchContext & context,
                                                                   DispatchMatch &              dispatch_match) {
    const RoutedGateUpMatch match = match_qwen_routed_gate_up_swiglu(context);
    if (!match.matched()) {
        return false;
    }

    const ValueId f16_output(context.next_plan_value.value);
    const size_t  f16_output_bytes = f16_gate_up_output_size(match.token_count);
    dispatch_match.transients.push_back(
        { f16_output, kQwenMoeF16GateUpOutputName, f16_output_bytes, kQwenMoePlanTransientAlignment });
    Status metadata_status;
    if (!dispatch_match.metadata.append_alternate_value(
            { match.glu_output->id, f16_output, GGML_TYPE_F16, f16_output_bytes, kQwenMoeF16GateUpOutputName },
            metadata_status)) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kQwenRoutedGateUpSwiGLUQ4KF16WmmaKernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.routed_gate_up.input_size",
                                               to_config_value(kQwenMoeInputSize));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.routed_gate_up.expert_count",
                                               to_config_value(kQwenMoeExpertCount));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.routed_gate_up.route_count",
                                               to_config_value(kQwenMoeRouteCount));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.routed_gate_up.output_size",
                                               to_config_value(kQwenMoeOutputSize));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.workload.token_capacity", to_config_value(match.token_count));

    dispatch.bindings.push_back({ match.input->id, 0, match.input->byte_count });
    dispatch.bindings.push_back(
        { match.expert_table_resource->generated_value, 0, match.expert_table_resource->byte_count });
    dispatch.bindings.push_back(
        { match.partition_table_resource->generated_value, 0, match.partition_table_resource->byte_count });
    dispatch.bindings.push_back({ match.gate_weight->id, 0, match.gate_weight->byte_count });
    dispatch.bindings.push_back({ match.up_weight->id, 0, match.up_weight->byte_count });
    dispatch.bindings.push_back({ f16_output, 0, f16_output_bytes });

    if (!append_covered_node(context, match.gate_node, dispatch_match) ||
        !append_covered_node(context, match.up_node, dispatch_match) ||
        !append_covered_node(context, match.glu_node, dispatch_match)) {
        return false;
    }
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_qwen_moe_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "qwen.moe.routed_gate_up_swiglu_q4k_f16_wmma",
        GGML_OP_MUL_MAT_ID,
        DispatchMatchKind::Fused,
        1000,
        DispatchSource::Qwen,
        match_qwen_routed_gate_up_swiglu_q4k_f16_wmma_dispatch,
    });
}

}  // namespace ggml::hrx
