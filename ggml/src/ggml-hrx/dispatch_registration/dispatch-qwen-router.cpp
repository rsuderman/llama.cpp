#include "dispatch-qwen-router.h"

#include "ggml.h"
#include "graph/graph-matcher.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kQwenRouterTop8F32Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_router_top8_f32");
static constexpr KernelCatalogRef kQwenBuildExpertTableKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_build_expert_table");
static constexpr KernelCatalogRef kQwenBuildExpertPartitionTableKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_build_expert_partition_table");

static constexpr int64_t kQwenRouterExpertCount            = 128;
static constexpr int64_t kQwenRouterRouteCount             = 8;
static constexpr size_t  kQwenRouterPlanTransientAlignment = 256;

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool nearly_equal(float lhs, float rhs) {
    return std::fabs(lhs - rhs) <= 1.0e-12f;
}

static const GraphNode * find_consumer_with_op(const Graph & graph, ValueId value, ggml_op op) {
    const std::vector<const GraphNode *> consumers = consumers_with_op_through_layout_aliases(graph, value, op);
    return consumers.empty() ? nullptr : consumers.front();
}

static const GraphNode * find_consumer_with_op_and_input(const Graph & graph,
                                                         ValueId       value,
                                                         ggml_op       op,
                                                         ValueId       input) {
    for (const GraphNode * consumer : consumers_with_op_through_layout_aliases(graph, value, op)) {
        if (consumer != nullptr && node_has_input_or_alias(graph, *consumer, input)) {
            return consumer;
        }
    }
    return nullptr;
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

static bool is_qwen_softmax(const GraphNode & node) {
    const SoftMaxParams * params = op_params_as<SoftMaxParams>(node.params);
    return params != nullptr && nearly_equal(params->scale, 1.0f) && nearly_equal(params->max_bias, 0.0f);
}

static bool is_descending_argsort(const GraphNode & node) {
    const ArgsortParams * params = op_params_as<ArgsortParams>(node.params);
    return params != nullptr && params->order == GGML_SORT_ORDER_DESC;
}

static bool is_qwen_topk_clamp(const GraphNode & node) {
    const ClampParams * params = op_params_as<ClampParams>(node.params);
    return params != nullptr && params->min >= 0.0f && params->min <= 1.0e-6f && std::isinf(params->max) &&
           params->max > 0.0f;
}

static std::string to_config_value(int64_t value) {
    return std::to_string(value);
}

static size_t expert_table_size(int64_t token_count) {
    return static_cast<size_t>(kQwenRouterExpertCount + kQwenRouterExpertCount * token_count) * sizeof(int32_t);
}

static size_t partition_table_size(int64_t token_count) {
    const int64_t assignment_count           = token_count * kQwenRouterRouteCount;
    const int64_t assignment_partition_count = (assignment_count + 31) / 32;
    return static_cast<size_t>(1 + assignment_partition_count + kQwenRouterExpertCount) * sizeof(int32_t);
}

static bool append_covered_node(const DispatchMatchContext & context, const GraphNode * node, DispatchMatch & match) {
    return append_covered_node_index_once(context.graph, context.covered_nodes, node, match.covered_nodes);
}

struct RouterTop8Match {
    const Value * logits        = nullptr;
    const Value * route_ids     = nullptr;
    const Value * route_weights = nullptr;
    int64_t       token_count   = 0;
    int64_t       route_stride  = 0;

    bool matched() const {
        return logits != nullptr && route_ids != nullptr && route_weights != nullptr && token_count > 0 &&
               route_stride >= kQwenRouterRouteCount;
    }
};

static RouterTop8Match match_qwen_router_top8(const Graph & graph, const GraphNode * softmax_node) {
    RouterTop8Match match;
    if (softmax_node == nullptr || softmax_node->op != GGML_OP_SOFT_MAX || softmax_node->inputs.size() != 1 ||
        !graph.has_index() || !is_qwen_softmax(*softmax_node)) {
        return match;
    }

    const Value * logits = graph_value(graph, softmax_node->inputs[0]);
    const Value * probs  = graph_value(graph, softmax_node->output);
    if (logits == nullptr || probs == nullptr || logits->type != GGML_TYPE_F32 || probs->type != GGML_TYPE_F32 ||
        !same_shape(*logits, *probs) || !logits->contiguous || !probs->contiguous) {
        return {};
    }
    const int64_t token_count = logits->ne[1];
    if (!is_shape(*logits, kQwenRouterExpertCount, token_count, 1, 1) || !is_supported_token_count(token_count)) {
        return {};
    }

    const GraphNode * probs_reshape = find_consumer_with_op(graph, softmax_node->output, GGML_OP_RESHAPE);
    const GraphNode * argsort       = find_consumer_with_op(graph, softmax_node->output, GGML_OP_ARGSORT);
    if (probs_reshape == nullptr || argsort == nullptr || argsort->inputs.size() != 1 ||
        !is_descending_argsort(*argsort)) {
        return {};
    }

    const Value * probs_reshaped = graph_value(graph, probs_reshape->output);
    const Value * argsort_output = graph_value(graph, argsort->output);
    if (probs_reshaped == nullptr || argsort_output == nullptr || probs_reshaped->type != GGML_TYPE_F32 ||
        argsort_output->type != GGML_TYPE_I32 ||
        !is_shape(*probs_reshaped, 1, kQwenRouterExpertCount, token_count, 1) ||
        !is_shape(*argsort_output, kQwenRouterExpertCount, token_count, 1, 1)) {
        return {};
    }

    const GraphNode * topk_view = find_consumer_with_op(graph, argsort->output, GGML_OP_VIEW);
    if (topk_view == nullptr || topk_view->inputs.size() != 1) {
        return {};
    }
    const Value * route_ids = graph_value(graph, topk_view->output);
    if (route_ids == nullptr || route_ids->type != GGML_TYPE_I32 ||
        !is_shape(*route_ids, kQwenRouterRouteCount, token_count, 1, 1) || route_ids->nb[0] != sizeof(int32_t) ||
        route_ids->nb[1] % sizeof(int32_t) != 0) {
        return {};
    }
    const int64_t route_stride = static_cast<int64_t>(route_ids->nb[1] / sizeof(int32_t));
    if (route_stride < kQwenRouterRouteCount || route_stride > kQwenRouterExpertCount) {
        return {};
    }

    const GraphNode * get_rows =
        find_consumer_with_op_and_input(graph, probs_reshape->output, GGML_OP_GET_ROWS, topk_view->output);
    if (get_rows == nullptr || get_rows->inputs.size() != 2) {
        return {};
    }
    const Value * selected_weights = graph_value(graph, get_rows->output);
    if (selected_weights == nullptr || selected_weights->type != GGML_TYPE_F32 ||
        !is_shape(*selected_weights, 1, kQwenRouterRouteCount, token_count, 1)) {
        return {};
    }

    const GraphNode * weights_reshape = find_consumer_with_op(graph, get_rows->output, GGML_OP_RESHAPE);
    const Value *     weights_flat = weights_reshape == nullptr ? nullptr : graph_value(graph, weights_reshape->output);
    if (weights_flat == nullptr || weights_flat->type != GGML_TYPE_F32 ||
        !is_shape(*weights_flat, kQwenRouterRouteCount, token_count, 1, 1)) {
        return {};
    }

    const GraphNode * sum_rows = find_consumer_with_op(graph, weights_reshape->output, GGML_OP_SUM_ROWS);
    const Value *     sum      = sum_rows == nullptr ? nullptr : graph_value(graph, sum_rows->output);
    if (sum == nullptr || sum->type != GGML_TYPE_F32 || !is_shape(*sum, 1, token_count, 1, 1)) {
        return {};
    }

    const GraphNode * clamp       = find_consumer_with_op(graph, sum_rows->output, GGML_OP_CLAMP);
    const Value *     clamped_sum = clamp == nullptr ? nullptr : graph_value(graph, clamp->output);
    if (clamped_sum == nullptr || clamped_sum->type != GGML_TYPE_F32 || !is_shape(*clamped_sum, 1, token_count, 1, 1) ||
        !is_qwen_topk_clamp(*clamp)) {
        return {};
    }

    const GraphNode * div = find_consumer_with_op_and_input(graph, weights_reshape->output, GGML_OP_DIV, clamp->output);
    const Value *     normalized = div == nullptr ? nullptr : graph_value(graph, div->output);
    if (normalized == nullptr || normalized->type != GGML_TYPE_F32 ||
        !is_shape(*normalized, kQwenRouterRouteCount, token_count, 1, 1)) {
        return {};
    }

    const GraphNode * output_reshape = find_consumer_with_op(graph, div->output, GGML_OP_RESHAPE);
    const Value *     route_weights  = output_reshape == nullptr ? nullptr : graph_value(graph, output_reshape->output);
    if (route_weights == nullptr || route_weights->type != GGML_TYPE_F32 ||
        !is_shape(*route_weights, 1, kQwenRouterRouteCount, token_count, 1) || !route_weights->contiguous) {
        return {};
    }

    match.logits        = logits;
    match.route_ids     = route_ids;
    match.route_weights = route_weights;
    match.token_count   = token_count;
    match.route_stride  = route_stride;
    return match;
}

static bool append_qwen_router_top8_coverage(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * softmax       = context.root_node;
    const GraphNode * probs_reshape = find_consumer_with_op(context.graph, softmax->output, GGML_OP_RESHAPE);
    const GraphNode * argsort       = find_consumer_with_op(context.graph, softmax->output, GGML_OP_ARGSORT);
    const GraphNode * topk_view =
        argsort == nullptr ? nullptr : find_consumer_with_op(context.graph, argsort->output, GGML_OP_VIEW);
    const GraphNode * get_rows =
        probs_reshape == nullptr || topk_view == nullptr ?
            nullptr :
            find_consumer_with_op_and_input(context.graph, probs_reshape->output, GGML_OP_GET_ROWS, topk_view->output);
    const GraphNode * weights_reshape =
        get_rows == nullptr ? nullptr : find_consumer_with_op(context.graph, get_rows->output, GGML_OP_RESHAPE);
    const GraphNode * sum_rows = weights_reshape == nullptr ?
                                     nullptr :
                                     find_consumer_with_op(context.graph, weights_reshape->output, GGML_OP_SUM_ROWS);
    const GraphNode * clamp =
        sum_rows == nullptr ? nullptr : find_consumer_with_op(context.graph, sum_rows->output, GGML_OP_CLAMP);
    const GraphNode * div =
        weights_reshape == nullptr || clamp == nullptr ?
            nullptr :
            find_consumer_with_op_and_input(context.graph, weights_reshape->output, GGML_OP_DIV, clamp->output);
    const GraphNode * output_reshape =
        div == nullptr ? nullptr : find_consumer_with_op(context.graph, div->output, GGML_OP_RESHAPE);

    return append_covered_node(context, softmax, match) && append_covered_node(context, probs_reshape, match) &&
           append_covered_node(context, argsort, match) && append_covered_node(context, topk_view, match) &&
           append_covered_node(context, get_rows, match) && append_covered_node(context, weights_reshape, match) &&
           append_covered_node(context, sum_rows, match) && append_covered_node(context, clamp, match) &&
           append_covered_node(context, div, match) && append_covered_node(context, output_reshape, match);
}

static void add_routed_gate_up_compile_parameters(Dispatch & dispatch) {
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.routed_gate_up.expert_count",
                                               to_config_value(kQwenRouterExpertCount));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.routed_gate_up.route_count",
                                               to_config_value(kQwenRouterRouteCount));
}

}  // namespace

static bool match_qwen_router_top8_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const RouterTop8Match router_match = match_qwen_router_top8(context.graph, context.root_node);
    if (!router_match.matched()) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kQwenRouterTop8F32Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", router_match.token_count);
    dispatch.kernel.integer_parameters.emplace("route_id_stride", router_match.route_stride);
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.router.expert_count",
                                               to_config_value(kQwenRouterExpertCount));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.router.route_count", to_config_value(kQwenRouterRouteCount));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.workload.token_capacity",
                                               to_config_value(router_match.token_count));

    const size_t route_id_length =
        static_cast<size_t>(router_match.token_count * router_match.route_stride) * sizeof(int32_t);
    dispatch.bindings.push_back({ router_match.logits->id, 0, router_match.logits->byte_count });
    dispatch.bindings.push_back({ router_match.route_ids->id, 0, route_id_length });
    dispatch.bindings.push_back({ router_match.route_weights->id, 0, router_match.route_weights->byte_count });

    if (!append_qwen_router_top8_coverage(context, dispatch_match)) {
        return false;
    }
    dispatch_match.dispatches.push_back(std::move(dispatch));

    const ValueId expert_table_value(context.next_plan_value.value);
    const ValueId partition_table_value(context.next_plan_value.value + 1);
    const size_t  expert_table_bytes    = expert_table_size(router_match.token_count);
    const size_t  partition_table_bytes = partition_table_size(router_match.token_count);
    dispatch_match.transients.push_back(
        { expert_table_value, "qwen.router.expert_table", expert_table_bytes, kQwenRouterPlanTransientAlignment });
    dispatch_match.transients.push_back({ partition_table_value, "qwen.router.partition_table", partition_table_bytes,
                                          kQwenRouterPlanTransientAlignment });
    const CommandPlanResourceMetadata routing_metadata =
        make_command_plan_resource_metadata(QwenMoeRoutingResourceMetadata{
            router_match.token_count,
            kQwenRouterRouteCount,
            router_match.route_stride,
            kQwenRouterExpertCount,
        });
    Status metadata_status;
    if (!dispatch_match.metadata.append_generated_resource(
            {
                router_match.route_ids->id,
                GeneratedResourceRole::QwenMoeExpertTable,
                expert_table_value,
                expert_table_bytes,
                routing_metadata,
            },
            metadata_status) ||
        !dispatch_match.metadata.append_generated_resource(
            {
                router_match.route_ids->id,
                GeneratedResourceRole::QwenMoePartitionTable,
                partition_table_value,
                partition_table_bytes,
                routing_metadata,
            },
            metadata_status) ||
        !dispatch_match.metadata.append_qwen_routing_bundle(
            {
                router_match.route_ids->id,
                router_match.route_weights->id,
                expert_table_value,
                partition_table_value,
                expert_table_bytes,
                partition_table_bytes,
                router_match.token_count,
                kQwenRouterRouteCount,
                router_match.route_stride,
                kQwenRouterExpertCount,
            },
            metadata_status)) {
        return false;
    }

    Dispatch expert_table_dispatch;
    expert_table_dispatch.kernel = make_kernel_specialization(kQwenBuildExpertTableKernel);
    expert_table_dispatch.kernel.integer_parameters.emplace("token_count", router_match.token_count);
    expert_table_dispatch.kernel.integer_parameters.emplace("route_count", kQwenRouterRouteCount);
    expert_table_dispatch.kernel.integer_parameters.emplace("route_stride", router_match.route_stride);
    expert_table_dispatch.kernel.integer_parameters.emplace("expert_count", kQwenRouterExpertCount);
    expert_table_dispatch.kernel.compile_parameters.emplace("qwen3_moe.workload.token_capacity",
                                                            to_config_value(router_match.token_count));
    add_routed_gate_up_compile_parameters(expert_table_dispatch);
    expert_table_dispatch.bindings.push_back({ router_match.route_ids->id, 0, route_id_length });
    expert_table_dispatch.bindings.push_back({ expert_table_value, 0, expert_table_bytes });
    dispatch_match.dispatches.push_back(std::move(expert_table_dispatch));

    Dispatch partition_table_dispatch;
    partition_table_dispatch.kernel = make_kernel_specialization(kQwenBuildExpertPartitionTableKernel);
    partition_table_dispatch.kernel.integer_parameters.emplace("token_count", router_match.token_count);
    partition_table_dispatch.kernel.integer_parameters.emplace("route_count", kQwenRouterRouteCount);
    partition_table_dispatch.kernel.integer_parameters.emplace("expert_count", kQwenRouterExpertCount);
    partition_table_dispatch.kernel.compile_parameters.emplace("qwen3_moe.workload.token_capacity",
                                                               to_config_value(router_match.token_count));
    add_routed_gate_up_compile_parameters(partition_table_dispatch);
    partition_table_dispatch.bindings.push_back({ expert_table_value, 0, expert_table_bytes });
    partition_table_dispatch.bindings.push_back({ partition_table_value, 0, partition_table_bytes });
    dispatch_match.dispatches.push_back(std::move(partition_table_dispatch));
    return true;
}

void register_qwen_router_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "qwen.router.top8_f32",
        GGML_OP_SOFT_MAX,
        DispatchMatchKind::Fused,
        1000,
        DispatchSource::Qwen,
        match_qwen_router_top8_dispatch,
    });
}

}  // namespace ggml::hrx
