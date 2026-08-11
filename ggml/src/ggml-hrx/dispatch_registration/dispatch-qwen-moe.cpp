#include "dispatch-qwen-moe.h"

#include "ggml.h"
#include "graph/graph-matcher.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kQwenRoutedGateUpSwiGLUQ4KF16WmmaKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_routed_gate_up_swiglu_q4k_f16_wmma");
static constexpr KernelCatalogRef kQwenRoutedDownQ4KF16WmmaGroupedKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_routed_down_q4k_f16_wmma_grouped");
static constexpr KernelCatalogRef kQwenRoutedDownQ6KF16WmmaGroupedKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_routed_down_q6k_f16_wmma_grouped");
static constexpr KernelCatalogRef kQwenRoutedDownWeightedReduceF16F32Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_routed_down_weighted_reduce_f16_f32");
static constexpr KernelCatalogRef kQwenRoutedDownWeightedReduceNextRmsNormF32Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_routed_down_weighted_reduce_next_rmsnorm_f32");

static constexpr int64_t      kQwenMoeInputSize               = 2048;
static constexpr int64_t      kQwenMoeOutputSize              = 768;
static constexpr int64_t      kQwenMoeExpertCount             = 128;
static constexpr int64_t      kQwenMoeRouteCount              = 8;
static constexpr size_t       kQwenMoePlanTransientAlignment  = 256;
static constexpr const char * kQwenMoeF16GateUpOutputName     = "qwen.moe.gate_up_swiglu_f16";
static constexpr const char * kQwenMoeF16RoutedDownOutputName = "qwen.moe.routed_down_f16";

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

static bool is_qwen_rms_norm_epsilon(float eps) {
    return eps >= 0.0000009f && eps <= 0.0000011f;
}

static bool is_supported_token_count(int64_t token_count) {
    return token_count >= 1 && token_count <= 2048;
}

static bool is_qwen_routed_weight(const Value & value) {
    return value.type == GGML_TYPE_Q4_K && value.contiguous &&
           is_shape(value, kQwenMoeInputSize, kQwenMoeOutputSize, kQwenMoeExpertCount, 1);
}

static bool is_qwen_routed_down_weight(const Value & value) {
    return (value.type == GGML_TYPE_Q4_K || value.type == GGML_TYPE_Q6_K) && value.contiguous &&
           is_shape(value, kQwenMoeOutputSize, kQwenMoeInputSize, kQwenMoeExpertCount, 1);
}

static bool is_qwen_routed_projection_output(const Value & value, int64_t token_count) {
    return value.type == GGML_TYPE_F32 && value.contiguous &&
           is_shape(value, kQwenMoeOutputSize, kQwenMoeRouteCount, token_count, 1);
}

static bool is_qwen_routed_down_output(const Value & value, int64_t token_count) {
    return value.type == GGML_TYPE_F32 && value.contiguous &&
           is_shape(value, kQwenMoeInputSize, kQwenMoeRouteCount, token_count, 1);
}

static const GraphNode * find_consumer_with_op(const Graph & graph, ValueId value, ggml_op op) {
    for (const GraphNode * consumer : graph.index().consumers(value)) {
        if (consumer != nullptr && consumer->op == op) {
            return consumer;
        }
    }
    return nullptr;
}

static std::vector<const GraphNode *> find_consumers_with_op(const Graph & graph, ValueId value, ggml_op op) {
    std::vector<const GraphNode *> matches;
    for (const GraphNode * consumer : graph.index().consumers(value)) {
        if (consumer != nullptr && consumer->op == op) {
            matches.push_back(consumer);
        }
    }
    return matches;
}

static const GraphNode * find_single_consumer_with_op(const Graph & graph, ValueId value, ggml_op op) {
    const std::vector<const GraphNode *> consumers = find_consumers_with_op(graph, value, op);
    return consumers.size() == 1 ? consumers.front() : nullptr;
}

static const GraphNode * producer_with_op(const Graph & graph, ValueId value, ggml_op op) {
    const GraphNode * producer = graph.index().producer(value);
    return producer != nullptr && producer->op == op ? producer : nullptr;
}

static bool append_covered_node(const DispatchMatchContext & context, const GraphNode * node, DispatchMatch & match) {
    return append_covered_node_index_once(context.graph, context.covered_nodes, node, match.covered_nodes);
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

static size_t f16_routed_down_output_size(int64_t token_count) {
    return static_cast<size_t>(token_count * kQwenMoeRouteCount * kQwenMoeInputSize) * sizeof(ggml_fp16_t);
}

static void add_routed_down_compile_parameters(Dispatch & dispatch, int64_t token_count) {
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.routed_down.input_size", to_config_value(kQwenMoeOutputSize));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.routed_down.route_count",
                                               to_config_value(kQwenMoeRouteCount));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.routed_down.expert_count",
                                               to_config_value(kQwenMoeExpertCount));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.routed_down.output_size", to_config_value(kQwenMoeInputSize));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.workload.token_capacity", to_config_value(token_count));
}

struct RoutedGateUpMatch {
    const Value *                        gate_weight    = nullptr;
    const Value *                        up_weight      = nullptr;
    const Value *                        input          = nullptr;
    const Value *                        route_ids      = nullptr;
    const Value *                        gate_output    = nullptr;
    const Value *                        up_output      = nullptr;
    const Value *                        glu_output     = nullptr;
    const CommandPlanQwenRoutingBundle * routing_bundle = nullptr;
    const GraphNode *                    gate_node      = nullptr;
    const GraphNode *                    up_node        = nullptr;
    const GraphNode *                    glu_node       = nullptr;
    int64_t                              token_count    = 0;

    bool matched() const {
        return gate_weight != nullptr && up_weight != nullptr && input != nullptr && route_ids != nullptr &&
               gate_output != nullptr && up_output != nullptr && glu_output != nullptr && routing_bundle != nullptr &&
               gate_node != nullptr && up_node != nullptr && glu_node != nullptr && token_count > 0;
    }
};

struct RoutedDownMatch {
    const Value *                        input_graph_value = nullptr;
    const CommandPlanAlternateValue *    input_alternate   = nullptr;
    const Value *                        weight            = nullptr;
    const Value *                        output            = nullptr;
    const Value *                        route_ids         = nullptr;
    const CommandPlanQwenRoutingBundle * routing_bundle    = nullptr;
    KernelCatalogRef                     kernel            = {};
    int64_t                              token_count       = 0;

    bool matched() const {
        return input_graph_value != nullptr && input_alternate != nullptr && weight != nullptr && output != nullptr &&
               route_ids != nullptr && routing_bundle != nullptr && kernel.id != kUncatalogedKernelId &&
               token_count > 0;
    }
};

struct WeightedReduceNextRmsNormMatch {
    const GraphNode * rms_node    = nullptr;
    const GraphNode * mul_node    = nullptr;
    const Value *     norm_weight = nullptr;
    const Value *     output      = nullptr;

    bool matched() const {
        return rms_node != nullptr && mul_node != nullptr && norm_weight != nullptr && output != nullptr;
    }
};

struct WeightedReduceMatch {
    const Value *                     route_weights    = nullptr;
    const Value *                     routed_output    = nullptr;
    const CommandPlanAlternateValue * routed_alternate = nullptr;
    const Value *                     output           = nullptr;
    const GraphNode *                 weighted_node    = nullptr;
    std::vector<const GraphNode *>    views;
    std::vector<const GraphNode *>    reductions;
    const GraphNode *                 residual = nullptr;
    WeightedReduceNextRmsNormMatch    next_rmsnorm;
    int64_t                           token_count = 0;

    bool matched() const {
        return route_weights != nullptr && routed_output != nullptr && routed_alternate != nullptr &&
               output != nullptr && weighted_node != nullptr && !views.empty() && residual != nullptr &&
               token_count > 0;
    }
};

static bool bundle_matches_qwen_router(const CommandPlanQwenRoutingBundle & bundle,
                                       ValueId                              route_ids,
                                       int64_t                              token_count) {
    return bundle.route_ids == route_ids && bundle.route_weights.value >= 0 && bundle.expert_table.value >= 0 &&
           bundle.partition_table.value >= 0 && bundle.expert_table_byte_count == expert_table_size(token_count) &&
           bundle.partition_table_byte_count == partition_table_size(token_count) &&
           bundle.token_count == token_count && bundle.route_count == kQwenMoeRouteCount &&
           bundle.expert_count == kQwenMoeExpertCount && bundle.route_stride >= kQwenMoeRouteCount;
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

static RoutedDownMatch match_qwen_routed_down_grouped(const DispatchMatchContext & context) {
    RoutedDownMatch   match;
    const GraphNode * root = context.root_node;
    if (root == nullptr || root->op != GGML_OP_MUL_MAT_ID || root->inputs.size() != 3 || !context.graph.has_index()) {
        return match;
    }

    const Value * weight      = graph_value(context.graph, root->inputs[0]);
    const Value * input       = graph_value(context.graph, root->inputs[1]);
    const Value * route_ids   = graph_value(context.graph, root->inputs[2]);
    const Value * root_output = graph_value(context.graph, root->output);
    if (weight == nullptr || input == nullptr || route_ids == nullptr || root_output == nullptr ||
        !is_qwen_routed_down_weight(*weight) || !is_qwen_routed_projection_output(*input, input->ne[2]) ||
        route_ids->type != GGML_TYPE_I32 || !is_shape(*route_ids, kQwenMoeRouteCount, input->ne[2], 1, 1)) {
        return {};
    }

    const int64_t token_count = input->ne[2];
    if (!is_supported_token_count(token_count) || !is_qwen_routed_down_output(*root_output, token_count)) {
        return {};
    }

    const CommandPlanQwenRoutingBundle * routing_bundle = context.plan.metadata.find_qwen_routing_bundle(route_ids->id);
    if (routing_bundle == nullptr || !bundle_matches_qwen_router(*routing_bundle, route_ids->id, token_count)) {
        return {};
    }

    const CommandPlanAlternateValue * input_alternate =
        find_alternate_value(context.plan, input->id, GGML_TYPE_F16, f16_gate_up_output_size(token_count));
    if (input_alternate == nullptr) {
        return {};
    }

    match.input_graph_value = input;
    match.input_alternate   = input_alternate;
    match.weight            = weight;
    match.output            = root_output;
    match.route_ids         = route_ids;
    match.routing_bundle    = routing_bundle;
    match.kernel            = weight->type == GGML_TYPE_Q4_K ? kQwenRoutedDownQ4KF16WmmaGroupedKernel :
                                                               kQwenRoutedDownQ6KF16WmmaGroupedKernel;
    match.token_count       = token_count;
    return match;
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

    const CommandPlanQwenRoutingBundle * routing_bundle = context.plan.metadata.find_qwen_routing_bundle(route_ids->id);
    if (routing_bundle == nullptr || !bundle_matches_qwen_router(*routing_bundle, route_ids->id, token_count)) {
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

    match.gate_weight    = gate_weight;
    match.up_weight      = up_weight;
    match.input          = input;
    match.route_ids      = route_ids;
    match.gate_output    = gate_output;
    match.up_output      = up_output;
    match.glu_output     = glu_output;
    match.routing_bundle = routing_bundle;
    match.gate_node      = gate_node;
    match.up_node        = up_node;
    match.glu_node       = glu_node;
    match.token_count    = token_count;
    return match;
}

static WeightedReduceNextRmsNormMatch match_qwen_weighted_reduce_next_rmsnorm(const DispatchMatchContext & context,
                                                                              const Value &                residual) {
    WeightedReduceNextRmsNormMatch match;
    const GraphNode * rms_node = find_single_consumer_with_op(context.graph, residual.id, GGML_OP_RMS_NORM);
    if (rms_node == nullptr || rms_node->inputs.size() != 1) {
        return match;
    }
    const RmsNormParams * rms_params = op_params_as<RmsNormParams>(rms_node->params);
    if (rms_params == nullptr || !is_qwen_rms_norm_epsilon(rms_params->eps)) {
        return {};
    }

    const Value * rms = graph_value(context.graph, rms_node->output);
    if (rms == nullptr || rms->type != GGML_TYPE_F32 || !same_shape(*rms, residual)) {
        return {};
    }

    const GraphNode * mul_node = find_single_consumer_with_op(context.graph, rms_node->output, GGML_OP_MUL);
    if (mul_node == nullptr || mul_node->inputs.size() != 2) {
        return {};
    }

    const Value * norm_weight = nullptr;
    for (ValueId input : mul_node->inputs) {
        if (input != rms_node->output) {
            norm_weight = graph_value(context.graph, input);
        }
    }
    const Value * output = graph_value(context.graph, mul_node->output);
    if (norm_weight == nullptr || output == nullptr || norm_weight->type != GGML_TYPE_F32 ||
        output->type != GGML_TYPE_F32 || !norm_weight->contiguous || !output->contiguous ||
        !is_shape(*norm_weight, kQwenMoeInputSize, 1, 1, 1) || !same_shape(*output, residual)) {
        return {};
    }

    match.rms_node    = rms_node;
    match.mul_node    = mul_node;
    match.norm_weight = norm_weight;
    match.output      = output;
    return match;
}

static bool append_node_if_uncovered(const DispatchMatchContext &     context,
                                     const GraphNode *                node,
                                     std::vector<const GraphNode *> & nodes) {
    size_t index = 0;
    if (node == nullptr || !context.graph.index().node_index(node, index) || index >= context.covered_nodes.size() ||
        context.covered_nodes[index]) {
        return false;
    }
    for (const GraphNode * existing : nodes) {
        if (existing == node) {
            return true;
        }
    }
    nodes.push_back(node);
    return true;
}

static WeightedReduceMatch match_qwen_routed_down_weighted_reduce(const DispatchMatchContext & context) {
    WeightedReduceMatch match;
    const GraphNode *   weighted = context.root_node;
    if (weighted == nullptr || weighted->op != GGML_OP_MUL || weighted->inputs.size() != 2 ||
        !context.graph.has_index()) {
        return match;
    }

    const Value * routed_output = nullptr;
    const Value * route_weights = nullptr;
    for (ValueId input : weighted->inputs) {
        const Value * value = graph_value(context.graph, input);
        if (value == nullptr) {
            return {};
        }
        if (is_qwen_routed_down_output(*value, value->ne[2])) {
            routed_output = value;
        } else if (value->type == GGML_TYPE_F32 && value->contiguous &&
                   is_shape(*value, 1, kQwenMoeRouteCount, value->ne[2], 1)) {
            route_weights = value;
        }
    }
    const Value * weighted_output = graph_value(context.graph, weighted->output);
    if (routed_output == nullptr || route_weights == nullptr || weighted_output == nullptr ||
        routed_output->ne[2] != route_weights->ne[2] || !same_shape(*weighted_output, *routed_output)) {
        return {};
    }

    const int64_t token_count = routed_output->ne[2];
    if (!is_supported_token_count(token_count)) {
        return {};
    }
    const CommandPlanAlternateValue * routed_alternate =
        find_alternate_value(context.plan, routed_output->id, GGML_TYPE_F16, f16_routed_down_output_size(token_count));
    if (routed_alternate == nullptr) {
        return {};
    }

    bool known_route_weights = false;
    for (const CommandPlanQwenRoutingBundle & bundle : context.plan.metadata.qwen_routing_bundles()) {
        if (bundle.route_weights == route_weights->id &&
            bundle_matches_qwen_router(bundle, bundle.route_ids, token_count)) {
            known_route_weights = true;
            break;
        }
    }
    if (!known_route_weights) {
        return {};
    }

    std::vector<const GraphNode *> views =
        layout_alias_consumers_with_op(context.graph, weighted->output, GGML_OP_VIEW);
    if (views.size() != kQwenMoeRouteCount) {
        return {};
    }

    std::set<int32_t>              routed_values;
    std::vector<const GraphNode *> owned_views;
    for (const GraphNode * view : views) {
        const Value * value = view == nullptr ? nullptr : graph_value(context.graph, view->output);
        if (value == nullptr || value->type != GGML_TYPE_F32 ||
            !is_shape(*value, kQwenMoeInputSize, token_count, 1, 1) ||
            !append_node_if_uncovered(context, view, owned_views)) {
            return {};
        }
        routed_values.insert(view->output.value);
    }

    std::vector<const GraphNode *> reductions;
    bool                           changed = true;
    while (changed) {
        changed                        = false;
        const std::set<int32_t> values = routed_values;
        for (int32_t value : values) {
            for (const GraphNode * add : find_consumers_with_op(context.graph, ValueId(value), GGML_OP_ADD)) {
                if (add == nullptr || add->inputs.size() != 2) {
                    continue;
                }
                bool already_owned = false;
                for (const GraphNode * reduction : reductions) {
                    if (reduction == add) {
                        already_owned = true;
                        break;
                    }
                }
                if (already_owned) {
                    continue;
                }
                bool all_routed = true;
                for (ValueId input : add->inputs) {
                    all_routed = all_routed && routed_values.count(input.value) != 0;
                }
                if (!all_routed) {
                    continue;
                }
                const Value * output = graph_value(context.graph, add->output);
                if (output == nullptr || output->type != GGML_TYPE_F32 ||
                    !is_shape(*output, kQwenMoeInputSize, token_count, 1, 1) ||
                    !append_node_if_uncovered(context, add, reductions)) {
                    return {};
                }
                routed_values.insert(add->output.value);
                changed = true;
            }
        }
    }

    const GraphNode * residual = nullptr;
    for (int32_t value : routed_values) {
        for (const GraphNode * add : find_consumers_with_op(context.graph, ValueId(value), GGML_OP_ADD)) {
            if (add == nullptr || add->inputs.size() != 2) {
                continue;
            }
            bool is_reduction = false;
            for (const GraphNode * reduction : reductions) {
                if (reduction == add) {
                    is_reduction = true;
                    break;
                }
            }
            if (is_reduction) {
                continue;
            }
            int routed_input_count = 0;
            for (ValueId input : add->inputs) {
                if (routed_values.count(input.value) != 0) {
                    ++routed_input_count;
                }
            }
            if (routed_input_count != 1 || residual != nullptr) {
                return {};
            }
            residual = add;
        }
    }
    if (residual == nullptr || reductions.size() + 1 != views.size()) {
        return {};
    }
    const Value * output = graph_value(context.graph, residual->output);
    if (output == nullptr || output->type != GGML_TYPE_F32 ||
        !is_shape(*output, kQwenMoeInputSize, token_count, 1, 1)) {
        return {};
    }

    match.route_weights    = route_weights;
    match.routed_output    = routed_output;
    match.routed_alternate = routed_alternate;
    match.output           = output;
    match.weighted_node    = weighted;
    match.views            = std::move(owned_views);
    match.reductions       = std::move(reductions);
    match.residual         = residual;
    match.next_rmsnorm     = match_qwen_weighted_reduce_next_rmsnorm(context, *output);
    match.token_count      = token_count;
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
        { match.routing_bundle->expert_table, 0, match.routing_bundle->expert_table_byte_count });
    dispatch.bindings.push_back(
        { match.routing_bundle->partition_table, 0, match.routing_bundle->partition_table_byte_count });
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

static bool build_qwen_routed_down_grouped_dispatch(const DispatchMatchContext & context,
                                                    DispatchMatch &              dispatch_match,
                                                    KernelCatalogRef             expected_kernel) {
    const RoutedDownMatch match = match_qwen_routed_down_grouped(context);
    if (!match.matched() || match.kernel.id != expected_kernel.id) {
        return false;
    }

    const ValueId f16_output(context.next_plan_value.value);
    const size_t  f16_output_bytes = f16_routed_down_output_size(match.token_count);
    dispatch_match.transients.push_back(
        { f16_output, kQwenMoeF16RoutedDownOutputName, f16_output_bytes, kQwenMoePlanTransientAlignment });
    Status metadata_status;
    if (!dispatch_match.metadata.append_alternate_value(
            { match.output->id, f16_output, GGML_TYPE_F16, f16_output_bytes, kQwenMoeF16RoutedDownOutputName },
            metadata_status)) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(match.kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    add_routed_down_compile_parameters(dispatch, match.token_count);
    dispatch.bindings.push_back({ match.input_alternate->alternate_value, 0, match.input_alternate->byte_count });
    dispatch.bindings.push_back(
        { match.routing_bundle->expert_table, 0, match.routing_bundle->expert_table_byte_count });
    dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    dispatch.bindings.push_back({ f16_output, 0, f16_output_bytes });

    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_qwen_routed_down_weighted_reduce_dispatch(const DispatchMatchContext & context,
                                                            DispatchMatch &              dispatch_match) {
    const WeightedReduceMatch match = match_qwen_routed_down_weighted_reduce(context);
    if (!match.matched()) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel =
        make_kernel_specialization(match.next_rmsnorm.matched() ? kQwenRoutedDownWeightedReduceNextRmsNormF32Kernel :
                                                                  kQwenRoutedDownWeightedReduceF16F32Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    add_routed_down_compile_parameters(dispatch, match.token_count);
    if (match.next_rmsnorm.matched()) {
        dispatch.kernel.compile_parameters.emplace("qwen3_moe.model.hidden_size", to_config_value(kQwenMoeInputSize));
        dispatch.kernel.compile_parameters.emplace("qwen3_moe.model.rms_epsilon", "0.000001");
        dispatch.bindings.push_back({ match.route_weights->id, 0, match.route_weights->byte_count });
        dispatch.bindings.push_back({ match.routed_alternate->alternate_value, 0, match.routed_alternate->byte_count });
        dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });
        dispatch.bindings.push_back(
            { match.next_rmsnorm.norm_weight->id, 0, match.next_rmsnorm.norm_weight->byte_count });
        dispatch.bindings.push_back({ match.next_rmsnorm.output->id, 0, match.next_rmsnorm.output->byte_count });
    } else {
        dispatch.bindings.push_back({ match.route_weights->id, 0, match.route_weights->byte_count });
        dispatch.bindings.push_back({ match.routed_alternate->alternate_value, 0, match.routed_alternate->byte_count });
        dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });
    }

    if (!append_covered_node(context, match.weighted_node, dispatch_match)) {
        return false;
    }
    for (const GraphNode * view : match.views) {
        if (!append_covered_node(context, view, dispatch_match)) {
            return false;
        }
    }
    for (const GraphNode * reduction : match.reductions) {
        if (!append_covered_node(context, reduction, dispatch_match)) {
            return false;
        }
    }
    if (!append_covered_node(context, match.residual, dispatch_match)) {
        return false;
    }
    if (match.next_rmsnorm.matched() && (!append_covered_node(context, match.next_rmsnorm.rms_node, dispatch_match) ||
                                         !append_covered_node(context, match.next_rmsnorm.mul_node, dispatch_match))) {
        return false;
    }

    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_qwen_routed_down_q4k_f16_wmma_grouped_dispatch(const DispatchMatchContext & context,
                                                                 DispatchMatch &              dispatch_match) {
    return build_qwen_routed_down_grouped_dispatch(context, dispatch_match, kQwenRoutedDownQ4KF16WmmaGroupedKernel);
}

static bool match_qwen_routed_down_q6k_f16_wmma_grouped_dispatch(const DispatchMatchContext & context,
                                                                 DispatchMatch &              dispatch_match) {
    return build_qwen_routed_down_grouped_dispatch(context, dispatch_match, kQwenRoutedDownQ6KF16WmmaGroupedKernel);
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
    registry.add({
        "qwen.moe.routed_down_q4k_f16_wmma_grouped",
        GGML_OP_MUL_MAT_ID,
        DispatchMatchKind::Fused,
        900,
        DispatchSource::Qwen,
        match_qwen_routed_down_q4k_f16_wmma_grouped_dispatch,
    });
    registry.add({
        "qwen.moe.routed_down_q6k_f16_wmma_grouped",
        GGML_OP_MUL_MAT_ID,
        DispatchMatchKind::Fused,
        900,
        DispatchSource::Qwen,
        match_qwen_routed_down_q6k_f16_wmma_grouped_dispatch,
    });
    registry.add({
        "qwen.moe.routed_down_weighted_reduce",
        GGML_OP_MUL,
        DispatchMatchKind::Fused,
        800,
        DispatchSource::Qwen,
        match_qwen_routed_down_weighted_reduce_dispatch,
    });
}

}  // namespace ggml::hrx
