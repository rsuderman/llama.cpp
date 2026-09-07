#include "dispatch-gated-mul-mat.h"

#include "dispatch-mul-mat-common.h"
#include "ggml.h"
#include "graph/graph-matcher.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kMulMatF32F32WmmaKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_f32_f32_wmma");
static constexpr KernelCatalogRef kMulMatSwiGLUF32F32WmmaKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_swiglu_f32_f32_wmma");
static constexpr KernelCatalogRef kQuantizeF32SymmetricI4K32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_quantize_f32_symmetric_i4_k32");
static constexpr KernelCatalogRef kMulMatSymmetricI4LowRowAdjacentDualWmmaKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_symmetric_i4_lowrow_adjacent_dual_wmma");
static constexpr KernelCatalogRef kMulMatSymmetricI4LowRowAdjacentDualDirectDotKernels[] = {
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_symmetric_i4_lowrow_adjacent_dual_direct_dot_c1"),
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_symmetric_i4_lowrow_adjacent_dual_direct_dot_c2"),
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_symmetric_i4_lowrow_adjacent_dual_direct_dot_c3"),
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_symmetric_i4_lowrow_adjacent_dual_direct_dot_c4"),
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_symmetric_i4_lowrow_adjacent_dual_direct_dot_c5"),
};
static constexpr KernelCatalogRef kMulMatSwiGLUSymmetricI4WmmaQ8PlaneKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_swiglu_symmetric_i4_wmma_q8_plane");
static constexpr KernelCatalogRef kMulMatQ5KQ8PlaneWmmaToken256Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_q5_k_q8_plane_wmmai8_token256");

struct MulMatSwiGLUMatch {
    const Value *            input       = nullptr;
    const Value *            gate_weight = nullptr;
    const Value *            up_weight   = nullptr;
    const Value *            gate_output = nullptr;
    const Value *            up_output   = nullptr;
    const Value *            output      = nullptr;
    const GraphNode *        gate_node   = nullptr;
    const GraphNode *        up_node     = nullptr;
    const GraphNode *        glu_node    = nullptr;
    CommonMulMatWeightFormat gate_format = CommonMulMatWeightFormat::Q4K;
    CommonMulMatWeightFormat up_format   = CommonMulMatWeightFormat::Q4K;
    int64_t                  input_size  = 0;
    int64_t                  output_size = 0;
    int64_t                  token_count = 0;

    bool topology_matched() const {
        return input != nullptr && gate_weight != nullptr && up_weight != nullptr && gate_output != nullptr &&
               up_output != nullptr && output != nullptr && gate_node != nullptr && up_node != nullptr &&
               glu_node != nullptr;
    }

    bool matched() const { return topology_matched() && token_count > 1; }
};

struct MulMatSwiGLUProjectionMatch {
    MulMatSwiGLUMatch gate_up;
    const GraphNode * projection_node   = nullptr;
    const Value *     projection_weight = nullptr;
    const Value *     projection_output = nullptr;
    int64_t           projection_size   = 0;

    bool matched() const {
        return gate_up.matched() && projection_node != nullptr && projection_weight != nullptr &&
               projection_output != nullptr;
    }
};

static bool supported_symmetric_i4_pair(ggml_type gate_type, ggml_type up_type) {
    return (gate_type == GGML_TYPE_Q4_K && up_type == GGML_TYPE_Q4_K) ||
           (gate_type == GGML_TYPE_Q5_K && up_type == GGML_TYPE_IQ4_XS) ||
           (gate_type == GGML_TYPE_IQ4_XS && up_type == GGML_TYPE_Q5_K);
}

static bool supported_mixed_symmetric_i4_pair(ggml_type gate_type, ggml_type up_type) {
    return (gate_type == GGML_TYPE_Q5_K && up_type == GGML_TYPE_IQ4_XS) ||
           (gate_type == GGML_TYPE_IQ4_XS && up_type == GGML_TYPE_Q5_K);
}

static bool distinct_storage(const Graph & graph, const Value & lhs, const Value & rhs) {
    return !graph.values().same_storage(lhs.id, rhs.id);
}

static size_t symmetric_i4_weight_byte_count(int64_t input_size, int64_t output_size) {
    return static_cast<size_t>(output_size) * static_cast<size_t>(input_size / 256) * size_t{ 144 };
}

static DispatchBinding symmetric_i4_weight_binding(const Value & weight, int64_t input_size, int64_t output_size) {
    DispatchBinding binding;
    binding.value         = weight.id;
    binding.length        = symmetric_i4_weight_byte_count(input_size, output_size);
    binding.layout        = kSymmetricI4K32Row64Layout;
    binding.source_type   = weight.type;
    binding.input_size    = input_size;
    binding.output_size   = output_size;
    binding.source_length = weight.byte_count;
    return binding;
}

static DispatchBinding symmetric_i5_weight_binding(const Value & weight, int64_t input_size, int64_t output_size) {
    DispatchBinding binding;
    binding.value         = weight.id;
    binding.length        = weight.byte_count;
    binding.layout        = kQ5KSymmetricI5K32Layout;
    binding.source_type   = weight.type;
    binding.input_size    = input_size;
    binding.output_size   = output_size;
    binding.source_length = weight.byte_count;
    return binding;
}

static MulMatSwiGLUMatch match_mul_mat_swiglu(const DispatchMatchContext & context) {
    MulMatSwiGLUMatch match;
    CommonMulMatMatch root =
        common_match_mul_mat_any_format(context.graph, context.root_node, kMulMatF32F32WmmaKernel, false);
    if (!root.matched()) {
        root = common_match_mul_mat_any_format(context.graph, context.root_node, kMulMatF32F32WmmaKernel, true);
    }
    if (!root.matched() || !context.graph.has_index()) {
        return match;
    }

    const std::vector<const GraphNode *> & root_consumers = context.graph.index().consumers(context.root_node->output);
    if (root_consumers.size() != 1 || root_consumers.front() == nullptr || root_consumers.front()->op != GGML_OP_GLU) {
        return {};
    }

    const GraphNode * glu_node = root_consumers.front();
    if (glu_node->inputs.size() != 2 || !common_is_swiglu_params(glu_node->params)) {
        return {};
    }

    size_t glu_index = 0;
    if (!context.graph.index().node_index(glu_node, glu_index) || glu_index >= context.covered_nodes.size() ||
        context.covered_nodes[glu_index]) {
        return {};
    }

    const bool root_is_gate = glu_node->inputs[0] == context.root_node->output;
    const bool root_is_up   = glu_node->inputs[1] == context.root_node->output;
    if (!root_is_gate && !root_is_up) {
        return {};
    }

    const ValueId     peer_output_id = root_is_gate ? glu_node->inputs[1] : glu_node->inputs[0];
    const GraphNode * peer_node      = context.graph.index().producer(peer_output_id);
    if (peer_node == nullptr || peer_node == context.root_node || peer_node->op != GGML_OP_MUL_MAT) {
        return {};
    }

    size_t peer_index = 0;
    if (!context.graph.index().node_index(peer_node, peer_index) || peer_index >= context.covered_nodes.size() ||
        context.covered_nodes[peer_index]) {
        return {};
    }

    const std::vector<const GraphNode *> & peer_consumers = context.graph.index().consumers(peer_output_id);
    if (peer_consumers.size() != 1 || peer_consumers.front() != glu_node) {
        return {};
    }

    CommonMulMatMatch peer = common_match_mul_mat_any_format(context.graph, peer_node, kMulMatF32F32WmmaKernel, false);
    if (!peer.matched()) {
        peer = common_match_mul_mat_any_format(context.graph, peer_node, kMulMatF32F32WmmaKernel, true);
    }
    if (!peer.matched() || peer.input->id != root.input->id || peer.input_size != root.input_size ||
        peer.output_size != root.output_size || peer.token_count != root.token_count ||
        !common_same_shape(*root.output, *peer.output)) {
        return {};
    }

    const Value * output = common_graph_value(context.graph, glu_node->output);
    if (output == nullptr || output->type != GGML_TYPE_F32 || !output->contiguous ||
        !common_same_shape(*output, *root.output)) {
        return {};
    }

    match.input       = root.input;
    match.gate_weight = root_is_gate ? root.weight : peer.weight;
    match.up_weight   = root_is_gate ? peer.weight : root.weight;
    match.gate_output = root_is_gate ? root.output : peer.output;
    match.up_output   = root_is_gate ? peer.output : root.output;
    match.output      = output;
    match.gate_node   = root_is_gate ? context.root_node : peer_node;
    match.up_node     = root_is_gate ? peer_node : context.root_node;
    match.glu_node    = glu_node;
    match.gate_format = root_is_gate ? root.weight_format : peer.weight_format;
    match.up_format   = root_is_gate ? peer.weight_format : root.weight_format;
    match.input_size  = root.input_size;
    match.output_size = root.output_size;
    match.token_count = root.token_count;
    return match;
}

static bool match_mul_mat_swiglu_symmetric_i4_lowrow_dispatch(const DispatchMatchContext & context,
                                                              DispatchMatch &              dispatch_match) {
    const MulMatSwiGLUMatch match = match_mul_mat_swiglu(context);
    if (!match.topology_matched() ||
        !supported_mixed_symmetric_i4_pair(match.gate_weight->type, match.up_weight->type) ||
        match.gate_weight->alias_source.value >= 0 || match.up_weight->alias_source.value >= 0 ||
        match.token_count < 1 || match.token_count > 16 || match.input_size % 64 != 0 || match.output_size % 64 != 0 ||
        !distinct_storage(context.graph, *match.gate_weight, *match.up_weight) ||
        !distinct_storage(context.graph, *match.gate_output, *match.up_output)) {
        return false;
    }

    const CommonSymmetricI4ActivationLayout activation_layout =
        common_symmetric_i4_activation_layout(match.input_size, match.token_count);
    const CommandPlanAlternateValue * alternate = find_alternate_value(context.graph, context.plan, match.input->id,
                                                                       GGML_TYPE_COUNT, activation_layout.total_bytes);
    if (alternate != nullptr && alternate->name != kCommonSymmetricI4K32ActivationAlternateName) {
        alternate = nullptr;
    }

    const ValueId activation = alternate != nullptr ? alternate->alternate_value : context.next_plan_value;
    if (alternate == nullptr) {
        dispatch_match.transients.push_back(
            { activation, kCommonSymmetricI4K32ActivationAlternateName, activation_layout.total_bytes, 256 });

        Dispatch quantize;
        quantize.kernel = make_kernel_specialization(kQuantizeF32SymmetricI4K32Kernel);
        quantize.kernel.compile_parameters.emplace("ggml.quantize_symmetric_i4_k32.input_size",
                                                   common_to_config_value(match.input_size));
        quantize.kernel.compile_parameters.emplace("ggml.quantize_symmetric_i4_k32.token_count",
                                                   common_to_config_value(match.token_count));
        quantize.bindings.push_back({ match.input->id, 0, match.input->byte_count });
        quantize.bindings.push_back({ activation, 0, activation_layout.payload_bytes });
        quantize.bindings.push_back({ activation, activation_layout.scales_offset, activation_layout.metadata_bytes });
        quantize.bindings.push_back({ activation, activation_layout.sums_offset, activation_layout.metadata_bytes });
        dispatch_match.dispatches.push_back(std::move(quantize));

        Status metadata_status;
        if (!dispatch_match.metadata.append_alternate_value(
                { match.input->id, activation, GGML_TYPE_COUNT, activation_layout.total_bytes,
                  kCommonSymmetricI4K32ActivationAlternateName },
                metadata_status)) {
            dispatch_match.status.append(metadata_status);
            return false;
        }
    }

    const bool use_direct_dot = match.token_count <= 5 && match.input_size % 256 == 0 &&
                                match.output_size >= 2 * match.input_size;

    Dispatch gate_up;
    gate_up.kernel = make_kernel_specialization(
        use_direct_dot ? kMulMatSymmetricI4LowRowAdjacentDualDirectDotKernels[match.token_count - 1] :
                         kMulMatSymmetricI4LowRowAdjacentDualWmmaKernel);
    gate_up.kernel.compile_parameters.emplace("ggml.mul_mat.symmetric_i4.lowrow.input_size",
                                              common_to_config_value(match.input_size));
    gate_up.kernel.compile_parameters.emplace("ggml.mul_mat.symmetric_i4.lowrow.output_size",
                                              common_to_config_value(match.output_size));
    gate_up.kernel.compile_parameters.emplace("ggml.mul_mat.symmetric_i4.lowrow.token_count",
                                              common_to_config_value(match.token_count));
    gate_up.kernel.compile_parameters.emplace(
        "ggml.mul_mat.symmetric_i4.lowrow.row_group_size",
        common_to_config_value(static_cast<int64_t>(
            common_symmetric_shared4_row_group_size(match.input_size, match.output_size, 4))));
    gate_up.bindings.push_back(
        match.gate_weight->type == GGML_TYPE_Q5_K && match.token_count == 1 ?
            common_symmetric_i4_shared4_multistart_weight_binding(*match.gate_weight, match.input_size,
                                                                  match.output_size) :
            common_symmetric_i4_shared4_weight_binding(*match.gate_weight, match.input_size, match.output_size));
    gate_up.bindings.push_back(
        match.up_weight->type == GGML_TYPE_Q5_K && match.token_count == 1 ?
            common_symmetric_i4_shared4_multistart_weight_binding(*match.up_weight, match.input_size,
                                                                  match.output_size) :
            common_symmetric_i4_shared4_weight_binding(*match.up_weight, match.input_size, match.output_size));
    gate_up.bindings.push_back({ match.gate_output->id, 0, match.gate_output->byte_count });
    gate_up.bindings.push_back({ match.up_output->id, 0, match.up_output->byte_count });
    gate_up.bindings.push_back({ activation, 0, activation_layout.payload_bytes });
    gate_up.bindings.push_back({ activation, activation_layout.scales_offset, activation_layout.metadata_bytes });
    gate_up.bindings.push_back({ activation, activation_layout.sums_offset, activation_layout.metadata_bytes });

    if (!append_covered_node_index_once(context.graph, context.covered_nodes, match.gate_node,
                                        dispatch_match.covered_nodes) ||
        !append_covered_node_index_once(context.graph, context.covered_nodes, match.up_node,
                                        dispatch_match.covered_nodes)) {
        return false;
    }
    dispatch_match.dispatches.push_back(std::move(gate_up));
    return dispatch_match.status.success();
}

static MulMatSwiGLUProjectionMatch match_mul_mat_swiglu_q5_projection(const DispatchMatchContext & context) {
    MulMatSwiGLUProjectionMatch match;
    match.gate_up = match_mul_mat_swiglu(context);
    if (!match.gate_up.matched() ||
        !supported_symmetric_i4_pair(match.gate_up.gate_weight->type, match.gate_up.up_weight->type) ||
        match.gate_up.gate_weight->alias_source.value >= 0 || match.gate_up.up_weight->alias_source.value >= 0 ||
        match.gate_up.token_count < 256 || match.gate_up.token_count % 256 != 0 ||
        match.gate_up.output_size % 256 != 0 ||
        !distinct_storage(context.graph, *match.gate_up.gate_weight, *match.gate_up.up_weight)) {
        return {};
    }

    const GraphNode * projection_node =
        common_find_only_consumer_with_op(context.graph, match.gate_up.output->id, GGML_OP_MUL_MAT);
    if (projection_node == nullptr || projection_node->inputs.size() != 2 ||
        projection_node->inputs[1] != match.gate_up.output->id) {
        return {};
    }

    size_t projection_index = 0;
    if (!context.graph.index().node_index(projection_node, projection_index) ||
        projection_index >= context.covered_nodes.size() || context.covered_nodes[projection_index]) {
        return {};
    }

    const CommonMulMatMatch projection =
        common_match_mul_mat_any_format(context.graph, projection_node, kMulMatQ5KQ8PlaneWmmaToken256Kernel, false);
    if (!projection.matched() || projection.weight->type != GGML_TYPE_Q5_K ||
        projection.weight->alias_source.value >= 0 || projection.input->id != match.gate_up.output->id ||
        projection.input_size != match.gate_up.output_size || projection.token_count != match.gate_up.token_count ||
        projection.output_size % 64 != 0 || !distinct_storage(context.graph, *projection.weight, *projection.output)) {
        return {};
    }

    match.projection_node   = projection_node;
    match.projection_weight = projection.weight;
    match.projection_output = projection.output;
    match.projection_size   = projection.output_size;
    return match;
}

static bool match_mul_mat_swiglu_q5_projection_dispatch(const DispatchMatchContext & context,
                                                        DispatchMatch &              dispatch_match) {
    const MulMatSwiGLUProjectionMatch match = match_mul_mat_swiglu_q5_projection(context);
    if (!match.matched()) {
        return false;
    }

    const size_t input_elements =
        static_cast<size_t>(match.gate_up.token_count) * static_cast<size_t>(match.gate_up.input_size);
    const size_t i4_payload_bytes  = input_elements / 2;
    const size_t i4_metadata_bytes = input_elements / 8;
    const size_t q8_output_bytes =
        static_cast<size_t>(match.gate_up.token_count) * ggml_row_size(GGML_TYPE_Q8_1, match.gate_up.output_size);
    const ValueId i4_payload = context.next_plan_value;
    const ValueId i4_scales(context.next_plan_value.value + 1);
    const ValueId i4_sums(context.next_plan_value.value + 2);
    const ValueId q8_output(context.next_plan_value.value + 3);

    dispatch_match.transients.push_back(
        { i4_payload, "common.mul_mat_swiglu.symmetric_i4.payload", i4_payload_bytes, 256 });
    dispatch_match.transients.push_back(
        { i4_scales, "common.mul_mat_swiglu.symmetric_i4.scales", i4_metadata_bytes, 256 });
    dispatch_match.transients.push_back({ i4_sums, "common.mul_mat_swiglu.symmetric_i4.sums", i4_metadata_bytes, 256 });
    dispatch_match.transients.push_back({ q8_output, "common.mul_mat_swiglu.q8_plane", q8_output_bytes, 256 });

    Dispatch quantize;
    quantize.kernel = make_kernel_specialization(kQuantizeF32SymmetricI4K32Kernel);
    quantize.kernel.compile_parameters.emplace("ggml.quantize_symmetric_i4_k32.input_size",
                                               common_to_config_value(match.gate_up.input_size));
    quantize.kernel.compile_parameters.emplace("ggml.quantize_symmetric_i4_k32.token_count",
                                               common_to_config_value(match.gate_up.token_count));
    quantize.bindings.push_back({ match.gate_up.input->id, 0, match.gate_up.input->byte_count });
    quantize.bindings.push_back({ i4_payload, 0, i4_payload_bytes });
    quantize.bindings.push_back({ i4_scales, 0, i4_metadata_bytes });
    quantize.bindings.push_back({ i4_sums, 0, i4_metadata_bytes });

    Dispatch gate_up;
    gate_up.kernel = make_kernel_specialization(kMulMatSwiGLUSymmetricI4WmmaQ8PlaneKernel);
    gate_up.kernel.compile_parameters.emplace("ggml.mul_mat_swiglu.symmetric_i4.input_size",
                                              common_to_config_value(match.gate_up.input_size));
    gate_up.kernel.compile_parameters.emplace("ggml.mul_mat_swiglu.symmetric_i4.output_size",
                                              common_to_config_value(match.gate_up.output_size));
    gate_up.kernel.compile_parameters.emplace("ggml.mul_mat_swiglu.symmetric_i4.token_count",
                                              common_to_config_value(match.gate_up.token_count));
    gate_up.bindings.push_back(
        symmetric_i4_weight_binding(*match.gate_up.gate_weight, match.gate_up.input_size, match.gate_up.output_size));
    gate_up.bindings.push_back(
        symmetric_i4_weight_binding(*match.gate_up.up_weight, match.gate_up.input_size, match.gate_up.output_size));
    gate_up.bindings.push_back({ match.gate_up.input->id, 0, match.gate_up.input->byte_count });
    gate_up.bindings.push_back({ match.gate_up.output->id, 0, match.gate_up.output->byte_count });
    gate_up.bindings.push_back({ i4_payload, 0, i4_payload_bytes });
    gate_up.bindings.push_back({ i4_scales, 0, i4_metadata_bytes });
    gate_up.bindings.push_back({ i4_sums, 0, i4_metadata_bytes });
    gate_up.bindings.push_back({ q8_output, 0, q8_output_bytes });

    Dispatch projection;
    projection.kernel = make_kernel_specialization(kMulMatQ5KQ8PlaneWmmaToken256Kernel);
    projection.kernel.integer_parameters.emplace("token_count", match.gate_up.token_count);
    projection.kernel.compile_parameters.emplace("ggml.mul_mat_q5_k_q8_plane.input_size",
                                                 common_to_config_value(match.gate_up.output_size));
    projection.kernel.compile_parameters.emplace("ggml.mul_mat_q5_k_q8_plane.output_size",
                                                 common_to_config_value(match.projection_size));
    projection.kernel.compile_parameters.emplace("ggml.mul_mat_q5_k_q8_plane.token_capacity",
                                                 common_to_config_value(match.gate_up.token_count));
    projection.bindings.push_back({ q8_output, 0, q8_output_bytes });
    projection.bindings.push_back(
        symmetric_i5_weight_binding(*match.projection_weight, match.gate_up.output_size, match.projection_size));
    projection.bindings.push_back({ match.projection_output->id, 0, match.projection_output->byte_count });

    if (!append_covered_node_index_once(context.graph, context.covered_nodes, match.gate_up.gate_node,
                                        dispatch_match.covered_nodes) ||
        !append_covered_node_index_once(context.graph, context.covered_nodes, match.gate_up.up_node,
                                        dispatch_match.covered_nodes) ||
        !append_covered_node_index_once(context.graph, context.covered_nodes, match.gate_up.glu_node,
                                        dispatch_match.covered_nodes) ||
        !append_covered_node_index_once(context.graph, context.covered_nodes, match.projection_node,
                                        dispatch_match.covered_nodes)) {
        return false;
    }

    dispatch_match.dispatches.push_back(std::move(quantize));
    dispatch_match.dispatches.push_back(std::move(gate_up));
    dispatch_match.dispatches.push_back(std::move(projection));
    return true;
}

static bool match_mul_mat_swiglu_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const MulMatSwiGLUMatch match = match_mul_mat_swiglu(context);
    if (!match.matched()) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kMulMatSwiGLUF32F32WmmaKernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.compile_parameters.emplace("ggml.workload.token_capacity",
                                               common_to_config_value(match.token_count));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_swiglu.input_size",
                                               common_to_config_value(match.input_size));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_swiglu.output_size",
                                               common_to_config_value(match.output_size));
    dispatch.kernel.compile_parameters.emplace(
        "ggml.mul_mat_swiglu.gate_weight_format",
        common_to_config_value(common_mul_mat_format_config_value(match.gate_format)));
    dispatch.kernel.compile_parameters.emplace(
        "ggml.mul_mat_swiglu.up_weight_format",
        common_to_config_value(common_mul_mat_format_config_value(match.up_format)));
    dispatch.bindings.push_back({ match.input->id, 0, match.input->byte_count });
    dispatch.bindings.push_back({ match.gate_weight->id, 0, match.gate_weight->byte_count });
    dispatch.bindings.push_back({ match.up_weight->id, 0, match.up_weight->byte_count });
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });

    if (!append_covered_node_index_once(context.graph, context.covered_nodes, match.gate_node,
                                        dispatch_match.covered_nodes) ||
        !append_covered_node_index_once(context.graph, context.covered_nodes, match.up_node,
                                        dispatch_match.covered_nodes) ||
        !append_covered_node_index_once(context.graph, context.covered_nodes, match.glu_node,
                                        dispatch_match.covered_nodes)) {
        return false;
    }
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_gated_mul_mat_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "common.mul_mat_swiglu.symmetric_i4_lowrow_adjacent_dual",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        310,
        DispatchSource::Common,
        match_mul_mat_swiglu_symmetric_i4_lowrow_dispatch,
    });
    registry.add({
        "common.mul_mat_swiglu_q5_projection.symmetric_i4_q8_plane",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        300,
        DispatchSource::Common,
        match_mul_mat_swiglu_q5_projection_dispatch,
    });
    registry.add({
        "common.mul_mat_swiglu.f32_f32_wmma",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        290,
        DispatchSource::Common,
        match_mul_mat_swiglu_dispatch,
    });
}

}  // namespace ggml::hrx
