#include "dispatch-rmsnorm.h"

#include "dispatch-binary-common.h"
#include "dispatch-activation-publication.h"
#include "dispatch-layout-utils.h"
#include "dispatch-mul-mat-common.h"
#include "ggml.h"
#include "graph/graph-matcher.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kRmsNormBinaryF32Kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_rmsnorm_binary_f32");
static constexpr KernelCatalogRef kRmsNormBinaryStridedF32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_rmsnorm_binary_strided_f32");
static constexpr KernelCatalogRef kRmsNormMulAddF32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_rmsnorm_mul_add_f32");
static constexpr KernelCatalogRef kRmsNormBinaryF32F16Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_rmsnorm_binary_f32_f16");
static constexpr KernelCatalogRef kRmsNormBinaryF32K16Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_rmsnorm_binary_f32_k16");
static constexpr KernelCatalogRef kRmsNormBinaryQ8_1X4F16Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_rmsnorm_binary_q8_1_x4_f16");
static constexpr KernelCatalogRef kRmsNormF32Kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_rmsnorm_f32");
static constexpr KernelCatalogRef kRmsNormMulRopeF32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_rmsnorm_mul_rope_f32");
static constexpr KernelCatalogRef kAddRmsNormBinarySymmetricI4K32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_add_rmsnorm_binary_symmetric_i4_k32");
static constexpr KernelCatalogRef kRmsNormBinarySymmetricI4K32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_rmsnorm_binary_symmetric_i4_k32");
static constexpr KernelCatalogRef kRmsNormGateSiluMulSymmetricI4K32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_rmsnorm_gate_silu_mul_symmetric_i4_k32");
static constexpr KernelCatalogRef kRmsNormGateF32F16Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_rmsnorm_gate_f32_f16");
static constexpr KernelCatalogRef kRmsNormGateF32Q8_1X4Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_rmsnorm_gate_f32_q8_1_x4");

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

static bool is_supported_f32_hidden_size(int64_t hidden_size) {
    return hidden_size >= 64 && hidden_size <= 32768 && hidden_size % 64 == 0;
}

static bool is_supported_binary_f32_hidden_size(int64_t hidden_size) {
    return hidden_size >= 128 && hidden_size <= 32768 && hidden_size % 128 == 0;
}

static bool is_supported_symmetric_i4_hidden_size(int64_t hidden_size) {
    return hidden_size >= 128 && hidden_size <= 32768 && hidden_size % 128 == 0;
}

static bool is_supported_token_count(int64_t token_count) {
    return token_count >= 1 && token_count <= (1 << 20);
}

static bool supported_rmsnorm_input_layout(const Value & input,
                                           int64_t       hidden_size,
                                           int64_t       token_count,
                                           int64_t &     input_stride,
                                           size_t &      input_span_bytes) {
    if (input.type != GGML_TYPE_F32 || input.ne[0] != hidden_size || input.nb[0] != sizeof(float)) {
        return false;
    }

    if (input.contiguous) {
        input_stride     = hidden_size;
        input_span_bytes = input.byte_count;
        return true;
    }

    if (input.ne[1] != token_count || input.ne[2] != 1 || input.ne[3] != 1 || input.nb[1] % sizeof(float) != 0) {
        return false;
    }
    input_stride = static_cast<int64_t>(input.nb[1] / sizeof(float));
    if (input_stride < hidden_size || input_stride > 1048576) {
        return false;
    }

    return strided_f32_storage_span_bytes(input, input_span_bytes);
}

static bool is_packed_q8_consumer(const Graph & graph, const GraphNode * consumer, const Value & input) {
    if (common_accepts_q8_1_x4_prefill_mul_mat(graph, consumer, input)) {
        return true;
    }
    if (!common_accepts_q8_1_x4_decode_mul_mat(graph, consumer, input)) {
        return false;
    }
    const Value * weight = graph_value(graph, consumer->inputs[0]);
    const Value * output = graph_value(graph, consumer->output);
    return weight->type == GGML_TYPE_Q4_K || !graph.index().consumers(output->id).empty();
}

static bool rmsnorm_q8_consumer_accepts(const DispatchMatchContext & context,
                                        const GraphNode &            consumer,
                                        const Value &                input) {
    return is_packed_q8_consumer(context.graph, &consumer, input);
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

struct RmsNormBinaryMatch {
    const GraphNode * rms_node          = nullptr;
    const GraphNode * binary_node       = nullptr;
    const Value *     input             = nullptr;
    const Value *     rhs               = nullptr;
    const Value *     output            = nullptr;
    size_t            rms_node_index    = 0;
    size_t            binary_node_index = 0;
    int64_t           hidden_size       = 0;
    int64_t           token_count       = 0;
    int64_t           input_stride      = 0;
    size_t            input_span        = 0;
    BinaryKind        op                = BinaryKind::Add;
    float             epsilon           = 0.0f;

    bool matched() const {
        return rms_node != nullptr && binary_node != nullptr && input != nullptr && rhs != nullptr && output != nullptr;
    }
};

struct RmsNormMulAddMatch {
    RmsNormBinaryMatch rms_binary;
    const GraphNode *  add_node       = nullptr;
    const Value *      residual       = nullptr;
    const Value *      output         = nullptr;
    size_t             add_node_index = 0;
    BinaryKind         output_op      = BinaryKind::Add;
    bool               binary_lhs     = true;

    bool matched() const {
        return rms_binary.matched() && add_node != nullptr && residual != nullptr && output != nullptr;
    }
};

struct RmsNormMatch {
    const GraphNode * rms_node       = nullptr;
    const Value *     input          = nullptr;
    const Value *     output         = nullptr;
    size_t            input_span     = 0;
    size_t            rms_node_index = 0;
    int64_t           input_stride   = 0;
    int64_t           hidden_size    = 0;
    int64_t           token_count    = 0;
    float             epsilon        = 0.0f;

    bool matched() const { return rms_node != nullptr && input != nullptr && output != nullptr; }
};

struct RmsNormMulRopeMatch {
    const GraphNode *     rms         = nullptr;
    const GraphNode *     binary      = nullptr;
    const GraphNode *     rope        = nullptr;
    const Value *         input       = nullptr;
    const Value *         weight      = nullptr;
    const Value *         positions   = nullptr;
    const Value *         output      = nullptr;
    const RmsNormParams * rms_params  = nullptr;
    const RopeParams *    rope_params = nullptr;

    bool matched() const {
        return rms != nullptr && binary != nullptr && rope != nullptr && input != nullptr && weight != nullptr &&
               positions != nullptr && output != nullptr && rms_params != nullptr && rope_params != nullptr;
    }
};

struct AddRmsNormBinarySymmetricI4Match {
    const GraphNode *  add_node = nullptr;
    const Value *      lhs      = nullptr;
    const Value *      rhs      = nullptr;
    const Value *      residual = nullptr;
    RmsNormBinaryMatch rms_binary;
    size_t             add_node_index = 0;

    bool matched() const {
        return add_node != nullptr && lhs != nullptr && rhs != nullptr && residual != nullptr && rms_binary.matched();
    }
};

struct RmsNormGateMatch {
    std::vector<const GraphNode *> covered;
    const Value *                  input       = nullptr;
    const Value *                  weight      = nullptr;
    const Value *                  raw_gate    = nullptr;
    const Value *                  output      = nullptr;
    int64_t                        hidden_size = 0;
    int64_t                        token_count = 0;
    float                          epsilon     = 0.0f;
    UnaryKind                      gate_op     = UnaryKind::Silu;

    bool matched() const {
        return !covered.empty() && input != nullptr && weight != nullptr && raw_gate != nullptr && output != nullptr;
    }
};

template <size_t N> static bool pairwise_distinct_storage_roots(const std::array<const Value *, N> & values) {
    for (size_t lhs = 0; lhs < values.size(); ++lhs) {
        for (size_t rhs = lhs + 1; rhs < values.size(); ++rhs) {
            if (values[lhs]->storage_root == values[rhs]->storage_root) {
                return false;
            }
        }
    }
    return true;
}

static bool rope_has_set_rows_consumer(const Graph & graph, const GraphNode & rope) {
    std::vector<ValueId> pending = { rope.output };
    for (size_t i = 0; i < pending.size(); ++i) {
        for (const GraphNode * consumer : graph.index().consumers(pending[i])) {
            if (consumer == nullptr) {
                continue;
            }
            if (consumer->op == GGML_OP_SET_ROWS) {
                return true;
            }
            if (is_layout_alias_node(graph, *consumer)) {
                pending.push_back(consumer->output);
            }
        }
    }
    return false;
}

static bool fused_input_is_available(const Graph & graph, ValueId value, const std::vector<bool> & covered_nodes) {
    const GraphNode * producer = graph.index().producer(value);
    if (producer == nullptr) {
        return true;
    }
    size_t producer_index = 0;
    return graph.index().node_index(producer, producer_index) && producer_index < covered_nodes.size() &&
           covered_nodes[producer_index];
}

static RmsNormMulRopeMatch match_rmsnorm_mul_rope_f32(const Graph & graph, const GraphNode * rms) {
    RmsNormMulRopeMatch match;
    if (rms == nullptr || rms->op != GGML_OP_RMS_NORM || rms->inputs.size() != 1 || !graph.has_index()) {
        return match;
    }

    const std::vector<const GraphNode *> & rms_consumers = graph.index().consumers(rms->output);
    if (rms_consumers.size() != 1 || rms_consumers.front() == nullptr || rms_consumers.front()->op != GGML_OP_MUL ||
        rms_consumers.front()->inputs.size() != 2) {
        return {};
    }
    const GraphNode *                      binary           = rms_consumers.front();
    const std::vector<const GraphNode *> & binary_consumers = graph.index().consumers(binary->output);
    if (binary_consumers.size() != 1 || binary_consumers.front() == nullptr ||
        binary_consumers.front()->op != GGML_OP_ROPE || binary_consumers.front()->inputs.size() != 2 ||
        binary_consumers.front()->inputs[0] != binary->output) {
        return {};
    }
    const GraphNode * rope = binary_consumers.front();

    const Value * input         = graph_value(graph, rms->inputs[0]);
    const Value * rms_output    = graph_value(graph, rms->output);
    const Value * binary_output = graph_value(graph, binary->output);
    const Value * positions     = graph_value(graph, rope->inputs[1]);
    const Value * output        = graph_value(graph, rope->output);
    const Value * weight        = nullptr;
    if (binary->inputs[0] == rms->output) {
        weight = graph_value(graph, binary->inputs[1]);
    } else if (binary->inputs[1] == rms->output) {
        weight = graph_value(graph, binary->inputs[0]);
    }
    const RmsNormParams * rms_params  = op_params_as<RmsNormParams>(rms->params);
    const RopeParams *    rope_params = op_params_as<RopeParams>(rope->params);
    if (input == nullptr || rms_output == nullptr || binary_output == nullptr || positions == nullptr ||
        output == nullptr || weight == nullptr || rms_params == nullptr || rope_params == nullptr) {
        return {};
    }

    if (input->type != GGML_TYPE_F32 || rms_output->type != GGML_TYPE_F32 || binary_output->type != GGML_TYPE_F32 ||
        output->type != GGML_TYPE_F32 || weight->type != GGML_TYPE_F32 || positions->type != GGML_TYPE_I32 ||
        input->ne != rms_output->ne || input->ne != binary_output->ne || input->ne != output->ne ||
        !rms_output->contiguous || !binary_output->contiguous || !output->contiguous || !weight->contiguous ||
        !positions->contiguous || weight->ne[0] != input->ne[0] || weight->ne[1] != 1 || weight->ne[2] != 1 ||
        weight->ne[3] != 1 || input->ne[0] < 2 || input->ne[0] > 512 || input->ne[0] % 2 != 0 || input->ne[1] < 1 ||
        input->ne[1] > 1048576 || input->ne[2] < 1 || input->ne[2] > 1048576 || input->ne[3] < 1 ||
        input->ne[3] > 1024 || input->nb[0] != sizeof(float) || output->nb[0] != sizeof(float)) {
        return {};
    }
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (input->nb[i] % sizeof(float) != 0 || output->nb[i] % sizeof(float) != 0) {
            return {};
        }
    }

    const bool sectioned_mode = rope_params->mode == GGML_ROPE_TYPE_MROPE || rope_params->mode == GGML_ROPE_TYPE_IMROPE;
    const bool lfm_neox_mode  = rope_params->mode == GGML_ROPE_TYPE_NEOX && input->ne[0] == 64 &&
                               rope_params->n_dims == 64 && !rope_has_set_rows_consumer(graph, *rope);
    int64_t    section_count  = 0;
    if (sectioned_mode) {
        for (int section : rope_params->sections) {
            if (section < 0) {
                return {};
            }
            section_count += section;
        }
    }
    const int64_t required_positions = sectioned_mode ? 4 * input->ne[2] : input->ne[2];
    if ((!sectioned_mode && !lfm_neox_mode) || positions->element_count < required_positions ||
        rope_params->n_dims < 2 || rope_params->n_dims > input->ne[0] || rope_params->n_dims % 2 != 0 ||
        (sectioned_mode && section_count != rope_params->n_dims / 2) || !std::isfinite(rms_params->eps) ||
        rms_params->eps <= 0.0f || !std::isfinite(rope_params->freq_base) || rope_params->freq_base <= 0.0f ||
        !std::isfinite(rope_params->freq_scale) || rope_params->freq_scale <= 0.0f ||
        !std::isfinite(rope_params->attn_factor) || rope_params->attn_factor <= 0.0f ||
        rope_params->ext_factor != 0.0f || output->storage_root == input->storage_root ||
        output->storage_root == weight->storage_root || output->storage_root == positions->storage_root) {
        return {};
    }

    match.rms         = rms;
    match.binary      = binary;
    match.rope        = rope;
    match.input       = input;
    match.weight      = weight;
    match.positions   = positions;
    match.output      = output;
    match.rms_params  = rms_params;
    match.rope_params = rope_params;
    return match;
}

static RmsNormMatch match_rmsnorm_f32(const Graph & graph, const GraphNode * node, size_t node_index) {
    RmsNormMatch match;
    if (node == nullptr || node->op != GGML_OP_RMS_NORM || node->inputs.size() != 1) {
        return match;
    }

    const RmsNormParams * rms_params = op_params_as<RmsNormParams>(node->params);
    if (rms_params == nullptr || !std::isfinite(rms_params->eps) || rms_params->eps <= 0.0f) {
        return {};
    }

    const Value * input  = graph_value(graph, node->inputs[0]);
    const Value * output = graph_value(graph, node->output);
    if (input == nullptr || output == nullptr) {
        return {};
    }
    if (input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 || !output->contiguous ||
        !same_shape(*input, *output)) {
        return {};
    }
    if (input->storage_root == output->storage_root) {
        return {};
    }

    const int64_t hidden_size = output->ne[0];
    if (!is_supported_f32_hidden_size(hidden_size)) {
        return {};
    }
    if (hidden_size == 0 || output->element_count <= 0 || output->element_count % hidden_size != 0) {
        return {};
    }
    const int64_t token_count = output->element_count / hidden_size;
    if (!is_supported_token_count(token_count)) {
        return {};
    }
    int64_t input_stride = 0;
    size_t  input_span   = 0;
    if (!supported_rmsnorm_input_layout(*input, hidden_size, token_count, input_stride, input_span)) {
        return {};
    }

    match.rms_node       = node;
    match.input          = input;
    match.output         = output;
    match.input_span     = input_span;
    match.rms_node_index = node_index;
    match.input_stride   = input_stride;
    match.hidden_size    = hidden_size;
    match.token_count    = token_count;
    match.epsilon        = rms_params->eps;
    return match;
}

static RmsNormBinaryMatch match_rmsnorm_binary_f32(const Graph &     graph,
                                                   const GraphNode * node,
                                                   size_t            node_index,
                                                   bool              allow_strided_input = false) {
    RmsNormBinaryMatch match;
    if (node == nullptr || node->op != GGML_OP_RMS_NORM || node->inputs.size() != 1 || !graph.has_index()) {
        return match;
    }

    const RmsNormParams * rms_params = op_params_as<RmsNormParams>(node->params);
    if (rms_params == nullptr || !std::isfinite(rms_params->eps) || rms_params->eps <= 0.0f) {
        return {};
    }
    const std::vector<const GraphNode *> & consumers = graph.index().consumers(node->output);
    if (consumers.size() != 1) {
        return {};
    }
    const GraphNode * binary_node = consumers.front();
    size_t            binary_node_index;
    const CommonBinaryMatch binary_match =
        common_match_binary_consumer(binary_node, node->output, kCommonArithmeticBinaryKinds);
    if (!binary_match.matched() || !graph.index().node_index(binary_node, binary_node_index)) {
        return {};
    }
    if (!binary_match.producer_is_lhs && common_binary_kind_requires_order(binary_match.kind)) {
        return {};
    }

    const Value * input  = graph_value(graph, node->inputs[0]);
    const Value * rms    = graph_value(graph, node->output);
    const Value * rhs    = graph_value(graph, binary_match.operand);
    const Value * output = graph_value(graph, binary_node->output);
    if (input == nullptr || rms == nullptr || rhs == nullptr || output == nullptr) {
        return {};
    }
    if (input->type != GGML_TYPE_F32 || rms->type != GGML_TYPE_F32 || rhs->type != GGML_TYPE_F32 ||
        output->type != GGML_TYPE_F32) {
        return {};
    }
    if (!rms->contiguous || !rhs->contiguous || !output->contiguous) {
        return {};
    }
    if (!same_shape(*input, *rms) || !same_shape(*input, *output)) {
        return {};
    }

    const int64_t hidden_size = output->ne[0];
    if (!is_supported_binary_f32_hidden_size(hidden_size) || !is_weight_shape(*rhs, hidden_size)) {
        return {};
    }
    if (hidden_size == 0 || output->element_count <= 0 || output->element_count % hidden_size != 0) {
        return {};
    }
    const int64_t token_count = output->element_count / hidden_size;
    if (!is_supported_token_count(token_count)) {
        return {};
    }
    int64_t input_stride = 0;
    size_t  input_span   = 0;
    if (!supported_rmsnorm_input_layout(*input, hidden_size, token_count, input_stride, input_span) ||
        (!allow_strided_input && input_stride != hidden_size) ||
        !pairwise_distinct_storage_roots(std::array<const Value *, 3>{ input, rhs, output })) {
        return {};
    }

    match.rms_node          = node;
    match.binary_node       = binary_node;
    match.input             = input;
    match.rhs               = rhs;
    match.output            = output;
    match.rms_node_index    = node_index;
    match.binary_node_index = binary_node_index;
    match.hidden_size       = hidden_size;
    match.token_count       = token_count;
    match.input_stride      = input_stride;
    match.input_span        = input_span;
    match.op                = binary_match.kind;
    match.epsilon           = rms_params->eps;
    return match;
}

static RmsNormMulAddMatch match_rmsnorm_mul_add_f32(const DispatchMatchContext & context) {
    RmsNormMulAddMatch match;
    match.rms_binary = match_rmsnorm_binary_f32(context.graph, context.root_node, context.root_index);
    if (!match.rms_binary.matched()) {
        return {};
    }

    const std::vector<const GraphNode *> & consumers = context.graph.index().consumers(match.rms_binary.output->id);
    if (consumers.size() != 1) {
        return {};
    }
    const GraphNode * add = consumers.front();
    size_t            add_index = 0;
    const CommonBinaryMatch output_binary =
        common_match_binary_consumer(add, match.rms_binary.output->id, kCommonArithmeticBinaryKinds);
    if (!output_binary.matched() || !context.graph.index().node_index(add, add_index)) {
        return {};
    }
    const Value * residual   = graph_value(context.graph, output_binary.operand);
    const Value * output   = graph_value(context.graph, add->output);
    const Value * rms_output = graph_value(context.graph, match.rms_binary.rms_node->output);
    if (residual == nullptr || output == nullptr || rms_output == nullptr || residual->type != GGML_TYPE_F32 ||
        output->type != GGML_TYPE_F32 || !residual->contiguous || !output->contiguous ||
        !same_shape(*residual, *match.rms_binary.input) || !same_shape(*output, *match.rms_binary.input) ||
        !pairwise_distinct_storage_roots(std::array<const Value *, 6>{
            match.rms_binary.input, match.rms_binary.rhs, rms_output, match.rms_binary.output, residual, output })) {
        return {};
    }
    const size_t nodes_size = context.covered_nodes.size();
    if (match.rms_binary.rms_node_index >= nodes_size || match.rms_binary.binary_node_index >= nodes_size ||
        add_index >= nodes_size || context.covered_nodes[match.rms_binary.rms_node_index] ||
        context.covered_nodes[match.rms_binary.binary_node_index] || context.covered_nodes[add_index]) {
        return {};
    }
    const auto available = [&](const Value & value) {
        const GraphNode * producer = context.graph.index().producer(value.id);
        if (producer == nullptr) {
            return true;
        }
        size_t producer_index = 0;
        return context.graph.index().node_index(producer, producer_index) && producer_index < nodes_size &&
               context.covered_nodes[producer_index];
    };
    if (!available(*match.rms_binary.input) || !available(*match.rms_binary.rhs) || !available(*residual)) {
        return {};
    }

    match.add_node       = add;
    match.residual       = residual;
    match.output         = output;
    match.add_node_index = add_index;
    match.output_op      = output_binary.kind;
    match.binary_lhs     = output_binary.producer_is_lhs;
    return match;
}

static AddRmsNormBinarySymmetricI4Match match_add_rmsnorm_binary_symmetric_i4(const Graph &     graph,
                                                                              const GraphNode * node,
                                                                              size_t            node_index) {
    AddRmsNormBinarySymmetricI4Match match;
    if (node == nullptr || node->op != GGML_OP_ADD || node->inputs.size() != 2 || !graph.has_index()) {
        return match;
    }

    const Value * lhs      = graph_value(graph, node->inputs[0]);
    const Value * rhs      = graph_value(graph, node->inputs[1]);
    const Value * residual = graph_value(graph, node->output);
    if (lhs == nullptr || rhs == nullptr || residual == nullptr || lhs->type != GGML_TYPE_F32 ||
        rhs->type != GGML_TYPE_F32 || residual->type != GGML_TYPE_F32 || !lhs->contiguous || !rhs->contiguous ||
        !residual->contiguous || !same_shape(*lhs, *residual) || !same_shape(*rhs, *residual)) {
        return {};
    }

    const GraphNode * rms_node = nullptr;
    for (const GraphNode * consumer : graph.index().consumers(residual->id)) {
        if (consumer == nullptr || consumer->op != GGML_OP_RMS_NORM) {
            continue;
        }
        if (rms_node != nullptr) {
            return {};
        }
        rms_node = consumer;
    }
    size_t rms_node_index = 0;
    if (rms_node == nullptr || !graph.index().node_index(rms_node, rms_node_index)) {
        return {};
    }

    RmsNormBinaryMatch rms_binary = match_rmsnorm_binary_f32(graph, rms_node, rms_node_index);
    if (!rms_binary.matched() || rms_binary.input->id != residual->id || rms_binary.op != BinaryKind::Mul ||
        !is_supported_symmetric_i4_hidden_size(rms_binary.hidden_size) ||
        !common_has_symmetric_i4_lowrow_consumer(graph, *rms_binary.output) ||
        !pairwise_distinct_storage_roots(
            std::array<const Value *, 5>{ lhs, rhs, residual, rms_binary.rhs, rms_binary.output })) {
        return {};
    }

    match.add_node       = node;
    match.lhs            = lhs;
    match.rhs            = rhs;
    match.residual       = residual;
    match.rms_binary     = rms_binary;
    match.add_node_index = node_index;
    return match;
}

static RmsNormGateMatch match_rmsnorm_gate(const Graph & graph, const GraphNode * node, size_t node_index) {
    RmsNormGateMatch match;
    const RmsNormBinaryMatch rms_binary = match_rmsnorm_binary_f32(graph, node, node_index);
    if (!rms_binary.matched() || rms_binary.op != BinaryKind::Mul || rms_binary.hidden_size % 64 != 0) {
        return match;
    }

    const GraphNode * terminal = common_find_only_consumer_with_op(graph, rms_binary.output->id, GGML_OP_MUL);
    if (terminal == nullptr || !common_binary_node_is_mul(*terminal)) {
        return {};
    }
    const ValueId activated_id =
        terminal->inputs[0] == rms_binary.output->id ? terminal->inputs[1] : terminal->inputs[0];
    const Value *       activated    = graph_value(graph, activated_id);
    const GraphNode *   unary        = activated != nullptr ? graph.index().producer(activated->id) : nullptr;
    const UnaryParams * unary_params = unary != nullptr ? op_params_as<UnaryParams>(unary->params) : nullptr;
    if (unary == nullptr || unary->op != GGML_OP_UNARY || unary->inputs.size() != 1 || unary_params == nullptr ||
        !unary_kind_supported(unary_params->op) ||
        common_find_only_consumer_with_op(graph, activated->id, GGML_OP_MUL) != terminal) {
        return {};
    }

    const Value *     gate_input   = graph_value(graph, unary->inputs[0]);
    const GraphNode * gate_reshape = gate_input != nullptr ? graph.index().producer(gate_input->id) : nullptr;
    const bool        has_gate_reshape =
        gate_reshape != nullptr && gate_reshape->op == GGML_OP_RESHAPE && gate_reshape->inputs.size() == 1;
    const Value * raw_gate = has_gate_reshape ? graph_value(graph, gate_reshape->inputs[0]) : gate_input;
    const Value * output   = graph_value(graph, terminal->output);
    if (gate_input == nullptr || raw_gate == nullptr || output == nullptr || gate_input->type != GGML_TYPE_F32 ||
        activated->type != GGML_TYPE_F32 || raw_gate->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 ||
        !gate_input->contiguous || !activated->contiguous || !raw_gate->contiguous || !output->contiguous ||
        !same_shape(*rms_binary.input, *gate_input) || !same_shape(*rms_binary.input, *activated) ||
        !same_shape(*rms_binary.input, *output) || raw_gate->element_count != output->element_count ||
        graph.index().consumers(gate_input->id).size() != 1 ||
        !pairwise_distinct_storage_roots(
            std::array<const Value *, 4>{ rms_binary.input, rms_binary.rhs, raw_gate, output })) {
        return {};
    }

    match.covered = { rms_binary.rms_node, rms_binary.binary_node };
    if (has_gate_reshape) {
        match.covered.push_back(gate_reshape);
    }
    match.covered.push_back(unary);
    match.covered.push_back(terminal);
    match.input       = rms_binary.input;
    match.weight      = rms_binary.rhs;
    match.raw_gate    = raw_gate;
    match.output      = output;
    match.hidden_size = rms_binary.hidden_size;
    match.token_count = rms_binary.token_count;
    match.epsilon     = rms_binary.epsilon;
    match.gate_op     = unary_params->op;
    return match;
}

static std::string to_config_value(int64_t value) {
    return std::to_string(value);
}

static std::string to_config_value(float value) {
    std::ostringstream out;
    out.precision(9);
    out << value;
    return out.str();
}

}  // namespace

static bool match_rmsnorm_mul_rope_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const RmsNormMulRopeMatch fused = match_rmsnorm_mul_rope_f32(context.graph, context.root_node);
    if (!fused.matched()) {
        return false;
    }
    if (fused.rope_params->mode == GGML_ROPE_TYPE_NEOX &&
        (!fused_input_is_available(context.graph, fused.input->id, context.covered_nodes) ||
         !fused_input_is_available(context.graph, fused.weight->id, context.covered_nodes) ||
         !fused_input_is_available(context.graph, fused.positions->id, context.covered_nodes))) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kRmsNormMulRopeF32Kernel);
    auto & config   = dispatch.kernel.compile_parameters;
    config.emplace("ggml.rmsnorm_mul_rope.hidden_size", to_config_value(fused.input->ne[0]));
    config.emplace("ggml.rmsnorm_mul_rope.ne1", to_config_value(fused.input->ne[1]));
    config.emplace("ggml.rmsnorm_mul_rope.ne2", to_config_value(fused.input->ne[2]));
    config.emplace("ggml.rmsnorm_mul_rope.ne3", to_config_value(fused.input->ne[3]));
    config.emplace("ggml.rmsnorm_mul_rope.input_stride1",
                   to_config_value(static_cast<int64_t>(fused.input->nb[1] / sizeof(float))));
    config.emplace("ggml.rmsnorm_mul_rope.input_stride2",
                   to_config_value(static_cast<int64_t>(fused.input->nb[2] / sizeof(float))));
    config.emplace("ggml.rmsnorm_mul_rope.input_stride3",
                   to_config_value(static_cast<int64_t>(fused.input->nb[3] / sizeof(float))));
    config.emplace("ggml.rmsnorm_mul_rope.output_stride1",
                   to_config_value(static_cast<int64_t>(fused.output->nb[1] / sizeof(float))));
    config.emplace("ggml.rmsnorm_mul_rope.output_stride2",
                   to_config_value(static_cast<int64_t>(fused.output->nb[2] / sizeof(float))));
    config.emplace("ggml.rmsnorm_mul_rope.output_stride3",
                   to_config_value(static_cast<int64_t>(fused.output->nb[3] / sizeof(float))));
    config.emplace("ggml.rmsnorm_mul_rope.n_dims", to_config_value(static_cast<int64_t>(fused.rope_params->n_dims)));
    const bool neox_mode = fused.rope_params->mode == GGML_ROPE_TYPE_NEOX;
    config.emplace("ggml.rmsnorm_mul_rope.section0",
                   to_config_value(neox_mode ? static_cast<int64_t>(fused.rope_params->n_dims / 2) :
                                               static_cast<int64_t>(fused.rope_params->sections[0])));
    config.emplace("ggml.rmsnorm_mul_rope.section1",
                   to_config_value(static_cast<int64_t>(fused.rope_params->sections[1])));
    config.emplace("ggml.rmsnorm_mul_rope.section2",
                   to_config_value(static_cast<int64_t>(fused.rope_params->sections[2])));
    config.emplace("ggml.rmsnorm_mul_rope.section3",
                   to_config_value(static_cast<int64_t>(fused.rope_params->sections[3])));
    config.emplace("ggml.rmsnorm_mul_rope.mode", to_config_value(static_cast<int64_t>(fused.rope_params->mode)));
    config.emplace("ggml.rmsnorm_mul_rope.workgroup_size", "256");
    config.emplace("ggml.rmsnorm_mul_rope.epsilon", to_config_value(fused.rms_params->eps));
    config.emplace("ggml.rmsnorm_mul_rope.freq_base", to_config_value(fused.rope_params->freq_base));
    config.emplace("ggml.rmsnorm_mul_rope.freq_scale", to_config_value(fused.rope_params->freq_scale));
    config.emplace("ggml.rmsnorm_mul_rope.attn_factor", to_config_value(fused.rope_params->attn_factor));
    dispatch.bindings.push_back({ fused.input->id, 0, fused.input->byte_count });
    dispatch.bindings.push_back({ fused.weight->id, 0, fused.weight->byte_count });
    dispatch.bindings.push_back({ fused.positions->id, 0, fused.positions->byte_count });
    dispatch.bindings.push_back({ fused.output->id, 0, fused.output->byte_count });

    match.covered_nodes.push_back(context.root_index);
    if (!append_covered_node_index_once(context.graph, context.covered_nodes, fused.binary, match.covered_nodes) ||
        !append_covered_node_index_once(context.graph, context.covered_nodes, fused.rope, match.covered_nodes)) {
        return false;
    }
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_add_rmsnorm_binary_symmetric_i4_dispatch(const DispatchMatchContext & context,
                                                           DispatchMatch &              match) {
    const AddRmsNormBinarySymmetricI4Match fused =
        match_add_rmsnorm_binary_symmetric_i4(context.graph, context.root_node, context.root_index);
    if (!fused.matched() || fused.add_node_index >= context.covered_nodes.size() ||
        fused.rms_binary.rms_node_index >= context.covered_nodes.size() ||
        fused.rms_binary.binary_node_index >= context.covered_nodes.size() ||
        context.covered_nodes[fused.add_node_index] || context.covered_nodes[fused.rms_binary.rms_node_index] ||
        context.covered_nodes[fused.rms_binary.binary_node_index]) {
        return false;
    }

    const CommonSymmetricI4ActivationLayout activation_layout =
        common_symmetric_i4_activation_layout(fused.rms_binary.hidden_size, fused.rms_binary.token_count);
    if (activation_layout.total_bytes == 0) {
        return false;
    }
    const ValueId activation = context.next_plan_value;

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kAddRmsNormBinarySymmetricI4K32Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", fused.rms_binary.token_count);
    dispatch.kernel.compile_parameters.emplace("ggml.add_rmsnorm_binary_symmetric_i4.hidden_size",
                                               to_config_value(fused.rms_binary.hidden_size));
    dispatch.kernel.compile_parameters.emplace("ggml.add_rmsnorm_binary_symmetric_i4.rms_epsilon",
                                               to_config_value(fused.rms_binary.epsilon));
    dispatch.bindings.push_back({ fused.lhs->id, 0, fused.lhs->byte_count });
    dispatch.bindings.push_back({ fused.rhs->id, 0, fused.rhs->byte_count });
    dispatch.bindings.push_back({ fused.residual->id, 0, fused.residual->byte_count });
    dispatch.bindings.push_back({ fused.rms_binary.rhs->id, 0, fused.rms_binary.rhs->byte_count });
    dispatch.bindings.push_back({ fused.rms_binary.output->id, 0, fused.rms_binary.output->byte_count });
    dispatch.bindings.push_back({ activation, 0, activation_layout.payload_bytes });
    dispatch.bindings.push_back({ activation, activation_layout.scales_offset, activation_layout.metadata_bytes });
    dispatch.bindings.push_back({ activation, activation_layout.sums_offset, activation_layout.metadata_bytes });

    Status metadata_status;
    if (!match.metadata.append_alternate_value(
            { fused.rms_binary.output->id, activation, GGML_TYPE_COUNT, activation_layout.total_bytes,
              kCommonSymmetricI4K32ActivationAlternateName },
            metadata_status)) {
        match.status.append(metadata_status);
        return false;
    }

    match.covered_nodes.push_back(fused.add_node_index);
    match.covered_nodes.push_back(fused.rms_binary.rms_node_index);
    match.covered_nodes.push_back(fused.rms_binary.binary_node_index);
    match.dispatches.push_back(std::move(dispatch));
    match.transients.push_back(
        { activation, kCommonSymmetricI4K32ActivationAlternateName, activation_layout.total_bytes, 256 });
    return match.status.success();
}

static bool match_rmsnorm_binary_symmetric_i4_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const std::vector<GraphNode> & nodes = context.graph.nodes();
    if (context.root_index >= nodes.size()) {
        return false;
    }
    const RmsNormBinaryMatch fused =
        match_rmsnorm_binary_f32(context.graph, &nodes[context.root_index], context.root_index);
    if (!fused.matched() || fused.op != BinaryKind::Mul || !is_supported_symmetric_i4_hidden_size(fused.hidden_size) ||
        fused.token_count > 16 || fused.rms_node_index >= context.covered_nodes.size() ||
        fused.binary_node_index >= context.covered_nodes.size() || context.covered_nodes[fused.rms_node_index] ||
        context.covered_nodes[fused.binary_node_index] ||
        !common_has_symmetric_i4_lowrow_consumer(context.graph, *fused.output) ||
        !pairwise_distinct_storage_roots(std::array<const Value *, 3>{ fused.input, fused.rhs, fused.output })) {
        return false;
    }

    const CommonSymmetricI4ActivationLayout activation_layout =
        common_symmetric_i4_activation_layout(fused.hidden_size, fused.token_count);
    if (activation_layout.total_bytes == 0) {
        return false;
    }
    const ValueId activation = context.next_plan_value;

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kRmsNormBinarySymmetricI4K32Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", fused.token_count);
    dispatch.kernel.compile_parameters.emplace("ggml.rmsnorm_binary_symmetric_i4.hidden_size",
                                               to_config_value(fused.hidden_size));
    dispatch.kernel.compile_parameters.emplace("ggml.rmsnorm_binary_symmetric_i4.rms_epsilon",
                                               to_config_value(fused.epsilon));
    dispatch.bindings.push_back({ fused.input->id, 0, fused.input->byte_count });
    dispatch.bindings.push_back({ fused.rhs->id, 0, fused.rhs->byte_count });
    dispatch.bindings.push_back({ fused.output->id, 0, fused.output->byte_count });
    dispatch.bindings.push_back({ activation, 0, activation_layout.payload_bytes });
    dispatch.bindings.push_back({ activation, activation_layout.scales_offset, activation_layout.metadata_bytes });
    dispatch.bindings.push_back({ activation, activation_layout.sums_offset, activation_layout.metadata_bytes });

    Status metadata_status;
    if (!match.metadata.append_alternate_value(
            { fused.output->id, activation, GGML_TYPE_COUNT, activation_layout.total_bytes,
              kCommonSymmetricI4K32ActivationAlternateName },
            metadata_status)) {
        match.status.append(metadata_status);
        return false;
    }

    match.covered_nodes.push_back(fused.rms_node_index);
    match.covered_nodes.push_back(fused.binary_node_index);
    match.dispatches.push_back(std::move(dispatch));
    match.transients.push_back(
        { activation, kCommonSymmetricI4K32ActivationAlternateName, activation_layout.total_bytes, 256 });
    return match.status.success();
}

static bool match_rmsnorm_gate_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const RmsNormGateMatch fused = match_rmsnorm_gate(context.graph, context.root_node, context.root_index);
    if (!fused.matched()) {
        return false;
    }

    const bool use_i4 =
        fused.gate_op == UnaryKind::Silu && common_has_symmetric_i4_lowrow_consumer(context.graph, *fused.output);
    if (!use_i4) {
        if (fused.hidden_size > 1024) {
            return false;
        }
        const Value * packed_input = fused.output;
        const GraphNode * consumer =
            common_find_only_consumer_with_op(context.graph, packed_input->id, GGML_OP_MUL_MAT);
        if (consumer == nullptr) {
            const GraphNode * reshape =
                common_find_only_consumer_with_op(context.graph, packed_input->id, GGML_OP_RESHAPE);
            const Value * reshaped = reshape != nullptr ? graph_value(context.graph, reshape->output) : nullptr;
            if (reshaped != nullptr && is_layout_alias_node(context.graph, *reshape) && reshaped->contiguous &&
                same_full_value_range(*packed_input, *reshaped)) {
                packed_input = reshaped;
                consumer = common_find_only_consumer_with_op(context.graph, packed_input->id, GGML_OP_MUL_MAT);
            }
        }
        const CommonMulMatMatch projection =
            common_match_mul_mat_any_format(context.graph, consumer, kRmsNormGateF32F16Kernel, false);
        const bool packed_output = projection.matched() && projection.input->id == packed_input->id &&
                                   projection.weight->alias_source.value < 0 &&
                                   common_mul_mat_uses_k16_major_f16(projection.weight_format, projection.input_size,
                                                                     projection.output_size, projection.token_count);
        const bool q8_output =
            !packed_output && packed_input->ne[1] <= 5 && is_packed_q8_consumer(context.graph, consumer, *packed_input);
        const size_t activation_bytes =
            q8_output ? q8_1_x4_byte_count(fused.token_count, fused.hidden_size) : fused.output->byte_count / 2;
        const ValueId          activation = context.next_plan_value;
        const char * activation_name = q8_output ? "common.rmsnorm_gate.q8_1_x4" :
                                        packed_output ? "common.rmsnorm_gate.k16_major_f16" :
                                                        "common.rmsnorm_gate.f16";
        Dispatch dispatch;
        dispatch.kernel =
            make_kernel_specialization(q8_output ? kRmsNormGateF32Q8_1X4Kernel : kRmsNormGateF32F16Kernel);
        dispatch.kernel.integer_parameters.emplace("token_count", fused.token_count);
        dispatch.kernel.compile_parameters.emplace("ggml.rmsnorm_gate_f32.hidden_size",
                                                   to_config_value(fused.hidden_size));
        dispatch.kernel.compile_parameters.emplace("ggml.rmsnorm_gate_f32.rms_epsilon", to_config_value(fused.epsilon));
        dispatch.kernel.compile_parameters.emplace("ggml.rmsnorm_gate_f32.gate_op",
                                                   std::to_string(unary_kind_config_value(fused.gate_op)));
        dispatch.kernel.compile_parameters.emplace("ggml.rmsnorm_gate_f32.f16_output_row_width",
                                                   to_config_value(packed_output ? projection.input_size : 0));
        dispatch.bindings.push_back({ fused.input->id, 0, fused.input->byte_count });
        dispatch.bindings.push_back({ fused.weight->id, 0, fused.weight->byte_count });
        dispatch.bindings.push_back({ fused.raw_gate->id, 0, fused.raw_gate->byte_count });
        dispatch.bindings.push_back({ fused.output->id, 0, fused.output->byte_count });
        dispatch.bindings.push_back({ activation, 0, activation_bytes });
        Status status;
        const bool recorded =
            packed_output ?
            match.metadata.append_generated_resource(
                    { packed_input->id, GeneratedResourceRole::F16K16Major, activation, activation_bytes, {} },
                    status) :
            match.metadata.append_alternate_value(
                    { fused.output->id, activation, q8_output ? GGML_TYPE_Q8_1 : GGML_TYPE_F16, activation_bytes,
                      activation_name },
                    status);
        if (!recorded) {
            match.status.append(status);
            return false;
        }
        for (const GraphNode * covered : fused.covered) {
            if (!append_covered_node_index_once(context.graph, context.covered_nodes, covered, match.covered_nodes)) {
                return false;
            }
        }
        match.transients.push_back({ activation, activation_name, activation_bytes, 256 });
        match.dispatches.push_back(std::move(dispatch));
        return true;
    }

    const CommonSymmetricI4ActivationLayout activation_layout =
        common_symmetric_i4_activation_layout(fused.hidden_size, fused.token_count);
    if (activation_layout.total_bytes == 0) {
        return false;
    }
    const ValueId activation = context.next_plan_value;

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kRmsNormGateSiluMulSymmetricI4K32Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", fused.token_count);
    dispatch.kernel.compile_parameters.emplace("ggml.rmsnorm_gate_silu_mul_symmetric_i4.hidden_size",
                                               to_config_value(fused.hidden_size));
    dispatch.kernel.compile_parameters.emplace("ggml.rmsnorm_gate_silu_mul_symmetric_i4.rms_epsilon",
                                               to_config_value(fused.epsilon));
    dispatch.bindings.push_back({ fused.input->id, 0, fused.input->byte_count });
    dispatch.bindings.push_back({ fused.weight->id, 0, fused.weight->byte_count });
    dispatch.bindings.push_back({ fused.raw_gate->id, 0, fused.raw_gate->byte_count });
    dispatch.bindings.push_back({ fused.output->id, 0, fused.output->byte_count });
    dispatch.bindings.push_back({ activation, 0, activation_layout.payload_bytes });
    dispatch.bindings.push_back({ activation, activation_layout.scales_offset, activation_layout.metadata_bytes });
    dispatch.bindings.push_back({ activation, activation_layout.sums_offset, activation_layout.metadata_bytes });

    Status metadata_status;
    if (!match.metadata.append_alternate_value(
            { fused.output->id, activation, GGML_TYPE_COUNT, activation_layout.total_bytes,
              kCommonSymmetricI4K32ActivationAlternateName },
            metadata_status)) {
        match.status.append(metadata_status);
        return false;
    }
    for (const GraphNode * covered : fused.covered) {
        if (!append_covered_node_index_once(context.graph, context.covered_nodes, covered, match.covered_nodes)) {
            return false;
        }
    }
    match.dispatches.push_back(std::move(dispatch));
    match.transients.push_back(
        { activation, kCommonSymmetricI4K32ActivationAlternateName, activation_layout.total_bytes, 256 });
    return match.status.success();
}

static bool rmsnorm_k16_publication_enabled(const RmsNormBinaryMatch & rms) {
    const bool qualified_shape =
        (rms.token_count == 512 && (rms.hidden_size == 2048 || rms.hidden_size == 3072 || rms.hidden_size == 4096 ||
                                   rms.hidden_size == 5120 || rms.hidden_size == 6144 || rms.hidden_size == 8192)) ||
        (rms.token_count == 1024 && rms.hidden_size == 5120);
    return qualified_shape && rms.op == BinaryKind::Mul && rms.output->ne[1] == rms.token_count &&
           rms.output->ne[2] == 1 && rms.output->ne[3] == 1 &&
           pairwise_distinct_storage_roots(std::array<const Value *, 3>{ rms.input, rms.rhs, rms.output });
}

static bool rmsnorm_f16_publication_enabled(const RmsNormBinaryMatch & rms) {
    return rms.token_count >= 128 && rms.output->ne[1] == rms.token_count && rms.output->ne[2] == 1 &&
           rms.output->ne[3] == 1;
}

static bool match_rmsnorm_binary_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const std::vector<GraphNode> & nodes = context.graph.nodes();
    if (context.root_index >= nodes.size()) {
        return false;
    }
    const RmsNormBinaryMatch rms_match =
        match_rmsnorm_binary_f32(context.graph, &nodes[context.root_index], context.root_index, true);
    if (!rms_match.matched() || rms_match.rms_node_index >= context.covered_nodes.size() ||
        rms_match.binary_node_index >= context.covered_nodes.size() ||
        context.covered_nodes[rms_match.rms_node_index] || context.covered_nodes[rms_match.binary_node_index]) {
        return false;
    }

    const bool strided_input = rms_match.input_stride != rms_match.hidden_size;
    const CommonActivationPublicationCandidate publication_candidates[] = {
        { CommonActivationPublicationFormat::F16K16Major, common_accepts_f16_k16_major_mul_mat,
          CommonActivationConsumerTraversal::Direct, !strided_input && rmsnorm_k16_publication_enabled(rms_match) },
        { CommonActivationPublicationFormat::F16Row, common_accepts_f16_row_prefill_mul_mat,
          CommonActivationConsumerTraversal::Direct, !strided_input && rmsnorm_f16_publication_enabled(rms_match) },
    };
    const CommonActivationPublicationDemand publication_demand = common_select_activation_publication(
        context, *rms_match.output, publication_candidates, std::size(publication_candidates));
    const bool packed_output = publication_demand.format == CommonActivationPublicationFormat::F16K16Major;
    const bool f16_output    = publication_demand.format == CommonActivationPublicationFormat::F16Row;
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(strided_input ? kRmsNormBinaryStridedF32Kernel :
                                                  packed_output ? kRmsNormBinaryF32K16Kernel :
                                                  f16_output ? kRmsNormBinaryF32F16Kernel :
                                                               kRmsNormBinaryF32Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", rms_match.token_count);
    dispatch.kernel.compile_parameters.emplace("ggml.rmsnorm_binary_f32.hidden_size",
                                               to_config_value(rms_match.hidden_size));
    dispatch.kernel.compile_parameters.emplace("ggml.rmsnorm_binary_f32.rms_epsilon",
                                               to_config_value(rms_match.epsilon));
    dispatch.kernel.compile_parameters.emplace("ggml.rmsnorm_binary_f32.op",
                                               std::to_string(binary_kind_config_value(rms_match.op)));
    if (strided_input) {
        dispatch.kernel.compile_parameters.emplace("ggml.rmsnorm_binary_f32.input_stride",
                                                   to_config_value(rms_match.input_stride));
    }
    dispatch.bindings.push_back(
        strided_input ?
            DispatchBinding{ rms_match.input->storage_root, rms_match.input->storage_offset, rms_match.input_span } :
                                    DispatchBinding{ rms_match.input->id, 0, rms_match.input->byte_count });
    dispatch.bindings.push_back({ rms_match.rhs->id, 0, rms_match.rhs->byte_count });
    dispatch.bindings.push_back({ rms_match.output->id, 0, rms_match.output->byte_count });

    if (publication_demand.matched()) {
        const char * publication_name = packed_output ? "common.rmsnorm_binary.k16_major_f16" :
                                                        "common.rmsnorm_binary.f16";
        CommonActivationPublication publication;
        if (!common_reserve_activation_publication(context, match, *rms_match.output, publication_demand,
                                                   publication_name, publication) ||
            !common_append_activation_publication(match, publication, publication_name)) {
            return false;
        }
        dispatch.bindings.push_back(publication.binding());
    }

    match.covered_nodes.push_back(rms_match.rms_node_index);
    match.covered_nodes.push_back(rms_match.binary_node_index);
    match.dispatches.push_back(std::move(dispatch));
    return match.status.success();
}

static bool match_rmsnorm_mul_add_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const RmsNormMulAddMatch fused = match_rmsnorm_mul_add_f32(context);
    if (!fused.matched()) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kRmsNormMulAddF32Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", fused.rms_binary.token_count);
    dispatch.kernel.compile_parameters.emplace("ggml.rmsnorm_mul_add_f32.hidden_size",
                                               to_config_value(fused.rms_binary.hidden_size));
    dispatch.kernel.compile_parameters.emplace("ggml.rmsnorm_mul_add_f32.rms_epsilon",
                                               to_config_value(fused.rms_binary.epsilon));
    common_set_binary_compile_parameters(dispatch.kernel, "ggml.rmsnorm_mul_add_f32.normalized_op", nullptr,
                                         fused.rms_binary.op, true);
    common_set_binary_compile_parameters(dispatch.kernel, "ggml.rmsnorm_mul_add_f32.output_op",
                                         "ggml.rmsnorm_mul_add_f32.binary_lhs", fused.output_op, fused.binary_lhs);
    dispatch.bindings.push_back({ fused.rms_binary.input->id, 0, fused.rms_binary.input->byte_count });
    dispatch.bindings.push_back({ fused.rms_binary.rhs->id, 0, fused.rms_binary.rhs->byte_count });
    dispatch.bindings.push_back({ fused.residual->id, 0, fused.residual->byte_count });
    dispatch.bindings.push_back({ fused.output->id, 0, fused.output->byte_count });
    match.covered_nodes.push_back(fused.rms_binary.rms_node_index);
    match.covered_nodes.push_back(fused.rms_binary.binary_node_index);
    match.covered_nodes.push_back(fused.add_node_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_rmsnorm_binary_q8_1_x4_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const std::vector<GraphNode> & nodes = context.graph.nodes();
    if (context.root_index >= nodes.size()) {
        return false;
    }
    const RmsNormBinaryMatch rms_match =
        match_rmsnorm_binary_f32(context.graph, &nodes[context.root_index], context.root_index);
    const CommonActivationPublicationCandidate q8_candidate = {
        CommonActivationPublicationFormat::Q8_1X4,
        rmsnorm_q8_consumer_accepts,
        CommonActivationConsumerTraversal::FullRangeOrderedAliases,
        true,
    };
    const CommonActivationPublicationDemand q8_demand =
        rms_match.matched() ? common_select_activation_publication(context, *rms_match.output, &q8_candidate, 1) :
                              CommonActivationPublicationDemand{};
    if (!rms_match.matched() || rms_match.rms_node_index >= context.covered_nodes.size() ||
        rms_match.binary_node_index >= context.covered_nodes.size() ||
        context.covered_nodes[rms_match.rms_node_index] || context.covered_nodes[rms_match.binary_node_index] ||
        !q8_demand.matched()) {
        return false;
    }

    const size_t q8_byte_count = q8_1_x4_byte_count(rms_match.token_count, rms_match.hidden_size);
    if (q8_byte_count == 0) {
        return false;
    }

    const ValueId q8_value = context.next_plan_value;
    const ValueId f16_value(context.next_plan_value.value + 1);
    const size_t  f16_byte_count = rms_match.output->byte_count / 2;

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kRmsNormBinaryQ8_1X4F16Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", rms_match.token_count);
    dispatch.kernel.compile_parameters.emplace("ggml.rmsnorm_binary_q8_1_x4.hidden_size",
                                               to_config_value(rms_match.hidden_size));
    dispatch.kernel.compile_parameters.emplace("ggml.rmsnorm_binary_q8_1_x4.rms_epsilon",
                                               to_config_value(rms_match.epsilon));
    dispatch.kernel.compile_parameters.emplace("ggml.rmsnorm_binary_q8_1_x4.op",
                                               std::to_string(binary_kind_config_value(rms_match.op)));
    dispatch.bindings.push_back({ rms_match.input->id, 0, rms_match.input->byte_count });
    dispatch.bindings.push_back({ rms_match.rhs->id, 0, rms_match.rhs->byte_count });
    dispatch.bindings.push_back({ rms_match.output->id, 0, rms_match.output->byte_count });
    dispatch.bindings.push_back({ q8_value, 0, q8_byte_count });
    dispatch.bindings.push_back({ f16_value, 0, f16_byte_count });

    Status metadata_status;
    if (!match.metadata.append_alternate_value(
            { rms_match.output->id, q8_value, GGML_TYPE_Q8_1, q8_byte_count, "common.rmsnorm_binary.q8_1_x4" },
            metadata_status) ||
        !match.metadata.append_alternate_value(
            { rms_match.output->id, f16_value, GGML_TYPE_F16, f16_byte_count, "common.rmsnorm_binary.f16" },
            metadata_status)) {
        match.status.append(metadata_status);
        return false;
    }

    match.covered_nodes.push_back(rms_match.rms_node_index);
    match.covered_nodes.push_back(rms_match.binary_node_index);
    match.dispatches.push_back(std::move(dispatch));
    match.transients.push_back({ q8_value, "common.rmsnorm_binary.q8_1_x4", q8_byte_count, 256 });
    match.transients.push_back({ f16_value, "common.rmsnorm_binary.f16", f16_byte_count, 256 });
    return match.status.success();
}

static bool match_rmsnorm_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const std::vector<GraphNode> & nodes = context.graph.nodes();
    if (context.root_index >= nodes.size()) {
        return false;
    }
    const RmsNormMatch rms_match = match_rmsnorm_f32(context.graph, &nodes[context.root_index], context.root_index);
    if (!rms_match.matched() || rms_match.rms_node_index >= context.covered_nodes.size() ||
        context.covered_nodes[rms_match.rms_node_index]) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kRmsNormF32Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", rms_match.token_count);
    dispatch.kernel.compile_parameters.emplace("ggml.rmsnorm_f32.hidden_size", to_config_value(rms_match.hidden_size));
    dispatch.kernel.compile_parameters.emplace("ggml.rmsnorm_f32.rms_epsilon", to_config_value(rms_match.epsilon));
    dispatch.kernel.compile_parameters.emplace("ggml.rmsnorm_f32.input_stride",
                                               to_config_value(rms_match.input_stride));
    dispatch.bindings.push_back(
        { rms_match.input->storage_root, rms_match.input->storage_offset, rms_match.input_span });
    dispatch.bindings.push_back({ rms_match.output->id, 0, rms_match.output->byte_count });

    match.covered_nodes.push_back(rms_match.rms_node_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

bool common_match_rmsnorm_gate_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    return match_rmsnorm_gate_dispatch(context, match);
}

void register_rmsnorm_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "common.add_rmsnorm_binary_symmetric_i4_k32",
        GGML_OP_ADD,
        DispatchMatchKind::Fused,
        400,
        DispatchSource::Common,
        match_add_rmsnorm_binary_symmetric_i4_dispatch,
    });
    registry.add({
        "common.rmsnorm_gate",
        GGML_OP_RMS_NORM,
        DispatchMatchKind::Fused,
        400,
        DispatchSource::Common,
        match_rmsnorm_gate_dispatch,
    });
    registry.add({
        "common.rmsnorm_binary_symmetric_i4_k32",
        GGML_OP_RMS_NORM,
        DispatchMatchKind::Fused,
        350,
        DispatchSource::Common,
        match_rmsnorm_binary_symmetric_i4_dispatch,
    });
    registry.add({
        "common.rmsnorm_mul_rope_f32",
        GGML_OP_RMS_NORM,
        DispatchMatchKind::Fused,
        300,
        DispatchSource::Common,
        match_rmsnorm_mul_rope_f32_dispatch,
    });
    registry.add({
        "common.rmsnorm_mul_add_f32",
        GGML_OP_RMS_NORM,
        DispatchMatchKind::Fused,
        250,
        DispatchSource::Common,
        match_rmsnorm_mul_add_f32_dispatch,
    });
    registry.add({
        "common.rmsnorm_binary_q8_1_x4",
        GGML_OP_RMS_NORM,
        DispatchMatchKind::Fused,
        150,
        DispatchSource::Common,
        match_rmsnorm_binary_q8_1_x4_dispatch,
    });
    registry.add({
        "common.rmsnorm_binary_f32",
        GGML_OP_RMS_NORM,
        DispatchMatchKind::Fused,
        100,
        DispatchSource::Common,
        match_rmsnorm_binary_f32_dispatch,
    });
    registry.add({
        "common.rmsnorm_f32",
        GGML_OP_RMS_NORM,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_rmsnorm_f32_dispatch,
    });
}

}  // namespace ggml::hrx
