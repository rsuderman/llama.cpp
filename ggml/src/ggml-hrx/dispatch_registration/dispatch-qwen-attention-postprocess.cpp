#include "dispatch-qwen-attention-postprocess.h"

#include "ggml.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kQwenAttentionPostprocessF32F16Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_attention_postprocess_f32_f16");
static constexpr int64_t kQwenAttentionHeadSize = 128;

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool is_shape(const Value & value, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    return value.ne[0] == ne0 && value.ne[1] == ne1 && value.ne[2] == ne2 && value.ne[3] == ne3;
}

static bool is_2d(const Value & value) {
    return value.ne[0] > 0 && value.ne[1] > 0 && value.ne[2] == 1 && value.ne[3] == 1;
}

static bool is_supported_token_count(int64_t token_count) {
    return token_count >= 1 && token_count <= 2048;
}

static bool is_supported_head_count(int64_t head_count) {
    return head_count >= 1 && head_count <= 64;
}

static bool is_qwen_rms_norm_epsilon(float eps) {
    return eps >= 0.0000009f && eps <= 0.0000011f;
}

static bool is_supported_cache_index_type(ggml_type type) {
    return type == GGML_TYPE_I64;
}

static const GraphNode * find_single_consumer_with_op(const Graph & graph, ValueId value, ggml_op op) {
    const GraphNode * match = nullptr;
    for (const GraphNode * consumer : graph.index().consumers(value)) {
        if (consumer == nullptr || consumer->op != op) {
            continue;
        }
        if (match != nullptr) {
            return nullptr;
        }
        match = consumer;
    }
    return match;
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

struct NormRopeChain {
    const GraphNode * projection_node = nullptr;
    const GraphNode * reshape_node    = nullptr;
    const GraphNode * rms_node        = nullptr;
    const GraphNode * mul_node        = nullptr;
    const GraphNode * rope_node       = nullptr;
    const Value *     raw_input       = nullptr;
    const Value *     reshaped        = nullptr;
    const Value *     norm_weight     = nullptr;
    const Value *     positions       = nullptr;
    const Value *     inverse_freqs   = nullptr;
    const Value *     output          = nullptr;
    int64_t           token_count     = 0;
    int64_t           head_count      = 0;

    bool matched() const {
        return projection_node != nullptr && reshape_node != nullptr && rms_node != nullptr && mul_node != nullptr &&
               rope_node != nullptr && raw_input != nullptr && reshaped != nullptr && norm_weight != nullptr &&
               positions != nullptr && inverse_freqs != nullptr && output != nullptr && token_count > 0 &&
               head_count > 0;
    }
};

struct CachePublishChain {
    NormRopeChain     key;
    const GraphNode * layout_node      = nullptr;
    const GraphNode * set_rows_node    = nullptr;
    const Value *     cache_indices    = nullptr;
    const Value *     cache            = nullptr;
    int64_t           cache_row_count  = 0;
    bool              key_publish_path = false;

    bool matched_key() const {
        return key_publish_path && key.matched() && layout_node != nullptr && set_rows_node != nullptr &&
               cache_indices != nullptr && cache != nullptr && cache_row_count > 0;
    }
};

struct ValuePublishChain {
    const GraphNode * projection_node = nullptr;
    const GraphNode * reshape_node    = nullptr;
    const GraphNode * layout_node     = nullptr;
    const GraphNode * set_rows_node   = nullptr;
    const Value *     raw_input       = nullptr;
    const Value *     cache_indices   = nullptr;
    const Value *     cache           = nullptr;
    int64_t           token_count     = 0;
    int64_t           head_count      = 0;
    int64_t           cache_row_count = 0;

    bool matched() const {
        return projection_node != nullptr && reshape_node != nullptr && layout_node != nullptr &&
               set_rows_node != nullptr && raw_input != nullptr && cache_indices != nullptr && cache != nullptr &&
               token_count > 0 && head_count > 0 && cache_row_count > 0;
    }
};

struct AttentionPostprocessMatch {
    NormRopeChain     query;
    CachePublishChain key;
    ValuePublishChain value;

    bool matched() const { return query.matched() && key.matched_key() && value.matched(); }
};

static bool has_qwen_rope_params(const GraphNode & node) {
    const RopeParams * params = op_params_as<RopeParams>(node.params);
    return params != nullptr && params->n_dims == kQwenAttentionHeadSize && params->mode == GGML_ROPE_TYPE_NEOX;
}

static bool has_qwen_rms_params(const GraphNode & node) {
    const RmsNormParams * params = op_params_as<RmsNormParams>(node.params);
    return params != nullptr && is_qwen_rms_norm_epsilon(params->eps);
}

static bool is_norm_weight(const Value & value) {
    return value.type == GGML_TYPE_F32 && is_shape(value, kQwenAttentionHeadSize, 1, 1, 1);
}

static bool is_inverse_frequency_table(const Value & value) {
    return value.type == GGML_TYPE_F32 && is_shape(value, kQwenAttentionHeadSize / 2, 1, 1, 1);
}

static bool match_projection_reshape(const Graph & graph, const GraphNode * reshape, NormRopeChain & chain) {
    if (reshape == nullptr || reshape->op != GGML_OP_RESHAPE || reshape->inputs.size() != 1) {
        return false;
    }

    const GraphNode * projection = producer_with_op(graph, reshape->inputs[0], GGML_OP_MUL_MAT);
    const Value *     raw_input  = graph_value(graph, reshape->inputs[0]);
    const Value *     reshaped   = graph_value(graph, reshape->output);
    if (projection == nullptr || raw_input == nullptr || reshaped == nullptr) {
        return false;
    }
    if (raw_input->type != GGML_TYPE_F32 || reshaped->type != GGML_TYPE_F32 || !is_2d(*raw_input) ||
        reshaped->ne[0] != kQwenAttentionHeadSize || reshaped->ne[3] != 1) {
        return false;
    }

    const int64_t head_count  = reshaped->ne[1];
    const int64_t token_count = reshaped->ne[2];
    if (!is_supported_head_count(head_count) || !is_supported_token_count(token_count) ||
        raw_input->ne[0] != head_count * kQwenAttentionHeadSize || raw_input->ne[1] != token_count) {
        return false;
    }

    chain.projection_node = projection;
    chain.reshape_node    = reshape;
    chain.raw_input       = raw_input;
    chain.reshaped        = reshaped;
    chain.token_count     = token_count;
    chain.head_count      = head_count;
    return true;
}

static bool match_norm_rope_chain_from_reshape(const Graph & graph, const GraphNode * reshape, NormRopeChain & chain) {
    if (!match_projection_reshape(graph, reshape, chain)) {
        return false;
    }

    const GraphNode * rms = find_single_consumer_with_op(graph, chain.reshaped->id, GGML_OP_RMS_NORM);
    if (rms == nullptr || rms->inputs.size() != 1 || !has_qwen_rms_params(*rms)) {
        return false;
    }

    const GraphNode * mul = find_single_consumer_with_op(graph, rms->output, GGML_OP_MUL);
    if (mul == nullptr || mul->inputs.size() != 2) {
        return false;
    }
    ValueId weight_id;
    if (mul->inputs[0] == rms->output) {
        weight_id = mul->inputs[1];
    } else if (mul->inputs[1] == rms->output) {
        weight_id = mul->inputs[0];
    } else {
        return false;
    }
    const Value * norm_weight = graph_value(graph, weight_id);
    if (norm_weight == nullptr || !is_norm_weight(*norm_weight)) {
        return false;
    }

    const GraphNode * rope = find_single_consumer_with_op(graph, mul->output, GGML_OP_ROPE);
    if (rope == nullptr || rope->inputs.size() < 3 || rope->inputs[0] != mul->output || !has_qwen_rope_params(*rope)) {
        return false;
    }
    const Value * positions     = graph_value(graph, rope->inputs[1]);
    const Value * inverse_freqs = graph_value(graph, rope->inputs[2]);
    const Value * output        = graph_value(graph, rope->output);
    if (positions == nullptr || inverse_freqs == nullptr || output == nullptr || positions->type != GGML_TYPE_I32 ||
        !is_shape(*positions, chain.token_count, 1, 1, 1) || !is_inverse_frequency_table(*inverse_freqs) ||
        output->type != GGML_TYPE_F32 ||
        !is_shape(*output, kQwenAttentionHeadSize, chain.head_count, chain.token_count, 1)) {
        return false;
    }

    chain.rms_node      = rms;
    chain.mul_node      = mul;
    chain.rope_node     = rope;
    chain.norm_weight   = norm_weight;
    chain.positions     = positions;
    chain.inverse_freqs = inverse_freqs;
    chain.output        = output;
    return true;
}

static bool layout_op(ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE;
}

static int64_t cache_row_count_for_value(const Value & cache, int64_t head_count) {
    if (cache.type != GGML_TYPE_F16 || head_count <= 0) {
        return 0;
    }
    if (cache.ne[0] == kQwenAttentionHeadSize * head_count && cache.ne[2] == 1 && cache.ne[3] == 1) {
        return cache.ne[1];
    }
    if (cache.ne[0] == kQwenAttentionHeadSize && cache.ne[2] == head_count && cache.ne[3] == 1) {
        return cache.ne[1];
    }
    return 0;
}

static bool match_key_publish_chain(const Graph & graph, const GraphNode * set_rows, CachePublishChain & chain) {
    if (set_rows == nullptr || set_rows->op != GGML_OP_SET_ROWS || set_rows->inputs.size() != 3) {
        return false;
    }
    const GraphNode * layout = graph.index().producer(set_rows->inputs[0]);
    if (layout == nullptr || !layout_op(layout->op) || layout->inputs.size() != 1) {
        return false;
    }

    const GraphNode * rope = producer_with_op(graph, layout->inputs[0], GGML_OP_ROPE);
    if (rope == nullptr) {
        return false;
    }
    const GraphNode * mul = producer_with_op(graph, rope->inputs.empty() ? ValueId() : rope->inputs[0], GGML_OP_MUL);
    if (mul == nullptr || mul->inputs.size() != 2) {
        return false;
    }
    const GraphNode * rms = nullptr;
    if (mul->inputs[0] != rope->inputs[0]) {
        rms = producer_with_op(graph, mul->inputs[0], GGML_OP_RMS_NORM);
    }
    if (rms == nullptr && mul->inputs[1] != rope->inputs[0]) {
        rms = producer_with_op(graph, mul->inputs[1], GGML_OP_RMS_NORM);
    }
    if (rms == nullptr || rms->inputs.size() != 1) {
        return false;
    }
    const GraphNode * reshape = producer_with_op(graph, rms->inputs[0], GGML_OP_RESHAPE);
    NormRopeChain     key_chain;
    if (!match_norm_rope_chain_from_reshape(graph, reshape, key_chain) || key_chain.rope_node != rope) {
        return false;
    }

    const Value * cache_indices = graph_value(graph, set_rows->inputs[1]);
    const Value * cache         = graph_value(graph, set_rows->inputs[2]);
    if (cache_indices == nullptr || cache == nullptr || !is_supported_cache_index_type(cache_indices->type) ||
        !is_shape(*cache_indices, key_chain.token_count, 1, 1, 1)) {
        return false;
    }
    const int64_t cache_row_count = cache_row_count_for_value(*cache, key_chain.head_count);
    if (cache_row_count <= 0) {
        return false;
    }

    chain.key              = key_chain;
    chain.layout_node      = layout;
    chain.set_rows_node    = set_rows;
    chain.cache_indices    = cache_indices;
    chain.cache            = cache;
    chain.cache_row_count  = cache_row_count;
    chain.key_publish_path = true;
    return true;
}

static bool match_value_publish_chain(const Graph & graph, const GraphNode * set_rows, ValuePublishChain & chain) {
    if (set_rows == nullptr || set_rows->op != GGML_OP_SET_ROWS || set_rows->inputs.size() != 3) {
        return false;
    }
    const GraphNode * layout = graph.index().producer(set_rows->inputs[0]);
    if (layout == nullptr || !layout_op(layout->op) || layout->inputs.size() != 1) {
        return false;
    }

    const GraphNode * reshape = producer_with_op(graph, layout->inputs[0], GGML_OP_RESHAPE);
    if (reshape == nullptr) {
        reshape = layout;
    }
    if (reshape == nullptr || reshape->op != GGML_OP_RESHAPE || reshape->inputs.size() != 1) {
        return false;
    }

    NormRopeChain projection_shape;
    if (!match_projection_reshape(graph, reshape, projection_shape)) {
        return false;
    }

    const Value * cache_indices = graph_value(graph, set_rows->inputs[1]);
    const Value * cache         = graph_value(graph, set_rows->inputs[2]);
    if (cache_indices == nullptr || cache == nullptr || !is_supported_cache_index_type(cache_indices->type) ||
        !is_shape(*cache_indices, projection_shape.token_count, 1, 1, 1)) {
        return false;
    }
    const int64_t cache_row_count = cache_row_count_for_value(*cache, projection_shape.head_count);
    if (cache_row_count <= 0) {
        return false;
    }

    chain.projection_node = projection_shape.projection_node;
    chain.reshape_node    = projection_shape.reshape_node;
    chain.layout_node     = layout;
    chain.set_rows_node   = set_rows;
    chain.raw_input       = projection_shape.raw_input;
    chain.cache_indices   = cache_indices;
    chain.cache           = cache;
    chain.token_count     = projection_shape.token_count;
    chain.head_count      = projection_shape.head_count;
    chain.cache_row_count = cache_row_count;
    return true;
}

static AttentionPostprocessMatch match_qwen_attention_postprocess(const Graph & graph, const GraphNode * root) {
    AttentionPostprocessMatch match;
    if (root == nullptr || root->op != GGML_OP_RESHAPE || !graph.has_index()) {
        return match;
    }
    if (!match_norm_rope_chain_from_reshape(graph, root, match.query)) {
        return {};
    }

    for (const GraphNode & node : graph.nodes()) {
        if (node.op != GGML_OP_SET_ROWS) {
            continue;
        }
        CachePublishChain key;
        if (!match.key.matched_key() && match_key_publish_chain(graph, &node, key)) {
            match.key = key;
            continue;
        }
        ValuePublishChain value;
        if (!match.value.matched() && match_value_publish_chain(graph, &node, value)) {
            match.value = value;
        }
    }

    if (!match.matched()) {
        return {};
    }
    if (match.query.token_count != match.key.key.token_count || match.query.token_count != match.value.token_count ||
        match.key.key.head_count != match.value.head_count ||
        match.query.positions->id != match.key.key.positions->id ||
        match.query.inverse_freqs->id != match.key.key.inverse_freqs->id ||
        match.key.cache_row_count != match.value.cache_row_count) {
        return {};
    }
    return match;
}

static bool append_postprocess_covered_nodes(const DispatchMatchContext &      context,
                                             const AttentionPostprocessMatch & postprocess,
                                             DispatchMatch &                   dispatch_match) {
    // TODO: move fused matcher coverage into a shared builder that records GraphNode pointers during matching and
    // materializes scheduler indices once. This is constant-size today, but the explicit list will not scale well as
    // Qwen fused patterns grow.
    if (!append_covered_node(context, postprocess.query.reshape_node, dispatch_match) ||
        !append_covered_node(context, postprocess.query.rms_node, dispatch_match) ||
        !append_covered_node(context, postprocess.query.mul_node, dispatch_match) ||
        !append_covered_node(context, postprocess.query.rope_node, dispatch_match) ||
        !append_covered_node(context, postprocess.key.key.reshape_node, dispatch_match) ||
        !append_covered_node(context, postprocess.key.key.rms_node, dispatch_match) ||
        !append_covered_node(context, postprocess.key.key.mul_node, dispatch_match) ||
        !append_covered_node(context, postprocess.key.key.rope_node, dispatch_match) ||
        !append_covered_node(context, postprocess.key.layout_node, dispatch_match) ||
        !append_covered_node(context, postprocess.key.set_rows_node, dispatch_match) ||
        !append_covered_node(context, postprocess.value.reshape_node, dispatch_match) ||
        !append_covered_node(context, postprocess.value.layout_node, dispatch_match) ||
        !append_covered_node(context, postprocess.value.set_rows_node, dispatch_match)) {
        return false;
    }
    return true;
}

}  // namespace

static bool match_qwen_attention_postprocess_dispatch(const DispatchMatchContext & context,
                                                      DispatchMatch &              dispatch_match) {
    const AttentionPostprocessMatch match = match_qwen_attention_postprocess(context.graph, context.root_node);
    if (!match.matched()) {
        return false;
    }
    if (!append_postprocess_covered_nodes(context, match, dispatch_match)) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kQwenAttentionPostprocessF32F16Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.query.token_count);
    dispatch.kernel.integer_parameters.emplace("cache_row_count", match.key.cache_row_count);
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.model.rms_epsilon", "0.000001");
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.attention.head_size",
                                               to_config_value(kQwenAttentionHeadSize));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.attention.query_size",
                                               to_config_value(match.query.head_count * kQwenAttentionHeadSize));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.attention.key_value_size",
                                               to_config_value(match.key.key.head_count * kQwenAttentionHeadSize));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.workload.token_capacity",
                                               to_config_value(match.query.token_count));
    dispatch.bindings.push_back({ match.query.positions->id, 0, match.query.positions->byte_count });
    dispatch.bindings.push_back({ match.key.cache_indices->id, 0, match.key.cache_indices->byte_count });
    dispatch.bindings.push_back({ match.value.cache_indices->id, 0, match.value.cache_indices->byte_count });
    dispatch.bindings.push_back({ match.query.raw_input->id, 0, match.query.raw_input->byte_count });
    dispatch.bindings.push_back({ match.key.key.raw_input->id, 0, match.key.key.raw_input->byte_count });
    dispatch.bindings.push_back({ match.value.raw_input->id, 0, match.value.raw_input->byte_count });
    dispatch.bindings.push_back({ match.query.norm_weight->id, 0, match.query.norm_weight->byte_count });
    dispatch.bindings.push_back({ match.key.key.norm_weight->id, 0, match.key.key.norm_weight->byte_count });
    dispatch.bindings.push_back({ match.query.inverse_freqs->id, 0, match.query.inverse_freqs->byte_count });
    dispatch.bindings.push_back({ match.query.output->id, 0, match.query.output->byte_count });
    dispatch.bindings.push_back({ match.key.cache->id, 0, match.key.cache->byte_count });
    dispatch.bindings.push_back({ match.value.cache->id, 0, match.value.cache->byte_count });

    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

void register_qwen_attention_postprocess_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "qwen.attention_postprocess_f32_f16",
        GGML_OP_RESHAPE,
        DispatchMatchKind::Fused,
        100,
        DispatchSource::Qwen,
        match_qwen_attention_postprocess_dispatch,
    });
}

}  // namespace ggml::hrx
