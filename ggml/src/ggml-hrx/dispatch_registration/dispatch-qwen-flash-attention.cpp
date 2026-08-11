#include "dispatch-qwen-flash-attention.h"

#include "ggml.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kQwenFlashAttentionF32F16WmmaKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_flash_attention_f32_f16_wmma");
static constexpr int64_t kQwenAttentionHeadSize = 128;

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool nearly_equal(float lhs, float rhs) {
    return std::fabs(lhs - rhs) <= 1.0e-6f;
}

static bool is_supported_token_count(int64_t token_count) {
    return token_count >= 1 && token_count <= 2048;
}

static bool is_supported_key_value_token_count(int64_t token_count) {
    return token_count >= 1 && token_count <= 32768;
}

static bool is_supported_head_count(int64_t head_count) {
    return head_count >= 1 && head_count <= 64;
}

static bool has_query_layout(const Value & value, int64_t query_head_count) {
    const size_t element_size = sizeof(float);
    return value.nb[0] == element_size &&
           value.nb[1] == static_cast<size_t>(query_head_count * kQwenAttentionHeadSize) * element_size &&
           (value.ne[2] == 1 || value.nb[2] == static_cast<size_t>(kQwenAttentionHeadSize) * element_size);
}

static bool has_key_value_layout(const Value & value, int64_t key_value_head_count) {
    const size_t element_size = sizeof(ggml_fp16_t);
    return value.nb[0] == element_size &&
           value.nb[1] == static_cast<size_t>(key_value_head_count * kQwenAttentionHeadSize) * element_size &&
           (value.ne[2] == 1 || value.nb[2] == static_cast<size_t>(kQwenAttentionHeadSize) * element_size);
}

static bool has_mask_layout(const Value & value, int64_t key_value_token_count) {
    const size_t element_size = sizeof(ggml_fp16_t);
    return value.nb[0] == element_size && value.nb[1] == static_cast<size_t>(key_value_token_count) * element_size;
}

static bool has_output_layout(const Value & value, int64_t query_head_count) {
    const size_t element_size = sizeof(float);
    return value.nb[0] == element_size && value.nb[1] == static_cast<size_t>(kQwenAttentionHeadSize) * element_size &&
           value.nb[2] == static_cast<size_t>(query_head_count * kQwenAttentionHeadSize) * element_size;
}

static std::string to_config_value(int64_t value) {
    return std::to_string(value);
}

struct QwenFlashAttentionMatch {
    const Value *     query                 = nullptr;
    const Value *     key                   = nullptr;
    const Value *     value                 = nullptr;
    const Value *     mask                  = nullptr;
    const Value *     output                = nullptr;
    const GraphNode * output_layout         = nullptr;
    int64_t           query_token_count     = 0;
    int64_t           key_value_token_count = 0;
    int64_t           query_head_count      = 0;
    int64_t           key_value_head_count  = 0;

    bool matched() const {
        return query != nullptr && key != nullptr && value != nullptr && mask != nullptr && output != nullptr;
    }
};

static bool layout_op(ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE;
}

static const GraphNode * find_single_layout_consumer(const Graph & graph, ValueId value) {
    const GraphNode * match = nullptr;
    for (const GraphNode * consumer : graph.index().consumers(value)) {
        if (consumer == nullptr || !layout_op(consumer->op)) {
            continue;
        }
        if (match != nullptr) {
            return nullptr;
        }
        match = consumer;
    }
    return match;
}

static bool has_qwen_flash_attention_params(const GraphNode & node) {
    const FlashAttnExtParams * params = op_params_as<FlashAttnExtParams>(node.params);
    if (params == nullptr) {
        return false;
    }
    const float expected_scale = 1.0f / std::sqrt(static_cast<float>(kQwenAttentionHeadSize));
    return nearly_equal(params->scale, expected_scale) && nearly_equal(params->max_bias, 0.0f) &&
           nearly_equal(params->logit_softcap, 0.0f) &&
           (params->prec == GGML_PREC_DEFAULT || params->prec == GGML_PREC_F32);
}

static QwenFlashAttentionMatch match_qwen_flash_attention(const Graph & graph, const GraphNode * node) {
    QwenFlashAttentionMatch match;
    if (node == nullptr || node->op != GGML_OP_FLASH_ATTN_EXT || node->inputs.size() != 4 ||
        !has_qwen_flash_attention_params(*node)) {
        return match;
    }

    const Value * query  = graph_value(graph, node->inputs[0]);
    const Value * key    = graph_value(graph, node->inputs[1]);
    const Value * value  = graph_value(graph, node->inputs[2]);
    const Value * mask   = graph_value(graph, node->inputs[3]);
    const Value * output = graph_value(graph, node->output);
    if (query == nullptr || key == nullptr || value == nullptr || mask == nullptr || output == nullptr) {
        return {};
    }
    if (query->type != GGML_TYPE_F32 || key->type != GGML_TYPE_F16 || value->type != GGML_TYPE_F16 ||
        mask->type != GGML_TYPE_F16 || output->type != GGML_TYPE_F32) {
        return {};
    }
    if (query->ne[0] != kQwenAttentionHeadSize || key->ne[0] != kQwenAttentionHeadSize ||
        value->ne[0] != kQwenAttentionHeadSize || output->ne[0] != kQwenAttentionHeadSize) {
        return {};
    }
    if (query->ne[3] != 1 || key->ne[3] != 1 || value->ne[3] != 1 || output->ne[3] != 1 || mask->ne[2] != 1 ||
        mask->ne[3] != 1) {
        return {};
    }

    const int64_t query_token_count     = query->ne[1];
    const int64_t query_head_count      = query->ne[2];
    const int64_t key_value_token_count = key->ne[1];
    const int64_t key_value_head_count  = key->ne[2];
    if (!is_supported_token_count(query_token_count) || !is_supported_key_value_token_count(key_value_token_count) ||
        !is_supported_head_count(query_head_count) || !is_supported_head_count(key_value_head_count) ||
        query_head_count % key_value_head_count != 0) {
        return {};
    }
    if (value->ne[1] != key_value_token_count || value->ne[2] != key_value_head_count ||
        mask->ne[0] != key_value_token_count || mask->ne[1] != query_token_count || output->ne[1] != query_head_count ||
        output->ne[2] != query_token_count) {
        return {};
    }
    if (!has_query_layout(*query, query_head_count) || !has_key_value_layout(*key, key_value_head_count) ||
        !has_key_value_layout(*value, key_value_head_count) || !has_mask_layout(*mask, key_value_token_count) ||
        !has_output_layout(*output, query_head_count)) {
        return {};
    }

    match.query                 = query;
    match.key                   = key;
    match.value                 = value;
    match.mask                  = mask;
    match.output                = output;
    match.output_layout         = find_single_layout_consumer(graph, output->id);
    match.query_token_count     = query_token_count;
    match.key_value_token_count = key_value_token_count;
    match.query_head_count      = query_head_count;
    match.key_value_head_count  = key_value_head_count;
    return match;
}

}  // namespace

static bool match_qwen_flash_attention_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const QwenFlashAttentionMatch match = match_qwen_flash_attention(context.graph, context.root_node);
    if (!match.matched()) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kQwenFlashAttentionF32F16WmmaKernel);
    dispatch.kernel.integer_parameters.emplace("query_token_count", match.query_token_count);
    dispatch.kernel.integer_parameters.emplace("key_value_token_count", match.key_value_token_count);
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.attention.query_head_count",
                                               to_config_value(match.query_head_count));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.attention.key_value_head_count",
                                               to_config_value(match.key_value_head_count));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.workload.token_capacity",
                                               to_config_value(match.query_token_count));
    dispatch.bindings.push_back({ match.query->id, 0, match.query->byte_count });
    dispatch.bindings.push_back({ match.key->id, 0, match.key->byte_count });
    dispatch.bindings.push_back({ match.value->id, 0, match.value->byte_count });
    dispatch.bindings.push_back({ match.mask->id, 0, match.mask->byte_count });
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });

    dispatch_match.covered_nodes.push_back(context.root_index);
    if (match.output_layout != nullptr) {
        size_t layout_index = 0;
        if (!context.graph.index().node_index(match.output_layout, layout_index) ||
            layout_index >= context.covered_nodes.size() || context.covered_nodes[layout_index]) {
            return false;
        }
        dispatch_match.covered_nodes.push_back(layout_index);
    }
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

void register_qwen_flash_attention_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "qwen.flash_attention_f32_f16_wmma",
        GGML_OP_FLASH_ATTN_EXT,
        DispatchMatchKind::SingleOp,
        100,
        DispatchSource::Qwen,
        match_qwen_flash_attention_dispatch,
    });
}

}  // namespace ggml::hrx
