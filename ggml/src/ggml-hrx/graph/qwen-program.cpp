#include "qwen-program.h"

#include "kernel-corpus-catalog-verify.h"

#include <algorithm>
#include <array>
#include <set>
#include <sstream>

namespace ggml::hrx {
namespace {

static constexpr const char * kOracleRevision = "hrx-system:b01fe3eb2cdd/qwen-program-full-logits-v1";
static constexpr size_t kLayerCount = 48;
static constexpr size_t kRegularLayerOperationCount = 63;
static constexpr OperationId kFirstLayerOperation = 1;
static constexpr OperationId kTerminalLayerOperation = 2962;
static constexpr OperationId kEndpointOperation = 3027;
static constexpr int64_t kAttentionKvTileSize = 64;
static constexpr int64_t kAttentionKvMaximum = 32768;
// The current grouped routed-gate/up prefill specialization has the narrowest
// token-domain contract in the owned kernel schedule.
static constexpr size_t kPrefillTokenMaximum = 512;

static constexpr KernelCatalogRef kNoKernelRef = {};

#define K(name_literal) GGML_HRX_KERNEL_REF("qwen3_moe", name_literal)

static const std::vector<enum ggml_op> & regular_layer_signature() {
    static const std::vector<enum ggml_op> value = {
        GGML_OP_RMS_NORM, GGML_OP_MUL,
        GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE,
        GGML_OP_MUL_MAT, GGML_OP_RESHAPE,
        GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE,
        GGML_OP_VIEW, GGML_OP_SET_ROWS, GGML_OP_VIEW, GGML_OP_SET_ROWS,
        GGML_OP_VIEW, GGML_OP_PERMUTE, GGML_OP_VIEW, GGML_OP_PERMUTE,
        GGML_OP_VIEW, GGML_OP_PERMUTE, GGML_OP_FLASH_ATTN_EXT, GGML_OP_RESHAPE,
        GGML_OP_MUL_MAT, GGML_OP_ADD, GGML_OP_RMS_NORM, GGML_OP_MUL,
        GGML_OP_MUL_MAT, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT,
        GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE, GGML_OP_SUM_ROWS,
        GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_RESHAPE,
        GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU, GGML_OP_MUL_MAT_ID, GGML_OP_MUL,
        GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
        GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
        GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD,
        GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD,
    };
    return value;
}

static std::vector<enum ggml_op> terminal_layer_signature() {
    const std::vector<enum ggml_op> & regular = regular_layer_signature();
    std::vector<enum ggml_op> value(regular.begin(), regular.begin() + 27);
    value.push_back(GGML_OP_GET_ROWS);
    value.push_back(GGML_OP_GET_ROWS);
    value.insert(value.end(), regular.begin() + 27, regular.end());
    return value;
}

static bool check_signature(const Graph & graph, OperationId first, const std::vector<enum ggml_op> & signature,
                            const std::string & name, std::vector<std::string> & errors) {
    if (first + signature.size() > graph.operations.size()) {
        errors.push_back(name + " extends beyond the graph");
        return false;
    }
    for (size_t i = 0; i < signature.size(); ++i) {
        if (graph.operations[first + i].op != signature[i]) {
            errors.push_back(name + " operation " + std::to_string(i) + " is " +
                ggml_op_name(graph.operations[first + i].op) + ", expected " + ggml_op_name(signature[i]));
            return false;
        }
    }
    return true;
}

static enum ggml_type weight_type(const Graph & graph, OperationId operation) {
    if (operation >= graph.operations.size() || graph.operations[operation].inputs.empty()) return GGML_TYPE_COUNT;
    const ValueId value = graph.operations[operation].inputs[0];
    return value < graph.values.size() ? graph.values[value].type : GGML_TYPE_COUNT;
}

static OperationId layer_start(size_t layer);

static bool has_shape(const Graph & graph, OperationId operation, std::initializer_list<int64_t> dimensions) {
    if (operation >= graph.operations.size()) return false;
    const Value & value = graph.values[graph.operations[operation].output];
    size_t i = 0;
    for (int64_t dimension : dimensions) {
        if (i >= value.access.shape.size() || value.access.shape[i++] != dimension) return false;
    }
    return true;
}

struct AttentionGeometry {
    int64_t head_size = 0;
    int64_t key_value_token_count = 0;
    int64_t key_value_head_count = 0;
    int64_t query_head_count = 0;
};

// Recovered model-family geometry. Qwen-specific operation ordinals identify
// the values, but none of these dimensions are selector constants.
struct RoutedTransformerGeometry {
    int64_t hidden_size = 0;
    int64_t query_size = 0;
    int64_t key_value_size = 0;
    int64_t expert_count = 0;
    int64_t route_count = 0;
    int64_t expert_intermediate_size = 0;
};

static bool recover_attention_geometry(const Graph & graph, size_t layer, AttentionGeometry & geometry,
                                       std::vector<std::string> & errors) {
    const OperationId flash = layer_start(layer) + 24;
    if (flash >= graph.operations.size() || graph.operations[flash].op != GGML_OP_FLASH_ATTN_EXT ||
        graph.operations[flash].inputs.size() != 4) {
        errors.push_back("layer " + std::to_string(layer) + " has no canonical flash-attention input tuple");
        return false;
    }
    const Operation & operation = graph.operations[flash];
    for (ValueId input : operation.inputs) {
        if (input >= graph.values.size()) {
            errors.push_back("layer " + std::to_string(layer) + " flash-attention input is invalid");
            return false;
        }
    }
    const Value & query = graph.values[operation.inputs[0]];
    const Value & key = graph.values[operation.inputs[1]];
    const Value & value = graph.values[operation.inputs[2]];
    const Value & mask = graph.values[operation.inputs[3]];
    geometry.head_size = key.access.shape[0];
    geometry.key_value_token_count = key.access.shape[1];
    geometry.key_value_head_count = key.access.shape[2];
    geometry.query_head_count = query.access.shape[2];
    if (geometry.head_size <= 0 || geometry.key_value_head_count <= 0 || geometry.query_head_count <= 0 ||
        query.access.shape[0] != geometry.head_size ||
        value.access.shape[0] != geometry.head_size ||
        value.access.shape[1] != geometry.key_value_token_count ||
        value.access.shape[2] != geometry.key_value_head_count ||
        mask.access.shape[0] != geometry.key_value_token_count) {
        errors.push_back("layer " + std::to_string(layer) + " K/V/mask attention geometry does not agree");
        return false;
    }
    if (geometry.key_value_token_count <= 0 || geometry.key_value_token_count > kAttentionKvMaximum) {
        errors.push_back("layer " + std::to_string(layer) + " KV extent is outside the kernel contract");
        return false;
    }
    if (geometry.key_value_token_count % kAttentionKvTileSize != 0) {
        errors.push_back("layer " + std::to_string(layer) + " HRX padded KV extent is not a 64-token tile multiple");
        return false;
    }
    return true;
}

static bool validate_layer_facts(const Graph & graph, size_t layer, size_t token_count,
                                 const RoutedTransformerGeometry & model,
                                 std::vector<std::string> & errors) {
    const OperationId start = layer_start(layer);
    const bool terminal = layer == 47;
    const int64_t active_rows = terminal ? 1 : static_cast<int64_t>(token_count);
    AttentionGeometry attention_geometry;
    if (!recover_attention_geometry(graph, layer, attention_geometry, errors)) return false;
    if (!has_shape(graph, start, { model.hidden_size, static_cast<int64_t>(token_count) }) ||
        !has_shape(graph, start + 2, { model.query_size, static_cast<int64_t>(token_count) }) ||
        !has_shape(graph, start + 7, { model.key_value_size, static_cast<int64_t>(token_count) }) ||
        !has_shape(graph, start + 9, { model.key_value_size, static_cast<int64_t>(token_count) }) ||
        model.query_size != attention_geometry.head_size * attention_geometry.query_head_count ||
        model.key_value_size != attention_geometry.head_size * attention_geometry.key_value_head_count) {
        errors.push_back("layer " + std::to_string(layer) + " attention dimensions disagree with recovered model geometry");
        return false;
    }
    const OperationId router = start + (terminal ? 32 : 30);
    const OperationId gate = start + (terminal ? 44 : 42);
    const OperationId up = gate + 1;
    const OperationId down = start + (terminal ? 47 : 45);
    if (!has_shape(graph, router, { model.expert_count, active_rows }) ||
        !has_shape(graph, gate, { model.expert_intermediate_size, model.route_count, active_rows }) ||
        !has_shape(graph, up, { model.expert_intermediate_size, model.route_count, active_rows }) ||
        !has_shape(graph, down, { model.hidden_size, model.route_count, active_rows }) ||
        weight_type(graph, router) != GGML_TYPE_F32 || weight_type(graph, gate) != GGML_TYPE_Q4_K ||
        weight_type(graph, up) != GGML_TYPE_Q4_K) {
        errors.push_back("layer " + std::to_string(layer) + " router or expert dimensions/storage do not match");
        return false;
    }
    for (OperationId writer : { start + 15, start + 17 }) {
        const bool has_write = std::any_of(graph.operations[writer].effects.begin(), graph.operations[writer].effects.end(),
            [](const Effect & effect) { return effect.kind == EffectKind::Write && effect.after_version == effect.before_version + 1; });
        if (!has_write) {
            errors.push_back("layer " + std::to_string(layer) + " does not publish both K/V storage versions");
            return false;
        }
    }
    if (layer > 0) {
        const OperationId previous_start = layer_start(layer - 1);
        const OperationId previous_residual = previous_start + 62;
        if (graph.operations[start].inputs.empty() ||
            graph.operations[start].inputs[0] != graph.operations[previous_residual].output) {
            errors.push_back("layer " + std::to_string(layer) + " breaks the hidden-state recurrence");
            return false;
        }
    }
    return true;
}

static RoutedTransformerGeometry recover_model_geometry(const Graph & graph) {
    const OperationId start = layer_start(0);
    RoutedTransformerGeometry model;
    model.hidden_size = graph.values[graph.operations[start].output].access.shape[0];
    model.query_size = graph.values[graph.operations[start + 2].output].access.shape[0];
    model.key_value_size = graph.values[graph.operations[start + 7].output].access.shape[0];
    model.expert_count = graph.values[graph.operations[start + 30].output].access.shape[0];
    model.route_count = graph.values[graph.operations[start + 34].output].access.shape[0];
    model.expert_intermediate_size = graph.values[graph.operations[start + 42].output].access.shape[0];
    return model;
}

static std::string type_suffix(enum ggml_type type) {
    return type == GGML_TYPE_COUNT ? "invalid" : ggml_type_name(type);
}

static KernelSpecialization invocation_kernel(const std::string & variant, int layer, size_t token_count) {
    KernelSpecialization result;
    result.family = "qwen3_moe";
    result.variant = variant;
    result.kernel_id = kUncatalogedKernelId;
    result.execution_kind = KernelSpecialization::ExecutionKind::Native;
    result.integer_parameters["token_count"] = token_count;
    if (layer >= 0) result.integer_parameters["layer"] = layer;
    return result;
}

static KernelSpecialization kernel(
    KernelCatalogRef ref, int layer, size_t token_count,
    KernelSpecialization::ExecutionKind kind = KernelSpecialization::ExecutionKind::Native) {
    KernelSpecialization result;
    result.family = ref.family != nullptr ? ref.family : "";
    result.variant = ref.name != nullptr ? ref.name : "";
    result.kernel_id = ref.id;
    result.execution_kind = kind;
    result.integer_parameters["token_count"] = token_count;
    if (layer >= 0) result.integer_parameters["layer"] = layer;
    return result;
}

static KernelSpecialization native_gap(const std::string & variant, int layer, size_t token_count) {
    KernelSpecialization result = invocation_kernel(variant, layer, token_count);
    result.execution_kind = KernelSpecialization::ExecutionKind::NativeGap;
    return result;
}

static KernelSpecialization storage_kernel(
    KernelCatalogRef q4_variant, KernelCatalogRef q6_variant, const std::string & semantic_role,
    enum ggml_type type, int layer, size_t token_count, std::vector<std::string> & gaps) {
    if (type == GGML_TYPE_Q4_K && q4_variant.valid()) {
        KernelSpecialization result = kernel(q4_variant, layer, token_count);
        result.integer_parameters["weight_type"] = GGML_TYPE_Q4_K;
        return result;
    }
    if (type == GGML_TYPE_Q6_K && q6_variant.valid()) {
        KernelSpecialization result = kernel(q6_variant, layer, token_count);
        result.integer_parameters["weight_type"] = GGML_TYPE_Q6_K;
        return result;
    }
    const std::string variant = "missing_" + type_suffix(type) + "_" + semantic_role;
    gaps.push_back("layer " + std::to_string(layer) + ": " + variant);
    return native_gap(variant, layer, token_count);
}

static void calculate_boundaries(const Graph & graph, Invocation & invocation) {
    std::set<OperationId> covered(invocation.covered_operations.begin(), invocation.covered_operations.end());
    std::vector<std::vector<OperationId>> consumers(graph.values.size());
    for (const Operation & operation : graph.operations) {
        for (ValueId input : operation.inputs) if (input < consumers.size()) consumers[input].push_back(operation.id);
    }
    std::set<ValueId> roots(graph.roots.begin(), graph.roots.end());
    std::set<ValueId> inputs;
    std::set<ValueId> outputs;
    for (OperationId operation_id : invocation.covered_operations) {
        const Operation & operation = graph.operations[operation_id];
        for (ValueId input : operation.inputs) {
            if (input >= graph.values.size()) continue;
            const OperationId producer = graph.values[input].producer;
            if (producer == kInvalidId || covered.count(producer) == 0) inputs.insert(input);
        }
        bool escapes = roots.count(operation.output) != 0;
        for (OperationId consumer : consumers[operation.output]) escapes |= covered.count(consumer) == 0;
        for (const Effect & effect : operation.effects) escapes |= effect.kind == EffectKind::Write;
        if (escapes) outputs.insert(operation.output);
    }
    size_t ordinal = 0;
    for (ValueId value : inputs) invocation.inputs.push_back({ "arg" + std::to_string(ordinal++), value });
    ordinal = 0;
    for (ValueId value : outputs) invocation.outputs.push_back({ "result" + std::to_string(ordinal++), value });
}

static void add_dispatch(Invocation & invocation, KernelSpecialization specialization, size_t & ordinal) {
    Dispatch dispatch;
    dispatch.kernel = std::move(specialization);
    dispatch.bindings.insert(dispatch.bindings.end(), invocation.inputs.begin(), invocation.inputs.end());
    dispatch.bindings.insert(dispatch.bindings.end(), invocation.outputs.begin(), invocation.outputs.end());
    if (ordinal != 0) dispatch.dependencies.push_back(static_cast<uint32_t>(ordinal - 1));
    invocation.dispatches.push_back(std::move(dispatch));
    ++ordinal;
}

static OperationId layer_start(size_t layer) {
    return layer < 47 ? kFirstLayerOperation + static_cast<OperationId>(layer * kRegularLayerOperationCount)
                      : kTerminalLayerOperation;
}

static std::vector<OperationId> inclusive_range(OperationId first, OperationId last) {
    std::vector<OperationId> result;
    for (OperationId operation = first; operation <= last; ++operation) result.push_back(operation);
    return result;
}

static void append_prefill_layer_dispatches(const Graph & graph, Invocation & invocation, size_t layer,
                                            size_t token_count, size_t & ordinal, std::vector<std::string> & gaps) {
    const OperationId start = layer_start(layer);
    const bool terminal = layer == 47;
    const size_t active_token_count = terminal ? 1 : token_count;
    if (layer == 0) add_dispatch(invocation, kernel(K("qwen3_moe_rmsnorm_f32"), layer, token_count), ordinal);
    const std::array<OperationId, 3> projection_ops = { start + 2, start + 7, start + 9 };
    for (size_t i = 0; i < projection_ops.size(); ++i) {
        add_dispatch(invocation, storage_kernel(K("qwen3_moe_dense_linear_q4k_f16_wmma"), K("qwen3_moe_dense_linear_q6k_f16_wmma"), "dense_attention_projection",
            weight_type(graph, projection_ops[i]), layer, token_count, gaps), ordinal);
    }
    add_dispatch(invocation, kernel(K("qwen3_moe_attention_postprocess_f32_f16"), layer, token_count), ordinal);
    add_dispatch(invocation, kernel(K("qwen3_moe_flash_attention_f32_f16_wmma"), layer, token_count), ordinal);
    add_dispatch(invocation, storage_kernel(K("qwen3_moe_dense_linear_q4k_f16_wmma"), K("qwen3_moe_dense_linear_q6k_f16_wmma"), "dense_attention_output",
        weight_type(graph, start + 26), layer, token_count, gaps), ordinal);
    add_dispatch(invocation, kernel(K("qwen3_moe_rmsnorm_f32"), layer, active_token_count), ordinal);
    add_dispatch(invocation, kernel(K("qwen3_moe_router_projection_f32_four_row_wave32"), layer, active_token_count), ordinal);
    add_dispatch(invocation, kernel(K("qwen3_moe_router_top8_f32"), layer, active_token_count), ordinal);
    add_dispatch(invocation, kernel(K("qwen3_moe_build_expert_table"), layer, active_token_count), ordinal);
    add_dispatch(invocation, kernel(K("qwen3_moe_build_expert_partition_table"), layer, active_token_count), ordinal);
    add_dispatch(invocation, kernel(K("qwen3_moe_routed_gate_up_swiglu_q4k_f16_wmma"), layer, active_token_count), ordinal);
    const OperationId down_operation = start + (terminal ? 47 : 45);
    add_dispatch(invocation, storage_kernel(K("qwen3_moe_routed_down_q4k_f16_wmma_grouped"), K("qwen3_moe_routed_down_q6k_f16_wmma_grouped"), "routed_down",
        weight_type(graph, down_operation), layer, active_token_count, gaps), ordinal);
    add_dispatch(invocation, kernel(K("qwen3_moe_routed_down_weighted_reduce_f16_f32"), layer, active_token_count), ordinal);
    if (!terminal) add_dispatch(invocation, kernel(K("qwen3_moe_rmsnorm_f32"), layer, token_count), ordinal);
}

static void append_decode_layer_dispatches(const Graph & graph, Invocation & invocation, size_t layer,
                                           size_t token_count, size_t & ordinal, std::vector<std::string> & gaps) {
    const OperationId start = layer_start(layer);
    const bool terminal = layer == 47;
    const enum ggml_type query_type = weight_type(graph, start + 2);
    // llama.cpp emits the projection matmuls in Q, V, K order. Keep the
    // physical operation ordinals separate from the semantic kernel roles:
    // Ben's mixed-format specialization requires Q4 Q/K and permits Q4/Q6 V.
    const enum ggml_type value_type = weight_type(graph, start + 7);
    const enum ggml_type key_type = weight_type(graph, start + 9);
    if ((query_type == GGML_TYPE_Q4_K) && (key_type == GGML_TYPE_Q4_K) &&
        (value_type == GGML_TYPE_Q4_K || value_type == GGML_TYPE_Q6_K)) {
        KernelSpecialization qkv = kernel(K("qwen3_moe_attention_qkv_quantized"), layer, token_count);
        qkv.integer_parameters["query_weight_type"] = query_type;
        qkv.integer_parameters["key_weight_type"] = key_type;
        qkv.integer_parameters["value_weight_type"] = value_type;
        add_dispatch(invocation, std::move(qkv), ordinal);
    } else {
        const std::string variant = "qwen3_moe_attention_qkv_postprocess_fused_decode_missing_" +
            type_suffix(query_type) + "_" + type_suffix(key_type) + "_" + type_suffix(value_type);
        gaps.push_back("layer " + std::to_string(layer) + ": " + variant);
        add_dispatch(invocation, native_gap(variant, layer, token_count), ordinal);
    }
    add_dispatch(invocation, kernel(K("qwen3_moe_attention_postprocess_f32_f16"), layer, token_count), ordinal);
    AttentionGeometry attention_geometry;
    std::vector<std::string> unreachable_errors;
    const bool has_attention_geometry = recover_attention_geometry(graph, layer, attention_geometry, unreachable_errors);
    KernelSpecialization flash = kernel(K("qwen3_moe_flash_attention_decode_split_f32_f16_wmma"), layer, token_count);
    if (has_attention_geometry) {
        flash.integer_parameters["key_value_token_count"] = attention_geometry.key_value_token_count;
        flash.integer_parameters["key_value_tile_size"] = kAttentionKvTileSize;
        flash.integer_parameters["key_value_block_count"] =
            (attention_geometry.key_value_token_count + kAttentionKvTileSize - 1) / kAttentionKvTileSize;
    }
    add_dispatch(invocation, std::move(flash), ordinal);
    add_dispatch(invocation, kernel(K("ggml_quantize_q8_1_x4_f32"), layer, token_count), ordinal);
    add_dispatch(invocation, storage_kernel(K("qwen3_moe_dense_linear_q4k_q8_1_x4"),
        kNoKernelRef, "dense_attention_output_q8", weight_type(graph, start + 26), layer, token_count, gaps), ordinal);
    add_dispatch(invocation, kernel(K("qwen3_moe_rmsnorm_f32_quantize_q8_1_x4"), layer, token_count), ordinal);
    add_dispatch(invocation, kernel(K("qwen3_moe_router_projection_f32_one_row_wave64"), layer, token_count), ordinal);
    add_dispatch(invocation, kernel(K("qwen3_moe_router_top8_f32"), layer, token_count), ordinal);
    add_dispatch(invocation, kernel(K("qwen3_moe_routed_gate_up_swiglu_q4k_q8"), layer, token_count), ordinal);
    add_dispatch(invocation, kernel(K("ggml_quantize_q8_1_x4_f32"), layer, token_count * 8), ordinal);
    const OperationId down_operation = start + (terminal ? 47 : 45);
    add_dispatch(invocation, storage_kernel(K("qwen3_moe_routed_down_q4k_q8_1_x4"), K("qwen3_moe_routed_down_q6k_q8_1_x4"), "routed_down_q8",
        weight_type(graph, down_operation), layer, token_count, gaps), ordinal);
    const KernelCatalogRef terminal_norm =
        terminal ? K("qwen3_moe_rmsnorm_f32_quantize_q8_1_x4")
                 : K("qwen3_moe_attention_rmsnorm_quantize_q8_1_x4");
    add_dispatch(invocation, kernel(terminal_norm, layer, token_count), ordinal);
}

} // namespace

bool QwenProgramProof::structurally_sufficient() const {
    return recognized() && schedule_dispatch_count(schedule) == schedule.expected_dispatch_count &&
        schedule_execution_kind_count(schedule, KernelSpecialization::ExecutionKind::CpuFallback) == 0;
}

bool QwenProgramProof::natively_complete() const {
    return structurally_sufficient() && native_gaps.empty() &&
        schedule_execution_kind_count(schedule, KernelSpecialization::ExecutionKind::NativeGap) == 0;
}

QwenProgramProof recover_owned_qwen3_moe_program(const Graph & graph) {
    QwenProgramProof proof;
    if (!graph.valid()) {
        proof.errors.push_back("cannot recover a Qwen program from an invalid graph");
        return proof;
    }
    if (graph.operations.size() != 3030 || graph.roots.size() != 2) {
        proof.errors.push_back("graph does not have the canonical 3030-op/two-root Qwen shape");
        return proof;
    }
    if (graph.operations[0].op != GGML_OP_GET_ROWS || graph.operations[3027].op != GGML_OP_RMS_NORM ||
        graph.operations[3028].op != GGML_OP_MUL || graph.operations[3029].op != GGML_OP_MUL_MAT) {
        proof.errors.push_back("Qwen graph preamble or endpoint signature does not match");
        return proof;
    }
    for (size_t layer = 0; layer < 47; ++layer) {
        if (!check_signature(graph, layer_start(layer), regular_layer_signature(),
                             "layer " + std::to_string(layer), proof.errors)) return proof;
    }
    if (!check_signature(graph, kTerminalLayerOperation, terminal_layer_signature(), "terminal layer", proof.errors)) return proof;
    const Value & activation = graph.values[graph.operations[0].output];
    const size_t token_count = static_cast<size_t>(activation.access.shape[1]);
    const Value & logits = graph.values[graph.operations[kEndpointOperation + 2].output];
    const size_t output_token_count = static_cast<size_t>(logits.access.shape[1]);
    const bool prefill = token_count > 1 && token_count <= kPrefillTokenMaximum;
    const bool decode = token_count == 1;
    if (!prefill && !decode) {
        proof.errors.push_back("canonical topology token count " + std::to_string(token_count) +
            " is outside the owned decode=1/prefill=2.." + std::to_string(kPrefillTokenMaximum) + " kernel contract");
        return proof;
    }
    if (output_token_count != 1) {
        proof.errors.push_back("endpoint token count " + std::to_string(output_token_count) +
            " requires runtime-selected multi-row materialization, which the owned schedule does not yet lower");
        return proof;
    }
    const RoutedTransformerGeometry model = recover_model_geometry(graph);
    const Value & embedding_weight = graph.values[graph.operations[0].inputs[0]];
    const int64_t vocabulary_count = embedding_weight.access.shape[1];
    if (model.hidden_size <= 0 || model.query_size <= 0 || model.key_value_size <= 0 ||
        model.expert_count <= 0 || model.route_count <= 0 || model.expert_intermediate_size <= 0 ||
        embedding_weight.access.shape[0] != model.hidden_size || vocabulary_count <= 0) {
        proof.errors.push_back("Qwen topology contains invalid recovered model geometry");
        return proof;
    }
    int64_t program_key_value_token_count = 0;
    for (size_t layer = 0; layer < kLayerCount; ++layer) {
        if (!validate_layer_facts(graph, layer, token_count, model, proof.errors)) return proof;
        AttentionGeometry attention_geometry;
        if (!recover_attention_geometry(graph, layer, attention_geometry, proof.errors)) return proof;
        if (layer == 0) program_key_value_token_count = attention_geometry.key_value_token_count;
        if (attention_geometry.key_value_token_count != program_key_value_token_count) {
            proof.errors.push_back("attention KV extent is not coherent across all owned layers");
            return proof;
        }
    }

    Schedule & schedule = proof.schedule;
    schedule.graph_fingerprint = graph.fingerprint;
    // Prefill specializations are selected by the number of query tokens. The
    // KV dimension is padded capacity (for example, 23 queries with a 256-row
    // attention view), so using it here aliases distinct JIT programs. Decode
    // is one query and remains specialized by its varying padded KV extent.
    schedule.workload = prefill ? "prefill-" + std::to_string(token_count)
                                : "decode-" + std::to_string(program_key_value_token_count);
    schedule.oracle_revision = kOracleRevision;
    schedule.expected_dispatch_count = prefill ? 724 : 580;
    for (ValueId root : graph.roots) {
        const Value & value = graph.values[root];
        const std::string materialization = value.access.shape[0] == vocabulary_count
            ? "full_f32_logits" : "normalized_f32_hidden_state";
        schedule.roots.push_back({ root, RootDisposition::Materialized, materialization });
    }

    size_t dispatch_ordinal = 0;
    Invocation preamble;
    preamble.stage = "program.preamble";
    preamble.covered_operations = prefill ? inclusive_range(0, 0) : inclusive_range(0, 2);
    preamble.kernel = invocation_kernel("owned_program_preamble", -1, token_count);
    calculate_boundaries(graph, preamble);
    KernelSpecialization embedding = kernel(K("qwen_token_embedding_q4k_bringup_workaround"), -1, token_count);
    embedding.integer_parameters["vocabulary_count"] = vocabulary_count;
    embedding.integer_parameters["hidden_size"] = model.hidden_size;
    add_dispatch(preamble, std::move(embedding), dispatch_ordinal);
    KernelSpecialization metadata = kernel(K("qwen_attention_metadata_bringup_workaround"), -1, token_count);
    metadata.integer_parameters["context_capacity"] = program_key_value_token_count;
    add_dispatch(preamble, std::move(metadata), dispatch_ordinal);
    if (decode) add_dispatch(preamble, kernel(K("qwen3_moe_attention_rmsnorm_quantize_q8_1_x4"), -1, token_count), dispatch_ordinal);
    schedule.invocations.push_back(std::move(preamble));

    for (size_t layer = 0; layer < kLayerCount; ++layer) {
        Invocation invocation;
        invocation.stage = layer == 47 ? "program.terminal_layer" : "program.layer";
        invocation.layer = static_cast<int32_t>(layer);
        invocation.kernel = invocation_kernel(prefill ? "owned_prefill_layer" : "owned_decode_layer", layer, token_count);
        const OperationId start = layer_start(layer);
        const OperationId first = start + ((prefill && layer == 0) ? 0 : 2);
        const OperationId last = layer == 47 ? 3026 : start + 62;
        invocation.covered_operations = inclusive_range(first, last);
        if (layer < 47) {
            invocation.covered_operations.push_back(layer_start(layer + 1));
            invocation.covered_operations.push_back(layer_start(layer + 1) + 1);
        } else if (decode) {
            invocation.covered_operations.push_back(3027);
            invocation.covered_operations.push_back(3028);
        }
        std::sort(invocation.covered_operations.begin(), invocation.covered_operations.end());
        calculate_boundaries(graph, invocation);
        if (prefill) append_prefill_layer_dispatches(
            graph, invocation, layer, token_count, dispatch_ordinal, proof.native_gaps);
        else append_decode_layer_dispatches(graph, invocation, layer, token_count, dispatch_ordinal, proof.native_gaps);
        schedule.invocations.push_back(std::move(invocation));
    }

    Invocation endpoint;
    endpoint.stage = "program.endpoint";
    endpoint.kernel = invocation_kernel("owned_program_endpoint", -1, 1);
    endpoint.covered_operations = prefill ? inclusive_range(3027, 3029) : inclusive_range(3029, 3029);
    calculate_boundaries(graph, endpoint);
    if (prefill) add_dispatch(endpoint, kernel(K("qwen3_moe_rmsnorm_f32_quantize_q8_1_x4"), -1, 1), dispatch_ordinal);
    add_dispatch(endpoint, kernel(K("ggml_linear_q6k_q8_1_x4"), -1, 1), dispatch_ordinal);
    schedule.invocations.push_back(std::move(endpoint));
    if (!proof.native_gaps.empty()) {
        proof.errors.push_back("owned Qwen recovery has no native implementation for " + proof.native_gaps.front());
    }
    return proof;
}

VerificationResult verify_owned_qwen3_moe_program(const Graph & graph, const QwenProgramProof & proof) {
    VerificationResult result;
    if (!proof.recognized()) {
        result.errors.insert(result.errors.end(), proof.errors.begin(), proof.errors.end());
        if (result.errors.empty()) result.errors.push_back("Qwen program was not recognized");
        return result;
    }
    result = verify_schedule(graph, proof.schedule);
    const bool prefill = proof.schedule.workload.rfind("prefill-", 0) == 0;
    const bool decode = proof.schedule.workload.rfind("decode-", 0) == 0;
    if (!prefill && !decode) result.errors.push_back("owned Qwen workload has an unknown execution mode");
    const size_t expected_dispatch_count = prefill ? 724 : 580;
    if (proof.schedule.expected_dispatch_count != expected_dispatch_count ||
        schedule_dispatch_count(proof.schedule) != expected_dispatch_count) {
        result.errors.push_back("owned Qwen dispatch topology does not match the pinned oracle");
    }
    if (schedule_execution_kind_count(proof.schedule, KernelSpecialization::ExecutionKind::CpuFallback) != 0) {
        result.errors.push_back("owned Qwen proof contains CPU fallback");
    }
    if (!proof.natively_complete()) result.errors.push_back("owned Qwen proof is not natively complete");
    if (proof.schedule.oracle_revision != kOracleRevision) result.errors.push_back("owned Qwen oracle revision is not pinned");
    if (proof.schedule.invocations.size() != 50) result.errors.push_back("owned Qwen proof must contain preamble, 48 layers, and endpoint");
    const QwenProgramProof expected = recover_owned_qwen3_moe_program(graph);
    if (!expected.recognized()) {
        result.errors.push_back("independent oracle reconstruction failed");
        return result;
    }
    if (proof.schedule.invocations.size() == expected.schedule.invocations.size()) {
        for (size_t i = 0; i < proof.schedule.invocations.size(); ++i) {
            const Invocation & actual_invocation = proof.schedule.invocations[i];
            const Invocation & expected_invocation = expected.schedule.invocations[i];
            if (actual_invocation.stage != expected_invocation.stage || actual_invocation.layer != expected_invocation.layer ||
                actual_invocation.covered_operations != expected_invocation.covered_operations) {
                result.errors.push_back("invocation " + std::to_string(i) + " does not match the owned program partition");
                continue;
            }
            if (actual_invocation.dispatches.size() != expected_invocation.dispatches.size()) {
                result.errors.push_back("invocation " + std::to_string(i) + " does not match the owned dispatch expansion");
                continue;
            }
            for (size_t j = 0; j < actual_invocation.dispatches.size(); ++j) {
                const Dispatch & actual = actual_invocation.dispatches[j];
                const Dispatch & oracle = expected_invocation.dispatches[j];
                if (actual.kernel.family != oracle.kernel.family || actual.kernel.variant != oracle.kernel.variant ||
                    actual.kernel.execution_kind != oracle.kernel.execution_kind ||
                    actual.kernel.integer_parameters != oracle.kernel.integer_parameters ||
                    actual.kernel.compile_parameters != oracle.kernel.compile_parameters ||
                    actual.dependencies != oracle.dependencies || actual.bindings != oracle.bindings) {
                    result.errors.push_back("dispatch in invocation " + std::to_string(i) +
                        " does not match the pinned owned program oracle");
                    break;
                }
            }
        }
    }
    return result;
}

std::string qwen_program_signature(const QwenProgramProof & proof) {
    std::ostringstream out;
    out << "oracle\t" << proof.schedule.oracle_revision << '\n';
    out << "workload\t" << proof.schedule.workload << '\n';
    out << "dispatches\t" << schedule_dispatch_count(proof.schedule) << '\n';
    out << "native_gaps\t" << proof.native_gaps.size() << '\n';
    for (const RootContract & root : proof.schedule.roots) {
        out << "root\t" << root.value << '\t' << root_disposition_name(root.disposition) << '\t' << root.replacement << '\n';
    }
    size_t ordinal = 0;
    for (const Invocation & invocation : proof.schedule.invocations) {
        for (const Dispatch & dispatch : invocation.dispatches) {
            out << "dispatch\t" << ordinal++ << '\t' << invocation.stage << '\t' << invocation.layer << '\t'
                << execution_kind_name(dispatch.kernel.execution_kind) << '\t' << dispatch.kernel.variant << '\n';
        }
    }
    for (const std::string & gap : proof.native_gaps) out << "gap\t" << gap << '\n';
    return out.str();
}

#undef K

} // namespace ggml::hrx
