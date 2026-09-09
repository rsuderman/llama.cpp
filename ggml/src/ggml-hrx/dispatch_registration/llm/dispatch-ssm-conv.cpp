#include "dispatch-ssm-conv.h"

#include "ggml.h"
#include "graph/graph-matcher.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kSsmConvSnapshotKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "llm_ssm_conv_snapshot_window_tail_f32");
static constexpr KernelCatalogRef kSsmConvPrefillKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "llm_ssm_conv_dconv4_silu_prefill_512_wg1024");
static constexpr KernelCatalogRef kSsmConvDecodeKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "llm_ssm_conv_dconv4_silu_decode_f32");
static constexpr KernelCatalogRef kSsmConvRollbackKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "llm_ssm_conv_dconv4_silu_rollback_f32");

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool is_f32(const Value * value) {
    return value != nullptr && value->type == GGML_TYPE_F32;
}

static bool is_shape(const Value & value, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    return value.ne[0] == ne0 && value.ne[1] == ne1 && value.ne[2] == ne2 && value.ne[3] == ne3;
}

static bool supported_hidden_size(int64_t hidden_size) {
    return hidden_size >= 32 && hidden_size <= 65536 && hidden_size % 32 == 0;
}

static const Value * root_alias_value(const Graph & graph, const Value * value) {
    for (size_t depth = 0; value != nullptr && value->alias_source.value >= 0 && depth < graph.values().size();
         ++depth) {
        value = graph_value(graph, value->alias_source);
    }
    return value;
}

static bool distinct_storage(const Value & lhs, const Value & rhs) {
    return lhs.storage != rhs.storage;
}

static bool ranges_overlap(const Value & lhs, const Value & rhs) {
    if (lhs.storage != rhs.storage || lhs.byte_count == 0 || rhs.byte_count == 0) {
        return false;
    }
    if (lhs.storage_offset <= rhs.storage_offset) {
        return rhs.storage_offset - lhs.storage_offset < lhs.byte_count;
    }
    return lhs.storage_offset - rhs.storage_offset < rhs.byte_count;
}

static bool append_covered_node(const DispatchMatchContext & context, const GraphNode * node, DispatchMatch & match) {
    return append_covered_node_index_once(context.graph, context.covered_nodes, node, match.covered_nodes);
}

static void set_compile_parameter(KernelSpecialization & kernel, const char * name, int64_t value) {
    kernel.compile_parameters.emplace(name, std::to_string(value));
}

struct SsmConvCacheUpdate {
    const GraphNode * state_tail_view = nullptr;
    const GraphNode * cache_view      = nullptr;
    const GraphNode * cache_copy      = nullptr;
    const Value *     state_tail      = nullptr;
    const Value *     cache           = nullptr;
    int64_t           source_row      = 0;
};

struct SsmConvPrefillMatch {
    const GraphNode *               concat = nullptr;
    const GraphNode *               ssm    = nullptr;
    const GraphNode *               silu   = nullptr;
    const Value *                   state  = nullptr;
    const Value *                   x      = nullptr;
    const Value *                   filter = nullptr;
    const Value *                   output = nullptr;
    std::vector<SsmConvCacheUpdate> cache_updates;
    int64_t                         hidden_size    = 0;
    int64_t                         token_count    = 0;
    int64_t                         sequence_count = 0;

    bool matched() const {
        return concat != nullptr && ssm != nullptr && silu != nullptr && state != nullptr && x != nullptr &&
               filter != nullptr && output != nullptr && !cache_updates.empty();
    }
};

static SsmConvPrefillMatch match_ssm_conv_prefill(const Graph & graph, const GraphNode * node) {
    SsmConvPrefillMatch match;
    if (node == nullptr || node->op != GGML_OP_CONCAT || node->inputs.size() != 2 || !graph.has_index()) {
        return match;
    }

    const Value * state        = graph_value(graph, node->inputs[0]);
    const Value * x_transposed = graph_value(graph, node->inputs[1]);
    const Value * window       = graph_value(graph, node->output);
    if (!is_f32(state) || !is_f32(x_transposed) || !is_f32(window)) {
        return {};
    }

    const GraphNode * transpose = graph.index().producer(x_transposed->id);
    if (transpose == nullptr || transpose->op != GGML_OP_TRANSPOSE || transpose->inputs.size() != 1) {
        return {};
    }
    const Value * x_layout = graph_value(graph, transpose->inputs[0]);
    const Value * x        = root_alias_value(graph, x_layout);
    if (!is_f32(x_layout) || !x_layout->contiguous || !is_f32(x) || !x->contiguous) {
        return {};
    }

    const int64_t hidden_size    = x_layout->ne[0];
    const int64_t token_count    = x_layout->ne[1];
    const int64_t sequence_count = x_layout->ne[2];
    if (!supported_hidden_size(hidden_size) || token_count < 1 || token_count > 512 || sequence_count < 1 ||
        sequence_count > 3 || !is_shape(*x_layout, hidden_size, token_count, sequence_count, 1) ||
        !is_shape(*x_transposed, token_count, hidden_size, sequence_count, 1) ||
        !is_shape(*state, 3, hidden_size, sequence_count, 1) ||
        !is_shape(*window, token_count + 3, hidden_size, sequence_count, 1)) {
        return {};
    }
    if (x_layout->nb[0] != sizeof(float) || x_layout->nb[1] != static_cast<size_t>(hidden_size) * sizeof(float) ||
        x_layout->nb[2] != static_cast<size_t>(hidden_size * token_count) * sizeof(float) ||
        x_transposed->nb[0] != static_cast<size_t>(hidden_size) * sizeof(float) ||
        x_transposed->nb[1] != sizeof(float) ||
        x_transposed->nb[2] != static_cast<size_t>(hidden_size * token_count) * sizeof(float) ||
        state->nb[0] != sizeof(float) || state->nb[1] != 3 * sizeof(float) ||
        state->nb[2] != static_cast<size_t>(3 * hidden_size) * sizeof(float) || window->nb[0] != sizeof(float) ||
        window->nb[1] != static_cast<size_t>(token_count + 3) * sizeof(float) ||
        window->nb[2] != static_cast<size_t>((token_count + 3) * hidden_size) * sizeof(float)) {
        return {};
    }

    const std::vector<const GraphNode *> & window_consumers = graph.index().consumers(window->id);
    if (window_consumers.size() < 2 || window_consumers.size() > 6) {
        return {};
    }
    std::vector<const GraphNode *> state_tail_views;
    const GraphNode *              ssm = nullptr;
    for (const GraphNode * consumer : window_consumers) {
        if (consumer != nullptr && consumer->op == GGML_OP_VIEW) {
            state_tail_views.push_back(consumer);
        } else if (consumer != nullptr && consumer->op == GGML_OP_SSM_CONV) {
            if (ssm != nullptr) {
                return {};
            }
            ssm = consumer;
        } else {
            return {};
        }
    }
    if (state_tail_views.empty() || state_tail_views.size() > 5 || ssm == nullptr || ssm->inputs.size() != 2 ||
        ssm->inputs[0] != window->id) {
        return {};
    }

    const Value * filter     = graph_value(graph, ssm->inputs[1]);
    const Value * ssm_output = graph_value(graph, ssm->output);
    if (!is_f32(filter) || !is_f32(ssm_output) || !filter->contiguous || !ssm_output->contiguous ||
        !is_shape(*filter, 4, hidden_size, 1, 1) ||
        !is_shape(*ssm_output, hidden_size, token_count, sequence_count, 1) || filter->nb[0] != sizeof(float) ||
        filter->nb[1] != 4 * sizeof(float)) {
        return {};
    }

    std::vector<SsmConvCacheUpdate> cache_updates;
    for (const GraphNode * state_tail_view : state_tail_views) {
        if (state_tail_view == nullptr || state_tail_view->inputs.size() != 1) {
            return {};
        }
        const Value * state_tail = graph_value(graph, state_tail_view->output);
        if (!is_f32(state_tail) || state_tail->alias_source != window->id ||
            state_tail->storage_offset < window->storage_offset ||
            (state_tail->storage_offset - window->storage_offset) % sizeof(float) != 0 ||
            !is_shape(*state_tail, 3, hidden_size, sequence_count, 1)) {
            return {};
        }
        const int64_t source_row =
            static_cast<int64_t>((state_tail->storage_offset - window->storage_offset) / sizeof(float));
        if (source_row < 0 || source_row > token_count) {
            return {};
        }
        const std::vector<const GraphNode *> & tail_consumers = graph.index().consumers(state_tail->id);
        if (tail_consumers.size() != 1 || tail_consumers.front() == nullptr ||
            tail_consumers.front()->op != GGML_OP_CPY || tail_consumers.front()->inputs.size() != 2 ||
            tail_consumers.front()->inputs[0] != state_tail->id) {
            return {};
        }
        const GraphNode * cache_copy   = tail_consumers.front();
        const Value *     cache_target = graph_value(graph, cache_copy->inputs[1]);
        const Value *     cache        = graph_value(graph, cache_copy->output);
        const GraphNode * cache_view   = cache_target != nullptr ? graph.index().producer(cache_target->id) : nullptr;
        if (!is_f32(cache_target) || !is_f32(cache) || cache_view == nullptr || cache_view->op != GGML_OP_VIEW ||
            cache_view->inputs.size() != 1 || !same_full_value_range(*cache_target, *cache) ||
            cache->byte_count != static_cast<size_t>(3 * hidden_size * sequence_count) * sizeof(float)) {
            return {};
        }
        cache_updates.push_back({ state_tail_view, cache_view, cache_copy, state_tail, cache, source_row });
    }

    std::sort(cache_updates.begin(), cache_updates.end(),
              [](const auto & lhs, const auto & rhs) { return lhs.cache->storage_offset < rhs.cache->storage_offset; });
    for (size_t slot = 0; slot < cache_updates.size(); ++slot) {
        const int64_t expected_source_row = std::max<int64_t>(0, token_count - static_cast<int64_t>(slot));
        if (cache_updates[slot].source_row != expected_source_row ||
            cache_updates[slot].cache->storage != cache_updates.front().cache->storage) {
            return {};
        }
        for (size_t prior = 0; prior < slot; ++prior) {
            if (ranges_overlap(*cache_updates[slot].cache, *cache_updates[prior].cache)) {
                return {};
            }
        }
    }

    const std::vector<const GraphNode *> & ssm_consumers = graph.index().consumers(ssm_output->id);
    if (ssm_consumers.size() != 1 || ssm_consumers.front() == nullptr || ssm_consumers.front()->op != GGML_OP_UNARY ||
        ssm_consumers.front()->inputs.size() != 1) {
        return {};
    }
    const GraphNode *   silu   = ssm_consumers.front();
    const UnaryParams * unary  = op_params_as<UnaryParams>(silu->params);
    const Value *       output = graph_value(graph, silu->output);
    if (unary == nullptr || unary->op != UnaryKind::Silu || !is_f32(output) || !output->contiguous ||
        !is_shape(*output, hidden_size, token_count, sequence_count, 1)) {
        return {};
    }

    if (!distinct_storage(*state, *x) || !distinct_storage(*state, *filter) || !distinct_storage(*x, *filter) ||
        (!distinct_storage(*output, *x) && !same_full_value_range(*output, *x)) || !distinct_storage(*output, *state) ||
        !distinct_storage(*output, *filter)) {
        return {};
    }
    for (const SsmConvCacheUpdate & update : cache_updates) {
        if (!distinct_storage(*state, *update.cache) || !distinct_storage(*x, *update.cache) ||
            !distinct_storage(*filter, *update.cache) || !distinct_storage(*output, *update.cache)) {
            return {};
        }
    }

    match.concat         = node;
    match.ssm            = ssm;
    match.silu           = silu;
    match.state          = state;
    match.x              = x;
    match.filter         = filter;
    match.output         = output;
    match.cache_updates  = std::move(cache_updates);
    match.hidden_size    = hidden_size;
    match.token_count    = token_count;
    match.sequence_count = sequence_count;
    return match;
}

static bool match_ssm_conv_prefill_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const SsmConvPrefillMatch match = match_ssm_conv_prefill(context.graph, context.root_node);
    if (!match.matched()) {
        return false;
    }

    if (!append_covered_node(context, match.concat, dispatch_match) ||
        !append_covered_node(context, match.ssm, dispatch_match) ||
        !append_covered_node(context, match.silu, dispatch_match)) {
        return false;
    }
    for (const SsmConvCacheUpdate & update : match.cache_updates) {
        if (!append_covered_node(context, update.state_tail_view, dispatch_match) ||
            !append_covered_node(context, update.cache_view, dispatch_match) ||
            !append_covered_node(context, update.cache_copy, dispatch_match)) {
            return false;
        }
    }

    const bool optimized_prefill =
        match.token_count == 512 && match.sequence_count == 1 && match.cache_updates.size() == 1;
    if (!optimized_prefill) {
        Dispatch   ssm_dispatch;
        const bool decode = match.token_count == 1 && match.sequence_count == 1 && match.cache_updates.size() == 1;
        if (decode) {
            ssm_dispatch.kernel = make_kernel_specialization(kSsmConvDecodeKernel);
            set_compile_parameter(ssm_dispatch.kernel, "llm.ssm_conv.decode.d_inner", match.hidden_size);
            set_compile_parameter(ssm_dispatch.kernel, "llm.ssm_conv.decode.workgroup_size", 256);
        } else {
            ssm_dispatch.kernel = make_kernel_specialization(kSsmConvRollbackKernel);
            set_compile_parameter(ssm_dispatch.kernel, "llm.ssm_conv.rollback.d_inner", match.hidden_size);
            set_compile_parameter(ssm_dispatch.kernel, "llm.ssm_conv.rollback.n_t", match.token_count);
            set_compile_parameter(ssm_dispatch.kernel, "llm.ssm_conv.rollback.n_s", match.sequence_count);
            set_compile_parameter(ssm_dispatch.kernel, "llm.ssm_conv.rollback.cache_count", match.cache_updates.size());
            set_compile_parameter(ssm_dispatch.kernel, "llm.ssm_conv.rollback.workgroup_size", 256);
        }
        ssm_dispatch.bindings.push_back({ match.state->id, 0, match.state->byte_count });
        ssm_dispatch.bindings.push_back({ match.x->id, 0, match.x->byte_count });
        ssm_dispatch.bindings.push_back({ match.filter->id, 0, match.filter->byte_count });
        ssm_dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });
        if (decode) {
            const Value * cache = match.cache_updates.front().cache;
            ssm_dispatch.bindings.push_back({ cache->id, 0, cache->byte_count });
        } else {
            for (size_t slot = 0; slot < 5; ++slot) {
                const Value * cache = match.cache_updates[std::min(slot, match.cache_updates.size() - 1)].cache;
                ssm_dispatch.bindings.push_back({ cache->id, 0, cache->byte_count });
            }
        }
        dispatch_match.dispatches.push_back(std::move(ssm_dispatch));
        return true;
    }

    const size_t  snapshot_bytes = static_cast<size_t>(64 * match.hidden_size) * sizeof(float);
    const ValueId snapshot       = context.next_plan_value;
    dispatch_match.transients.push_back({ snapshot, "llm.ssm_conv.window_snapshot", snapshot_bytes, 256 });

    Dispatch snapshot_dispatch;
    snapshot_dispatch.kernel = make_kernel_specialization(kSsmConvSnapshotKernel);
    set_compile_parameter(snapshot_dispatch.kernel, "llm.ssm_conv.snapshot.d_conv", 4);
    set_compile_parameter(snapshot_dispatch.kernel, "llm.ssm_conv.snapshot.d_inner", match.hidden_size);
    set_compile_parameter(snapshot_dispatch.kernel, "llm.ssm_conv.snapshot.n_t", match.token_count);
    set_compile_parameter(snapshot_dispatch.kernel, "llm.ssm_conv.snapshot.n_s", match.sequence_count);
    set_compile_parameter(snapshot_dispatch.kernel, "llm.ssm_conv.snapshot.state_row_stride", 1);
    set_compile_parameter(snapshot_dispatch.kernel, "llm.ssm_conv.snapshot.state_channel_stride", 3);
    set_compile_parameter(snapshot_dispatch.kernel, "llm.ssm_conv.snapshot.x_row_stride", match.hidden_size);
    set_compile_parameter(snapshot_dispatch.kernel, "llm.ssm_conv.snapshot.dst_row_stride", match.hidden_size);
    set_compile_parameter(snapshot_dispatch.kernel, "llm.ssm_conv.snapshot.cache_row_stride", 1);
    set_compile_parameter(snapshot_dispatch.kernel, "llm.ssm_conv.snapshot.cache_channel_stride", 3);
    set_compile_parameter(snapshot_dispatch.kernel, "llm.ssm_conv.snapshot.workgroup_size", 256);
    snapshot_dispatch.bindings.push_back({ match.state->id, 0, match.state->byte_count });
    snapshot_dispatch.bindings.push_back({ match.x->id, 0, match.x->byte_count });
    snapshot_dispatch.bindings.push_back({ snapshot, 0, snapshot_bytes });
    const Value * cache = match.cache_updates.front().cache;
    snapshot_dispatch.bindings.push_back({ cache->id, 0, cache->byte_count });

    Dispatch ssm_dispatch;
    ssm_dispatch.kernel = make_kernel_specialization(kSsmConvPrefillKernel);
    set_compile_parameter(ssm_dispatch.kernel, "llm.ssm_conv.prefill.d_conv", 4);
    set_compile_parameter(ssm_dispatch.kernel, "llm.ssm_conv.prefill.d_inner", match.hidden_size);
    set_compile_parameter(ssm_dispatch.kernel, "llm.ssm_conv.prefill.n_t", match.token_count);
    set_compile_parameter(ssm_dispatch.kernel, "llm.ssm_conv.prefill.n_s", match.sequence_count);
    set_compile_parameter(ssm_dispatch.kernel, "llm.ssm_conv.prefill.state_row_stride", match.hidden_size);
    set_compile_parameter(ssm_dispatch.kernel, "llm.ssm_conv.prefill.x_row_stride", match.hidden_size);
    ssm_dispatch.bindings.push_back({ snapshot, 0, snapshot_bytes });
    ssm_dispatch.bindings.push_back({ match.x->id, 0, match.x->byte_count });
    ssm_dispatch.bindings.push_back({ match.filter->id, 0, match.filter->byte_count });
    ssm_dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });

    dispatch_match.dispatches.push_back(std::move(snapshot_dispatch));
    dispatch_match.dispatches.push_back(std::move(ssm_dispatch));
    return true;
}

}  // namespace

void register_llm_ssm_conv_dispatch(DispatchRegistryBuilder & registry) {
    registry.add({
        "llm.ssm_conv.dconv4_silu",
        GGML_OP_CONCAT,
        DispatchMatchKind::Fused,
        200,
        DispatchSource::Llm,
        match_ssm_conv_prefill_dispatch,
    });
}

}  // namespace ggml::hrx
