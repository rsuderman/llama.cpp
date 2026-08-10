#include "dispatch-add.h"

#include "dispatch-scheduler.h"
#include "ggml.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <limits>
#include <utility>

namespace ggml::hrx {

static constexpr KernelCatalogRef kAddF32Kernel = GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_add_f32");

static bool same_shape(const Value & lhs, const Value & rhs) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (lhs.ne[i] != rhs.ne[i]) {
            return false;
        }
    }
    return true;
}

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool has_external_buffer(const Value * value) {
    return value != nullptr && value->kind == ValueKind::External && value->buffer.has_value() &&
           value->buffer->buffer != nullptr;
}

bool supports_add_f32_dispatch(const Graph & graph, const GraphNode * node) {
    if (node == nullptr || node->op != GGML_OP_ADD || node->inputs.size() != 2) {
        return false;
    }
    const Value * output = graph_value(graph, node->output);
    const Value * a      = graph_value(graph, node->inputs[0]);
    const Value * b      = graph_value(graph, node->inputs[1]);
    if (output == nullptr || a == nullptr || b == nullptr) {
        return false;
    }
    return output->type == GGML_TYPE_F32 && a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 &&
           same_shape(*output, *a) && same_shape(*output, *b) && output->contiguous && a->contiguous && b->contiguous &&
           output->element_count > 0 &&
           static_cast<uint64_t>(output->element_count) <= std::numeric_limits<uint32_t>::max();
}

bool try_match_add_f32_dispatch(const Graph & graph, const GraphNode * node, DispatchScheduler & scheduler) {
    if (!supports_add_f32_dispatch(graph, node)) {
        return false;
    }
    const Value * output = graph_value(graph, node->output);
    const Value * a      = graph_value(graph, node->inputs[0]);
    const Value * b      = graph_value(graph, node->inputs[1]);
    if (!has_external_buffer(a) || !has_external_buffer(b) || !has_external_buffer(output)) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kAddF32Kernel);
    dispatch.kernel.integer_parameters.emplace("element_count", output->element_count);
    dispatch.bindings.push_back({ a->id, a->buffer->buffer, a->buffer->offset, a->buffer->length });
    dispatch.bindings.push_back({ b->id, b->buffer->buffer, b->buffer->offset, b->buffer->length });
    dispatch.bindings.push_back({ output->id, output->buffer->buffer, output->buffer->offset, output->buffer->length });

    scheduler.enqueue(std::move(dispatch));
    return true;
}

}  // namespace ggml::hrx
