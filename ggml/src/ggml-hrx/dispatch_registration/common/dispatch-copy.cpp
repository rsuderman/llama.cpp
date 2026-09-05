#include "dispatch-copy.h"

#include "ggml.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr uint64_t         kMaximumCopyElements = uint64_t{ 1 } << 30;
static constexpr KernelCatalogRef kCopyF32Kernel       = GGML_HRX_KERNEL_REF("loom_libs", "ggml_copy_f32");
static constexpr KernelCatalogRef kCopyStridedSourceF32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_copy_strided_source_f32");
static constexpr KernelCatalogRef kConcatDim0F32Kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_concat_dim0_f32");

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool packed_f32_layout(const Value & value) {
    size_t expected_stride = sizeof(float);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (value.nb[i] != expected_stride) {
            return false;
        }
        expected_stride *= static_cast<size_t>(value.ne[i]);
    }
    return true;
}

static bool f32_element_strides(const Value & value) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (value.ne[i] <= 0 || value.nb[i] % sizeof(float) != 0) {
            return false;
        }
    }
    return value.byte_count % sizeof(float) == 0;
}

static bool match_copy_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_CPY || node->inputs.size() != 2) {
        return false;
    }

    const Value * source = graph_value(context.graph, node->inputs[0]);
    const Value * target = graph_value(context.graph, node->inputs[1]);
    const Value * output = graph_value(context.graph, node->output);
    if (source == nullptr || target == nullptr || output == nullptr || source->type != GGML_TYPE_F32 ||
        target->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 || !target->contiguous || !output->contiguous ||
        !packed_f32_layout(*target) || !packed_f32_layout(*output) || !f32_element_strides(*source) ||
        source->element_count <= 0 || source->element_count != output->element_count ||
        target->element_count != output->element_count || target->byte_count != output->byte_count ||
        source->storage == target->storage || source->storage == output->storage ||
        static_cast<uint64_t>(source->element_count) > kMaximumCopyElements) {
        return false;
    }

    Dispatch dispatch;
    if (packed_f32_layout(*source)) {
        dispatch.kernel = make_kernel_specialization(kCopyF32Kernel);
    } else {
        const size_t source_span = source->byte_count / sizeof(float);
        if (source_span > kMaximumCopyElements) {
            return false;
        }
        dispatch.kernel = make_kernel_specialization(kCopyStridedSourceF32Kernel);
        dispatch.kernel.integer_parameters.emplace("source_span", source_span);
        dispatch.kernel.compile_parameters.emplace("ggml.copy_strided_source_f32.ne0", std::to_string(source->ne[0]));
        dispatch.kernel.compile_parameters.emplace("ggml.copy_strided_source_f32.ne1", std::to_string(source->ne[1]));
        dispatch.kernel.compile_parameters.emplace("ggml.copy_strided_source_f32.ne2", std::to_string(source->ne[2]));
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            dispatch.kernel.compile_parameters.emplace("ggml.copy_strided_source_f32.stride" + std::to_string(i),
                                                       std::to_string(source->nb[i] / sizeof(float)));
        }
    }
    dispatch.kernel.integer_parameters.emplace("element_count", source->element_count);
    dispatch.bindings.push_back({ source->id, 0, source->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_concat_dim0_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_CONCAT || node->inputs.size() != 2) {
        return false;
    }

    const Value * lhs    = graph_value(context.graph, node->inputs[0]);
    const Value * rhs    = graph_value(context.graph, node->inputs[1]);
    const Value * output = graph_value(context.graph, node->output);
    if (lhs == nullptr || rhs == nullptr || output == nullptr || lhs->type != GGML_TYPE_F32 ||
        rhs->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 || !lhs->contiguous || !rhs->contiguous ||
        !output->contiguous || !packed_f32_layout(*lhs) || !packed_f32_layout(*rhs) || !packed_f32_layout(*output) ||
        lhs->ne[0] <= 0 || lhs->ne[0] > 65536 || rhs->ne[0] <= 0 || rhs->ne[0] > 65536 ||
        output->ne[0] != lhs->ne[0] + rhs->ne[0] || output->element_count <= 0 ||
        static_cast<uint64_t>(output->element_count) > kMaximumCopyElements || lhs->storage == rhs->storage ||
        lhs->storage == output->storage || rhs->storage == output->storage) {
        return false;
    }
    for (int dim = 1; dim < GGML_MAX_DIMS; ++dim) {
        if (lhs->ne[dim] != rhs->ne[dim] || lhs->ne[dim] != output->ne[dim]) {
            return false;
        }
    }

    const int64_t row_count = output->element_count / output->ne[0];
    if (row_count < 1 || row_count > 1048576) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kConcatDim0F32Kernel);
    dispatch.kernel.compile_parameters.emplace("ggml.concat_dim0_f32.lhs_width", std::to_string(lhs->ne[0]));
    dispatch.kernel.compile_parameters.emplace("ggml.concat_dim0_f32.rhs_width", std::to_string(rhs->ne[0]));
    dispatch.kernel.compile_parameters.emplace("ggml.concat_dim0_f32.row_count", std::to_string(row_count));
    dispatch.bindings.push_back({ lhs->id, 0, lhs->byte_count });
    dispatch.bindings.push_back({ rhs->id, 0, rhs->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_copy_dispatch(DispatchRegistryBuilder & registry) {
    registry.add({
        "common.copy_f32",
        GGML_OP_CPY,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_copy_f32_dispatch,
    });
    registry.add({
        "common.concat_dim0_f32",
        GGML_OP_CONCAT,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_concat_dim0_f32_dispatch,
    });
}

}  // namespace ggml::hrx
