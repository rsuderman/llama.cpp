#include "dispatch-add.h"

#include "dispatch-scheduler.h"
#include "ggml.h"
#include "kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <limits>
#include <utility>

namespace ggml::hrx {

static constexpr KernelCatalogRef kAddF32Kernel = GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_add_f32");

static bool same_shape(const ggml_tensor * lhs, const ggml_tensor * rhs) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (lhs->ne[i] != rhs->ne[i]) {
            return false;
        }
    }
    return true;
}

bool supports_add_f32_dispatch(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_ADD || op->src[0] == nullptr || op->src[1] == nullptr) {
        return false;
    }
    const ggml_tensor * a = op->src[0];
    const ggml_tensor * b = op->src[1];
    const int64_t       n = ggml_nelements(op);
    return op->type == GGML_TYPE_F32 && a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 && same_shape(op, a) &&
           same_shape(op, b) && ggml_is_contiguous(op) && ggml_is_contiguous(a) && ggml_is_contiguous(b) && n > 0 &&
           static_cast<uint64_t>(n) <= std::numeric_limits<uint32_t>::max();
}

bool try_match_add_f32_dispatch(const ggml_tensor *          node,
                                const DispatchMatchContext & context,
                                DispatchScheduler &          scheduler) {
    if (!supports_add_f32_dispatch(node) || context.bind_tensor == nullptr) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kAddF32Kernel);
    dispatch.kernel.integer_parameters.emplace("element_count", ggml_nelements(node));
    dispatch.bindings.resize(3);
    if (!context.bind_tensor(node->src[0], dispatch.bindings[0], context.user_data) ||
        !context.bind_tensor(node->src[1], dispatch.bindings[1], context.user_data) ||
        !context.bind_tensor(node, dispatch.bindings[2], context.user_data)) {
        return false;
    }

    scheduler.enqueue(std::move(dispatch));
    return true;
}

}  // namespace ggml::hrx
