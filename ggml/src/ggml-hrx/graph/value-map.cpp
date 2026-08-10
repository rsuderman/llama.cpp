#include "value-map.h"

#include <utility>

namespace ggml::hrx {

ValueId ValueMap::get_or_add_tensor_value(const ggml_tensor *               tensor,
                                          ValueKind                         kind,
                                          std::optional<ValueBufferBinding> buffer) {
    const auto found = tensor_values_.find(tensor);
    if (found != tensor_values_.end()) {
        Value & value = values_[found->second];
        if (kind == ValueKind::External) {
            value.kind   = ValueKind::External;
            value.buffer = std::move(buffer);
        }
        return value.id;
    }

    Value value = {
        ValueId(static_cast<int32_t>(values_.size())),
        kind,
        tensor->type,
        {},
        {},
        ggml_nelements(tensor),
        ggml_nbytes(tensor),
        ggml_is_contiguous(tensor),
        tensor,
        std::move(buffer),
    };
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        value.ne[i] = tensor->ne[i];
        value.nb[i] = tensor->nb[i];
    }

    values_.push_back(std::move(value));
    tensor_values_.emplace(tensor, values_.size() - 1);
    return values_.back().id;
}

const Value * ValueMap::find(ValueId id) const {
    if (id.value < 0 || static_cast<size_t>(id.value) >= values_.size()) {
        return nullptr;
    }
    return &values_[static_cast<size_t>(id.value)];
}

Value * ValueMap::find(ValueId id) {
    if (id.value < 0 || static_cast<size_t>(id.value) >= values_.size()) {
        return nullptr;
    }
    return &values_[static_cast<size_t>(id.value)];
}

const Value * ValueMap::find_tensor(const ggml_tensor * tensor) const {
    const auto found = tensor_values_.find(tensor);
    if (found == tensor_values_.end()) {
        return nullptr;
    }
    return &values_[found->second];
}

}  // namespace ggml::hrx
