#include "value-map.h"

#include <utility>

namespace ggml::hrx {

ValueId ValueMap::get_or_add_tensor_value(const ggml_tensor * tensor, ValueKind kind) {
    const auto found = tensor_values_.find(tensor);
    if (found != tensor_values_.end()) {
        Value & value = values_[found->second];
        if (kind == ValueKind::External) {
            value.kind = ValueKind::External;
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
        std::nullopt,
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

bool ValueMap::bind_buffer(ValueId id, ValueBufferBinding binding) {
    if (id.value < 0 || static_cast<size_t>(id.value) >= values_.size()) {
        return false;
    }
    Value & value = values_[static_cast<size_t>(id.value)];
    if (value.kind != ValueKind::External) {
        return false;
    }
    value.buffer = std::move(binding);
    return true;
}

std::vector<ValueId> ValueMap::external_value_ids() const {
    std::vector<ValueId> ids;
    for (const Value & value : values_) {
        if (value.kind == ValueKind::External) {
            ids.push_back(value.id);
        }
    }
    return ids;
}

const Value * ValueMap::find_tensor(const ggml_tensor * tensor) const {
    const auto found = tensor_values_.find(tensor);
    if (found == tensor_values_.end()) {
        return nullptr;
    }
    return &values_[found->second];
}

}  // namespace ggml::hrx
