#include "value-map.h"

#include "ggml-impl.h"

#include <utility>

namespace ggml::hrx {

const Value * ValueMap::find_alias_source(const ggml_tensor * tensor) const {
    if (tensor == nullptr || tensor->view_src == nullptr) {
        return nullptr;
    }
    return find_tensor(tensor->view_src);
}

ValueId ValueMap::get_or_add_tensor_value(const ggml_tensor * tensor, ValueKind kind) {
    const auto found = tensor_values_.find(tensor);
    if (found != tensor_values_.end()) {
        Value & value = values_[found->second];
        if (kind == ValueKind::External) {
            value.kind = ValueKind::External;
        }
        return value.id;
    }

    const ValueId  id(static_cast<int32_t>(values_.size()));
    const Value *  alias_source = find_alias_source(tensor);
    ValueStorageId storage;
    ValueId        storage_root;
    ValueId        alias_source_id;
    size_t         storage_offset     = 0;
    size_t         storage_byte_count = ggml_nbytes(tensor);
    if (alias_source != nullptr) {
        storage            = alias_source->storage;
        storage_root       = alias_source->storage_root;
        alias_source_id    = alias_source->id;
        storage_offset     = tensor->view_offs;
        storage_byte_count = alias_source->storage_byte_count;
    } else {
        storage      = ValueStorageId(static_cast<int32_t>(storages_.size()));
        storage_root = id;
        storages_.push_back({ storage, storage_root, storage_byte_count });
    }

    Value value = {
        id,
        kind,
        storage,
        storage_root,
        alias_source_id,
        storage_offset,
        storage_byte_count,
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

const ValueStorage * ValueMap::find_storage(ValueStorageId id) const {
    if (id.value < 0 || static_cast<size_t>(id.value) >= storages_.size()) {
        return nullptr;
    }
    return &storages_[static_cast<size_t>(id.value)];
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

std::optional<ValueBufferBinding> ValueMap::resolve_buffer_binding(ValueId id) const {
    const Value * value = find(id);
    if (value == nullptr) {
        return std::nullopt;
    }
    if (value->buffer.has_value()) {
        return value->buffer;
    }
    if (value->storage_root == value->id) {
        return std::nullopt;
    }
    const Value * root = find(value->storage_root);
    if (root == nullptr || !root->buffer.has_value()) {
        return std::nullopt;
    }
    ValueBufferBinding binding = *root->buffer;
    if (value->storage_offset > binding.length) {
        return std::nullopt;
    }
    if (value->byte_count > binding.length - value->storage_offset) {
        return std::nullopt;
    }
    binding.offset += value->storage_offset;
    binding.length = value->byte_count;
    return binding;
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

ValueId ValueMap::storage_root(ValueId id) const {
    const Value * value = find(id);
    return value == nullptr ? ValueId() : value->storage_root;
}

bool ValueMap::same_storage(ValueId lhs, ValueId rhs) const {
    const Value * lhs_value = find(lhs);
    const Value * rhs_value = find(rhs);
    return lhs_value != nullptr && rhs_value != nullptr && lhs_value->storage == rhs_value->storage;
}

const Value * ValueMap::find_tensor(const ggml_tensor * tensor) const {
    const auto found = tensor_values_.find(tensor);
    if (found == tensor_values_.end()) {
        return nullptr;
    }
    return &values_[found->second];
}

}  // namespace ggml::hrx
