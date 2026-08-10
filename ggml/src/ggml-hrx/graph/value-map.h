#pragma once

#include "ggml.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

struct ggml_tensor;
typedef struct hrx_buffer_s * hrx_buffer_t;

namespace ggml::hrx {

struct ValueId {
    explicit ValueId(int32_t value) : value(value) {}

    int32_t value;
};

inline bool operator==(ValueId lhs, ValueId rhs) {
    return lhs.value == rhs.value;
}

inline bool operator!=(ValueId lhs, ValueId rhs) {
    return !(lhs == rhs);
}

enum class ValueKind : uint8_t {
    External,
    Transient,
};

struct ValueBufferBinding {
    hrx_buffer_t buffer = nullptr;
    size_t       offset = 0;
    size_t       length = 0;
};

struct Value {
    ValueId                            id;
    ValueKind                          kind;
    ggml_type                          type;
    std::array<int64_t, GGML_MAX_DIMS> ne;
    std::array<size_t, GGML_MAX_DIMS>  nb;
    int64_t                            element_count = 0;
    size_t                             byte_count    = 0;
    bool                               contiguous    = false;
    const ggml_tensor *                tensor        = nullptr;
    std::optional<ValueBufferBinding>  buffer;
};

class ValueMap {
  public:
    ValueId get_or_add_tensor_value(const ggml_tensor *               tensor,
                                    ValueKind                         kind,
                                    std::optional<ValueBufferBinding> buffer);

    const Value * find(ValueId id) const;
    Value *       find(ValueId id);
    const Value * find_tensor(const ggml_tensor * tensor) const;

    const std::vector<Value> & values() const { return values_; }

    size_t size() const { return values_.size(); }

  private:
    std::vector<Value>                              values_;
    std::unordered_map<const ggml_tensor *, size_t> tensor_values_;
};

}  // namespace ggml::hrx
