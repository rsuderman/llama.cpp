#include "command-program-bindings.h"

namespace ggml::hrx {

CommandProgramBindings CommandProgramBindings::from_value_map(const ValueMap & values) {
    CommandProgramBindings result;
    for (const ValueId id : values.external_value_ids()) {
        const Value * value = values.find(id);
        if (value == nullptr) {
            result.errors.log("external value %d does not exist", id.value);
            continue;
        }
        if (!value->buffer.has_value() || value->buffer->buffer == nullptr) {
            result.errors.log("external value %d is not bound", id.value);
            continue;
        }
        if (value->buffer->length == 0) {
            result.errors.log("external value %d has an empty binding", id.value);
            continue;
        }
        result.bindings_.push_back({ value->id, value->buffer->buffer, value->buffer->offset, value->buffer->length });
    }
    return result;
}

const CommandProgramBinding * CommandProgramBindings::find(ValueId value) const {
    for (const CommandProgramBinding & binding : bindings_) {
        if (binding.value == value) {
            return &binding;
        }
    }
    return nullptr;
}

}  // namespace ggml::hrx
