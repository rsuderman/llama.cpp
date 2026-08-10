#include "command-program-bindings.h"

#include <sstream>
#include <utility>

namespace ggml::hrx {

CommandProgramBindings CommandProgramBindings::from_value_map(const ValueMap & values) {
    std::vector<CommandProgramBinding> bindings;
    CommandProgramBindings             result;
    for (const ValueId id : values.external_value_ids()) {
        const Value * value = values.find(id);
        if (value == nullptr) {
            result.errors.log("external value %d does not exist", id.value);
            continue;
        }
        if (!value->buffer.has_value()) {
            result.errors.log("external value %d is not bound", id.value);
            continue;
        }
        bindings.push_back({ value->id, value->buffer->buffer, value->buffer->offset, value->buffer->length,
                             value->buffer->identity, value->buffer->generation, value->buffer->capacity });
    }
    return from_bindings(std::move(bindings), result.errors);
}

CommandProgramBindings CommandProgramBindings::from_bindings(std::vector<CommandProgramBinding> bindings,
                                                             const ErrorLog &                   errors) {
    CommandProgramBindings result;
    result.errors.append(errors);
    result.bindings_ = std::move(bindings);
    for (const CommandProgramBinding & binding : result.bindings_) {
        if (binding.buffer == nullptr) {
            result.errors.log("external value %d has a null binding", binding.value.value);
        }
        if (binding.length == 0) {
            result.errors.log("external value %d has an empty binding", binding.value.value);
        }
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

CommandProgramBindingsFingerprint command_program_bindings_fingerprint(const CommandProgramBindings & bindings) {
    std::ostringstream out;
    out << "hrx-bindings-v1";
    for (const CommandProgramBinding & binding : bindings.bindings()) {
        out << "|value=" << binding.value.value << "|identity=" << binding.identity
            << "|generation=" << binding.generation << "|capacity=" << binding.capacity << "|offset=" << binding.offset
            << "|length=" << binding.length;
    }
    return { out.str() };
}

}  // namespace ggml::hrx
