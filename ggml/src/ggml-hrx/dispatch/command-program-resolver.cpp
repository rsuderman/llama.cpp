#include "command-program-resolver.h"

namespace ggml::hrx {
namespace {

static const char * binding_name(const CommandBinding & binding) {
    return binding.name.empty() ? "<unnamed>" : binding.name.c_str();
}

static bool resolve_command_binding(const Command &                command,
                                    const CommandBinding &         binding,
                                    const CommandProgramBindings & bindings,
                                    ResolvedBufferRef &            ref,
                                    ErrorLog &                     errors) {
    if (binding.origin != CommandBindingOrigin::GraphValue) {
        errors.log("command %u binding %s has an unsupported binding origin", command.ordinal, binding_name(binding));
        return false;
    }
    const CommandProgramBinding * concrete = bindings.find(binding.value);
    if (concrete == nullptr) {
        errors.log("command %u binding %s value %d is not bound", command.ordinal, binding_name(binding),
                   binding.value.value);
        return false;
    }
    if (concrete->buffer == nullptr) {
        errors.log("command %u binding %s value %d has a null buffer", command.ordinal, binding_name(binding),
                   binding.value.value);
        return false;
    }
    if (binding.length == 0) {
        errors.log("command %u binding %s has an empty range", command.ordinal, binding_name(binding));
        return false;
    }
    if (binding.offset > concrete->length || binding.length > concrete->length - binding.offset) {
        errors.log("command %u binding %s value %d range [%zu, %zu) is outside runtime binding length %zu",
                   command.ordinal, binding_name(binding), binding.value.value, binding.offset,
                   binding.offset + binding.length, concrete->length);
        return false;
    }
    ref = { concrete->buffer, concrete->offset + binding.offset, binding.length };
    return true;
}

}  // namespace

ResolvedCommandProgram resolve_command_program_bindings(const CommandProgram &         program,
                                                        const CommandProgramBindings & bindings) {
    ResolvedCommandProgram result;
    if (!program.valid()) {
        result.errors.append(program.errors);
    }
    if (!bindings.valid()) {
        result.errors.append(bindings.errors);
    }

    result.commands.reserve(program.commands.size());
    for (const Command & command : program.commands) {
        ResolvedCommand resolved_command;
        resolved_command.ordinal = command.ordinal;
        resolved_command.kind    = command.kind;
        resolved_command.kernel  = command.kernel;
        resolved_command.bindings.reserve(command.bindings.size());

        for (const CommandBinding & binding : command.bindings) {
            ResolvedCommandBinding resolved_binding;
            resolved_binding.binding = binding;
            if (resolve_command_binding(command, binding, bindings, resolved_binding.ref, result.errors)) {
                resolved_command.bindings.push_back(resolved_binding);
            }
        }
        result.commands.push_back(resolved_command);
    }
    return result;
}

}  // namespace ggml::hrx
