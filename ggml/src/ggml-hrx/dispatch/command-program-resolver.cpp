#include "command-program-resolver.h"

#include "command-program-diagnostics.h"

#include <string>

namespace ggml::hrx {
namespace {

static bool resolve_command_binding(const Command &                command,
                                    const CommandBinding &         binding,
                                    const CommandProgramBindings & bindings,
                                    ResolvedBufferRef &            ref,
                                    ErrorLog &                     errors) {
    const std::string command_context = format_command(command);
    const std::string binding_context = format_command_binding(binding);
    if (binding.length == 0) {
        errors.log("%s %s has an empty range", command_context.c_str(), binding_context.c_str());
        return false;
    }
    switch (binding.origin) {
        case CommandBindingOrigin::GraphValue:
            {
                const CommandProgramBinding * concrete = bindings.find(binding.value);
                if (concrete == nullptr) {
                    errors.log("%s %s is not bound", command_context.c_str(), binding_context.c_str());
                    return false;
                }
                if (concrete->buffer == nullptr) {
                    errors.log("%s %s has a null buffer", command_context.c_str(), binding_context.c_str());
                    return false;
                }
                if (binding.offset > concrete->length || binding.length > concrete->length - binding.offset) {
                    errors.log("%s %s is outside runtime binding length %zu", command_context.c_str(),
                               binding_context.c_str(), concrete->length);
                    return false;
                }
                ref = { concrete->buffer, concrete->offset + binding.offset, binding.length };
                return true;
            }
        case CommandBindingOrigin::Transient:
            errors.log("%s %s has no transient allocation", command_context.c_str(), binding_context.c_str());
            return false;
    }
    errors.log("%s %s has an unsupported binding origin", command_context.c_str(), binding_context.c_str());
    return false;
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
