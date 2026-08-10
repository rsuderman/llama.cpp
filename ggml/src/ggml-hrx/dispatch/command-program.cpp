#include "command-program.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

static bool string_equal(const char * lhs, const char * rhs) {
    return std::strcmp(lhs != nullptr ? lhs : "", rhs != nullptr ? rhs : "") == 0;
}

static std::string string_value(const char * value) {
    return value != nullptr ? value : "";
}

static std::string command_prefix(const Command & command) {
    return "command " + std::to_string(command.ordinal);
}

}  // namespace

CommandProgram build_command_program(const CommandPlan &  plan,
                                     const KernelCorpus & corpus,
                                     const std::string &  target) {
    CommandProgram result;
    if (!plan.valid()) {
        result.errors.append(plan.errors);
        return result;
    }

    result.commands.reserve(plan.dispatches.size());
    for (const Dispatch & dispatch : plan.dispatches) {
        Command command;
        command.ordinal                      = static_cast<uint32_t>(result.commands.size());
        command.kind                         = CommandKind::Kernel;
        command.kernel                       = dispatch.kernel;
        const KernelResolveResult resolved   = resolve_kernel_definition(corpus, target, command.kernel.kernel_id);
        const KernelDefinition *  definition = resolved.definition;
        if (!resolved.found()) {
            result.errors.log("%s", format_kernel_resolve_error(resolved, command.kernel.kernel_id).c_str());
            definition = nullptr;
        } else if (dispatch.bindings.size() != definition->bindings.size()) {
            result.errors.log("command %u kernel %s has %zu bindings but its ABI requires %zu", command.ordinal,
                              kernel_definition_name(*definition).c_str(), dispatch.bindings.size(),
                              definition->bindings.size());
        }
        command.bindings.reserve(dispatch.bindings.size());
        for (size_t binding_index = 0; binding_index < dispatch.bindings.size(); ++binding_index) {
            const DispatchBinding & binding = dispatch.bindings[binding_index];
            CommandBinding          command_binding;
            command_binding.value  = binding.value;
            command_binding.origin = CommandBindingOrigin::GraphValue;
            command_binding.buffer = binding.buffer;
            command_binding.offset = binding.offset;
            command_binding.length = binding.length;
            if (definition != nullptr && binding_index < definition->bindings.size()) {
                command_binding.name   = string_value(definition->bindings[binding_index].name);
                command_binding.access = definition->bindings[binding_index].access;
            }
            command.bindings.push_back(std::move(command_binding));
        }
        result.commands.push_back(std::move(command));
    }
    return result;
}

VerificationResult verify_command_program(const CommandProgram & program,
                                          const KernelCorpus &   corpus,
                                          const std::string &    target) {
    VerificationResult result;
    if (!program.valid()) {
        result.errors.append(program.errors);
    }
    for (size_t i = 0; i < program.commands.size(); ++i) {
        const Command & command = program.commands[i];
        if (command.ordinal != i) {
            result.errors.log("command ordinals are not contiguous");
        }
        if (command.kind != CommandKind::Kernel) {
            result.errors.log("%s is not a kernel command", command_prefix(command).c_str());
        }
        KernelResolveResult      resolved;
        const KernelDefinition * definition = nullptr;
        if (command.kind == CommandKind::Kernel) {
            resolved   = resolve_kernel_definition(corpus, target, command.kernel.kernel_id);
            definition = resolved.definition;
        }
        if (command.kind == CommandKind::Kernel && !resolved.found()) {
            result.errors.log("%s: %s", command_prefix(command).c_str(),
                              format_kernel_resolve_error(resolved, command.kernel.kernel_id).c_str());
        } else if (definition != nullptr) {
            if (command.bindings.size() != definition->bindings.size()) {
                result.errors.log("%s kernel %s has %zu bindings but its ABI requires %zu",
                                  command_prefix(command).c_str(), kernel_definition_name(*definition).c_str(),
                                  command.bindings.size(), definition->bindings.size());
            }
            const size_t shared_count = std::min(command.bindings.size(), definition->bindings.size());
            for (size_t binding_index = 0; binding_index < shared_count; ++binding_index) {
                const CommandBinding &          binding = command.bindings[binding_index];
                const KernelBindingDefinition & abi     = definition->bindings[binding_index];
                if (!string_equal(binding.name.c_str(), abi.name) || binding.access != abi.access) {
                    result.errors.log("%s binding %zu does not match the kernel ABI", command_prefix(command).c_str(),
                                      binding_index);
                }
            }
        }
        if (command.bindings.empty()) {
            result.errors.log("%s has no bindings", command_prefix(command).c_str());
        }
        for (uint32_t dependency : command.dependencies) {
            if (dependency >= command.ordinal) {
                result.errors.log("%s has a forward dependency", command_prefix(command).c_str());
            }
        }
        for (const CommandBinding & binding : command.bindings) {
            if (binding.origin != CommandBindingOrigin::GraphValue) {
                result.errors.log("%s has a non-graph binding", command_prefix(command).c_str());
            }
            if (binding.value.value < 0) {
                result.errors.log("%s has an invalid value id", command_prefix(command).c_str());
            }
            if (binding.buffer == nullptr) {
                result.errors.log("%s has an unbound buffer", command_prefix(command).c_str());
            }
            if (binding.length == 0) {
                result.errors.log("%s has an empty binding", command_prefix(command).c_str());
            }
        }
    }
    return result;
}

}  // namespace ggml::hrx
