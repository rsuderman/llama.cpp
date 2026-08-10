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
        result.errors.push_back(plan.error);
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
            result.errors.push_back(format_kernel_resolve_error(resolved, command.kernel.kernel_id));
            definition = nullptr;
        } else if (dispatch.bindings.size() != definition->bindings.size()) {
            result.errors.push_back("command " + std::to_string(command.ordinal) + " kernel " +
                                    kernel_definition_name(*definition) + " has " +
                                    std::to_string(dispatch.bindings.size()) + " bindings but its ABI requires " +
                                    std::to_string(definition->bindings.size()));
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
        result.errors.insert(result.errors.end(), program.errors.begin(), program.errors.end());
    }
    for (size_t i = 0; i < program.commands.size(); ++i) {
        const Command & command = program.commands[i];
        if (command.ordinal != i) {
            result.errors.push_back("command ordinals are not contiguous");
        }
        if (command.kind != CommandKind::Kernel) {
            result.errors.push_back(command_prefix(command) + " is not a kernel command");
        }
        KernelResolveResult      resolved;
        const KernelDefinition * definition = nullptr;
        if (command.kind == CommandKind::Kernel) {
            resolved   = resolve_kernel_definition(corpus, target, command.kernel.kernel_id);
            definition = resolved.definition;
        }
        if (command.kind == CommandKind::Kernel && !resolved.found()) {
            result.errors.push_back(command_prefix(command) + ": " +
                                    format_kernel_resolve_error(resolved, command.kernel.kernel_id));
        } else if (definition != nullptr) {
            if (command.bindings.size() != definition->bindings.size()) {
                result.errors.push_back(command_prefix(command) + " kernel " + kernel_definition_name(*definition) +
                                        " has " + std::to_string(command.bindings.size()) +
                                        " bindings but its ABI requires " +
                                        std::to_string(definition->bindings.size()));
            }
            const size_t shared_count = std::min(command.bindings.size(), definition->bindings.size());
            for (size_t binding_index = 0; binding_index < shared_count; ++binding_index) {
                const CommandBinding &          binding = command.bindings[binding_index];
                const KernelBindingDefinition & abi     = definition->bindings[binding_index];
                if (!string_equal(binding.name.c_str(), abi.name) || binding.access != abi.access) {
                    result.errors.push_back(command_prefix(command) + " binding " + std::to_string(binding_index) +
                                            " does not match the kernel ABI");
                }
            }
        }
        if (command.bindings.empty()) {
            result.errors.push_back(command_prefix(command) + " has no bindings");
        }
        for (uint32_t dependency : command.dependencies) {
            if (dependency >= command.ordinal) {
                result.errors.push_back(command_prefix(command) + " has a forward dependency");
            }
        }
        for (const CommandBinding & binding : command.bindings) {
            if (binding.origin != CommandBindingOrigin::GraphValue) {
                result.errors.push_back(command_prefix(command) + " has a non-graph binding");
            }
            if (binding.value.value < 0) {
                result.errors.push_back(command_prefix(command) + " has an invalid value id");
            }
            if (binding.buffer == nullptr) {
                result.errors.push_back(command_prefix(command) + " has an unbound buffer");
            }
            if (binding.length == 0) {
                result.errors.push_back(command_prefix(command) + " has an empty binding");
            }
        }
    }
    return result;
}

}  // namespace ggml::hrx
