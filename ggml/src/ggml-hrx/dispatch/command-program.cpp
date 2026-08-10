#include "command-program.h"

#include "kernel-corpus/kernel-corpus-catalog.h"

#include <string>
#include <utility>

namespace ggml::hrx {

CommandProgram build_command_program(const CommandPlan & plan) {
    CommandProgram result;
    if (!plan.valid()) {
        result.errors.push_back(plan.error);
        return result;
    }

    result.commands.reserve(plan.dispatches.size());
    for (const Dispatch & dispatch : plan.dispatches) {
        Command command;
        command.ordinal = static_cast<uint32_t>(result.commands.size());
        command.kind    = CommandKind::Kernel;
        command.kernel  = dispatch.kernel;
        command.bindings.reserve(dispatch.bindings.size());
        for (const DispatchBinding & binding : dispatch.bindings) {
            command.bindings.push_back(
                { binding.value, CommandBindingOrigin::GraphValue, binding.buffer, binding.offset, binding.length });
        }
        result.commands.push_back(std::move(command));
    }
    return result;
}

VerificationResult verify_command_program(const CommandProgram & program) {
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
            result.errors.push_back("command " + std::to_string(command.ordinal) + " is not a kernel command");
        }
        if (command.kernel.kernel_id == kUncatalogedKernelId) {
            result.errors.push_back("command " + std::to_string(command.ordinal) + " has no kernel id");
        }
        if (command.bindings.empty()) {
            result.errors.push_back("command " + std::to_string(command.ordinal) + " has no bindings");
        }
        for (uint32_t dependency : command.dependencies) {
            if (dependency >= command.ordinal) {
                result.errors.push_back("command " + std::to_string(command.ordinal) + " has a forward dependency");
            }
        }
        for (const CommandBinding & binding : command.bindings) {
            if (binding.origin != CommandBindingOrigin::GraphValue) {
                result.errors.push_back("command " + std::to_string(command.ordinal) + " has a non-graph binding");
            }
            if (binding.value.value < 0) {
                result.errors.push_back("command " + std::to_string(command.ordinal) + " has an invalid value id");
            }
            if (binding.buffer == nullptr) {
                result.errors.push_back("command " + std::to_string(command.ordinal) + " has an unbound buffer");
            }
            if (binding.length == 0) {
                result.errors.push_back("command " + std::to_string(command.ordinal) + " has an empty binding");
            }
        }
    }
    return result;
}

}  // namespace ggml::hrx
