#pragma once

#include "command-plan.h"
#include "kernel-corpus/kernel-types.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ggml::hrx {

enum class CommandKind : uint8_t {
    Kernel,
};

enum class CommandBindingOrigin : uint8_t {
    GraphValue,
};

struct CommandBinding {
    ValueId              value;
    CommandBindingOrigin origin = CommandBindingOrigin::GraphValue;
    hrx_buffer_t         buffer = nullptr;
    size_t               offset = 0;
    size_t               length = 0;
};

struct Command {
    uint32_t                    ordinal = 0;
    CommandKind                 kind    = CommandKind::Kernel;
    KernelSpecialization        kernel;
    std::vector<CommandBinding> bindings;
    std::vector<uint32_t>       dependencies;
};

struct CommandProgram {
    std::vector<Command>     commands;
    std::vector<std::string> errors;

    bool valid() const { return errors.empty(); }
};

CommandProgram     build_command_program(const CommandPlan & plan);
VerificationResult verify_command_program(const CommandProgram & program);

}  // namespace ggml::hrx
