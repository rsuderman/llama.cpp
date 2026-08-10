#pragma once

#include "command-plan.h"
#include "error-log.h"
#include "kernel-corpus/kernel-corpus.h"
#include "kernel-corpus/kernel-types.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ggml::hrx {

enum class CommandKind : uint8_t {
    Invalid,
    Kernel,
};

enum class CommandBindingOrigin : uint8_t {
    GraphValue,
};

struct CommandBinding {
    std::string          name;
    ValueId              value;
    CommandBindingOrigin origin = CommandBindingOrigin::GraphValue;
    size_t               offset = 0;
    size_t               length = 0;
    ResourceAccess       access = ResourceAccess::Read;
};

struct Command {
    uint32_t                    ordinal = 0;
    CommandKind                 kind    = CommandKind::Kernel;
    KernelSpecialization        kernel;
    std::vector<CommandBinding> bindings;
    std::vector<uint32_t>       dependencies;
};

struct CommandProgram {
    std::vector<Command> commands;
    ErrorLog             errors;

    bool valid() const { return errors.success(); }
};

CommandProgram build_command_program(const CommandPlan & plan, const KernelCorpus & corpus, const std::string & target);
VerificationResult verify_command_program(const CommandProgram & program,
                                          const KernelCorpus &   corpus,
                                          const std::string &    target);

}  // namespace ggml::hrx
