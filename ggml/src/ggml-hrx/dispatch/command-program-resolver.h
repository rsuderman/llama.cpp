#pragma once

#include "command-program-bindings.h"
#include "command-program.h"
#include "error-log.h"

#include <cstddef>
#include <vector>

namespace ggml::hrx {

struct ResolvedBufferRef {
    hrx_buffer_t buffer = nullptr;
    size_t       offset = 0;
    size_t       length = 0;
};

struct ResolvedCommandBinding {
    CommandBinding    binding;
    ResolvedBufferRef ref;
};

struct ResolvedCommand {
    uint32_t                            ordinal = 0;
    CommandKind                         kind    = CommandKind::Kernel;
    KernelSpecialization                kernel;
    std::vector<ResolvedCommandBinding> bindings;
};

struct ResolvedCommandProgram {
    std::vector<ResolvedCommand> commands;
    ErrorLog                     errors;

    bool valid() const { return errors.success(); }
};

ResolvedCommandProgram resolve_command_program_bindings(const CommandProgram &         program,
                                                        const CommandProgramBindings & bindings);

}  // namespace ggml::hrx
