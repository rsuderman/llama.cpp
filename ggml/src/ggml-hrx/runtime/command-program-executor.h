#pragma once

#include "dispatch/command-program-bindings.h"
#include "dispatch/command-program-resolver.h"
#include "dispatch/command-program.h"
#include "kernel-corpus/kernel-corpus.h"

#include <cstdint>
#include <memory>
#include <vector>

typedef struct hrx_device_s * hrx_device_t;
typedef struct hrx_stream_s * hrx_stream_t;
struct ggml_hrx_loom_jit_amdgpu;

namespace ggml::hrx {

class KernelExecutableCache;
struct KernelExecutable;

struct CommandProgramExecutionContext {
    hrx_device_t                device             = nullptr;
    hrx_stream_t                stream             = nullptr;
    const char *                target             = nullptr;
    const KernelCorpus *        corpus             = nullptr;
    ggml_hrx_loom_jit_amdgpu ** jit                = nullptr;
    KernelExecutableCache *     kernel_executables = nullptr;
};

struct PreparedCommandBinding {
    CommandBinding    binding;
    ResolvedBufferRef ref;
};

struct PreparedKernelCommand {
    KernelSpecialization                specialization;
    std::shared_ptr<KernelExecutable>   executable;
    std::vector<uint8_t>                constants;
    std::vector<PreparedCommandBinding> bindings;
};

struct PreparedCommand {
    uint32_t              ordinal = 0;
    CommandKind           kind    = CommandKind::Invalid;
    PreparedKernelCommand kernel;
};

struct PreparedCommandProgram {
    std::vector<PreparedCommand> commands;
    ErrorLog                     errors;

    bool valid() const { return errors.success(); }
};

PreparedCommandProgram prepare_command_program(const CommandProgramExecutionContext & context,
                                               const CommandProgram &                 commands,
                                               const CommandProgramBindings &         bindings);

bool execute_prepared_command_program(const CommandProgramExecutionContext & context,
                                      const PreparedCommandProgram &         commands);

bool execute_command_program(const CommandProgramExecutionContext & context,
                             const CommandProgram &                 commands,
                             const CommandProgramBindings &         bindings);

}  // namespace ggml::hrx
