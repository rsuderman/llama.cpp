#pragma once

#include "dispatch/command-program-bindings.h"
#include "dispatch/command-program.h"
#include "hrx_runtime.h"
#include "kernel-corpus/kernel-corpus.h"
#include "runtime/kernel-executable-cache.h"

struct ggml_hrx_loom_jit_amdgpu;

namespace ggml::hrx {

struct CommandProgramExecutionContext {
    hrx_device_t                device             = nullptr;
    hrx_stream_t                stream             = nullptr;
    const char *                target             = nullptr;
    const KernelCorpus *        corpus             = nullptr;
    ggml_hrx_loom_jit_amdgpu ** jit                = nullptr;
    KernelExecutableCache *     kernel_executables = nullptr;
};

bool execute_command_program(const CommandProgramExecutionContext & context,
                             const CommandProgram &                 commands,
                             const CommandProgramBindings &         bindings);

}  // namespace ggml::hrx
