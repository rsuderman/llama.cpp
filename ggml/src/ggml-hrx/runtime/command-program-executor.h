#pragma once

#include "dispatch/command-program-bindings.h"
#include "dispatch/command-program-resolver.h"
#include "dispatch/command-program.h"
#include "kernel-corpus/kernel-corpus.h"
#include "runtime/host-memory.h"

#include <cstdint>
#include <memory>
#include <vector>

typedef struct hrx_device_s * hrx_device_t;
typedef struct hrx_stream_s * hrx_stream_t;
struct ggml_hrx_loom_jit_amdgpu;

namespace ggml::hrx {

class KernelExecutableCache;
struct KernelExecutable;
class TransientArena;

struct CommandProgramExecutionContext {
    hrx_device_t                device             = nullptr;
    hrx_stream_t                stream             = nullptr;
    const char *                target             = nullptr;
    const KernelCorpus *        corpus             = nullptr;
    ggml_hrx_loom_jit_amdgpu ** jit                = nullptr;
    KernelExecutableCache *     kernel_executables = nullptr;
    TransientArena *            transient_arena    = nullptr;
    HostTransferManager *       host_transfers     = nullptr;
    HostWeightCache *           host_weights       = nullptr;
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
    std::vector<PreparedCommand>   initialization_commands;
    std::vector<PreparedCommand>   commands;
    std::vector<HostStagingBuffer> host_staging;
    std::vector<HostWeightLease>   resident_host_weights;
    Status                         status;
    uint64_t                       bound_transient_arena_allocation_id = kInvalidTransientArenaAllocationId;

    bool valid() const { return status.success(); }
};

PreparedCommandProgram prepare_command_program(const CommandProgramExecutionContext & context,
                                               const CommandProgram &                 commands,
                                               const CommandProgramBindings &         bindings);

bool execute_prepared_command_program(const CommandProgramExecutionContext & context,
                                      const PreparedCommandProgram &         commands);

bool bind_prepared_command_program_transients(const CommandProgram &              commands,
                                              const TransientArenaAllocationRef & transient_allocation,
                                              PreparedCommandProgram &            prepared);

bool bind_and_execute_prepared_command_program(const CommandProgramExecutionContext & context,
                                               const CommandProgram &                 commands,
                                               const CommandProgramBindings &         bindings,
                                               PreparedCommandProgram &               prepared);

bool execute_command_program(const CommandProgramExecutionContext & context,
                             const CommandProgram &                 commands,
                             const CommandProgramBindings &         bindings);

}  // namespace ggml::hrx
