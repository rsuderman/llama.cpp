#include "command-program-executor.h"

#include "dispatch/command-program-diagnostics.h"
#include "dispatch/command-program-resolver.h"
#include "ggml-impl.h"
#include "hrx-interop-utils.h"
#include "runtime/kernel-executable-cache.h"
#include "runtime/transient-arena.h"

#include <sstream>
#include <utility>

namespace ggml::hrx {
namespace {

static bool command_program_metadata_context_valid(const CommandProgramExecutionContext & context, ErrorLog & errors) {
    if (context.target == nullptr) {
        errors.log("missing HRX target");
        return false;
    }
    if (context.corpus == nullptr) {
        errors.log("missing HRX kernel corpus");
        return false;
    }
    return true;
}

static bool command_program_preparation_context_valid(const CommandProgramExecutionContext & context,
                                                      ErrorLog &                             errors) {
    if (context.device == nullptr) {
        errors.log("missing HRX device");
        return false;
    }
    if (context.jit == nullptr) {
        errors.log("missing HRX JIT storage");
        return false;
    }
    if (context.kernel_executables == nullptr) {
        errors.log("missing HRX kernel executable cache");
        return false;
    }
    return true;
}

static bool command_program_transient_context_valid(const CommandProgramExecutionContext & context,
                                                    const CommandProgram &                 commands,
                                                    ErrorLog &                             errors) {
    if (commands.transients.arena_size == 0) {
        return true;
    }
    if (context.transient_arena == nullptr) {
        errors.log("missing HRX transient arena");
        return false;
    }
    if (context.stream == nullptr) {
        errors.log("missing HRX stream for transient arena");
        return false;
    }
    return true;
}

static bool prepared_execution_context_valid(const CommandProgramExecutionContext & context) {
    if (context.stream == nullptr) {
        GGML_LOG_ERROR("%s: missing HRX stream\n", __func__);
        return false;
    }
    return true;
}

static bool ensure_transient_arena(const CommandProgramExecutionContext & context,
                                   const CommandProgram &                 commands,
                                   TransientArenaAllocationRef &          allocation,
                                   ErrorLog &                             errors) {
    allocation = {};
    if (!command_program_transient_context_valid(context, commands, errors)) {
        return false;
    }
    if (commands.transients.arena_size == 0) {
        return true;
    }
    if (!context.transient_arena->ensure_capacity(context.device, context.stream, commands.transients.arena_size,
                                                  errors)) {
        return false;
    }
    allocation = context.transient_arena->current_allocation();
    return true;
}

static std::string format_resolved_command_context(const ResolvedCommand & command) {
    std::ostringstream out;
    out << "command " << command.ordinal << " kind=" << command_kind_name(command.kind)
        << " kernel_id=" << command.kernel.kernel_id << " bindings=" << command.bindings.size();
    return out.str();
}

static std::string format_prepared_command_context(const PreparedCommand & command) {
    std::ostringstream out;
    out << "command " << command.ordinal << " kind=" << command_kind_name(command.kind);
    if (command.kind == CommandKind::Kernel) {
        out << " kernel_id=" << command.kernel.specialization.kernel_id
            << " bindings=" << command.kernel.bindings.size();
    }
    return out.str();
}

static Dispatch build_dispatch(const ResolvedCommand & command) {
    Dispatch dispatch;
    dispatch.kernel = command.kernel;
    dispatch.bindings.reserve(command.bindings.size());
    for (const ResolvedCommandBinding & binding : command.bindings) {
        dispatch.bindings.push_back({ binding.binding.value, binding.binding.offset, binding.binding.length });
    }
    return dispatch;
}

static PreparedCommand make_prepared_command_shape(const ResolvedCommand & command) {
    PreparedCommand prepared;
    prepared.ordinal               = command.ordinal;
    prepared.kind                  = command.kind;
    prepared.kernel.specialization = command.kernel;
    prepared.kernel.bindings.reserve(command.bindings.size());
    for (const ResolvedCommandBinding & binding : command.bindings) {
        prepared.kernel.bindings.push_back({
            binding.binding,
            { binding.ref.buffer, binding.ref.offset, binding.ref.length },
        });
    }
    return prepared;
}

static bool prepare_kernel_command(const CommandProgramExecutionContext & context,
                                   const ResolvedCommand &                command,
                                   PreparedCommand &                      prepared,
                                   ErrorLog &                             errors) {
    const std::string command_context = format_resolved_command_context(command);
    if (command.kind != CommandKind::Kernel) {
        errors.log("unsupported command kind in %s", command_context.c_str());
        return false;
    }
    Dispatch dispatch = build_dispatch(command);

    KernelResolveResult resolved =
        resolve_kernel_definition(*context.corpus, context.target, dispatch.kernel.kernel_id);
    if (!resolved.found()) {
        errors.log("%s: %s", command_context.c_str(),
                   format_kernel_resolve_error(resolved, dispatch.kernel.kernel_id).c_str());
        return false;
    }

    prepared                   = make_prepared_command_shape(command);
    prepared.kernel.executable = context.kernel_executables->prepare(
        { context.device, context.target, context.jit }, *resolved.definition, dispatch, prepared.kernel.constants);
    if (prepared.kernel.executable == nullptr) {
        errors.log("failed to prepare %s", command_context.c_str());
        return false;
    }
    return true;
}

static bool execute_prepared_kernel_command(const CommandProgramExecutionContext & context,
                                            const PreparedCommand &                command) {
    const std::string command_context = format_prepared_command_context(command);
    if (command.kind != CommandKind::Kernel) {
        GGML_LOG_ERROR("%s: unsupported command kind in %s\n", __func__, command_context.c_str());
        return false;
    }
    if (command.kernel.executable == nullptr) {
        GGML_LOG_ERROR("%s: missing kernel executable for %s\n", __func__, command_context.c_str());
        return false;
    }

    std::vector<hrx_buffer_ref_t> refs;
    refs.reserve(command.kernel.bindings.size());
    for (const PreparedCommandBinding & binding : command.kernel.bindings) {
        refs.push_back({ binding.ref.buffer, binding.ref.offset, binding.ref.length });
    }

    const KernelExecutable & executable = *command.kernel.executable;
    hrx_dispatch_config_t    config     = {
        { executable.launch.workgroup_count[0], executable.launch.workgroup_count[1],
         executable.launch.workgroup_count[2] },
        { executable.launch.workgroup_size[0],  executable.launch.workgroup_size[1],
         executable.launch.workgroup_size[2]  },
        executable.launch.subgroup_size,
    };
    if (ErrorResult error = take_status(hrx_stream_dispatch(
            context.stream, executable.executable, executable.export_ordinal, &config, command.kernel.constants.data(),
            command.kernel.constants.size(), refs.data(), refs.size(), 0))) {
        GGML_LOG_ERROR("%s: failed to execute %s: %s\n", __func__, command_context.c_str(), error->c_str());
        return false;
    }
    return true;
}

}  // namespace

PreparedCommandProgram prepare_command_program(const CommandProgramExecutionContext & context,
                                               const CommandProgram &                 commands,
                                               const CommandProgramBindings &         bindings) {
    PreparedCommandProgram prepared;
    if (!command_program_metadata_context_valid(context, prepared.errors)) {
        return prepared;
    }

    const VerificationResult verification = verify_command_program(commands, *context.corpus, context.target);
    if (!verification.valid()) {
        prepared.errors.append(verification.errors);
        return prepared;
    }
    if (!bindings.valid()) {
        prepared.errors.append(bindings.errors);
        return prepared;
    }

    TransientArenaAllocationRef transient_allocation;
    if (!ensure_transient_arena(context, commands, transient_allocation, prepared.errors)) {
        return prepared;
    }

    const TransientArenaAllocationRef * transient_allocation_ptr =
        commands.transients.arena_size == 0 ? nullptr : &transient_allocation;
    const ResolvedCommandProgram resolved =
        resolve_command_program_bindings(commands, bindings, transient_allocation_ptr);
    if (!resolved.valid()) {
        prepared.errors.append(resolved.errors);
        return prepared;
    }
    if (!command_program_preparation_context_valid(context, prepared.errors)) {
        return prepared;
    }

    prepared.commands.reserve(resolved.commands.size());
    for (const ResolvedCommand & command : resolved.commands) {
        PreparedCommand prepared_command;
        if (prepare_kernel_command(context, command, prepared_command, prepared.errors)) {
            prepared.commands.push_back(std::move(prepared_command));
        }
    }
    prepared.bound_transient_arena_allocation_id = transient_allocation.allocation_id;
    return prepared;
}

bool bind_prepared_command_program_transients(const CommandProgram &              commands,
                                              const TransientArenaAllocationRef & transient_allocation,
                                              PreparedCommandProgram &            prepared) {
    if (!prepared.valid()) {
        return false;
    }
    if (commands.transients.arena_size == 0) {
        prepared.bound_transient_arena_allocation_id = kInvalidTransientArenaAllocationId;
        return true;
    }
    if (transient_allocation.buffer == nullptr ||
        transient_allocation.allocation_id == kInvalidTransientArenaAllocationId) {
        GGML_LOG_ERROR("%s: missing transient arena allocation\n", __func__);
        return false;
    }
    if (prepared.bound_transient_arena_allocation_id == transient_allocation.allocation_id) {
        return true;
    }
    for (PreparedCommand & command : prepared.commands) {
        for (PreparedCommandBinding & binding : command.kernel.bindings) {
            if (binding.binding.origin != CommandBindingOrigin::Transient) {
                continue;
            }
            const TransientAllocation * allocation =
                find_transient_allocation(commands.transients, binding.binding.value);
            if (allocation == nullptr) {
                GGML_LOG_ERROR("%s: %s has no transient allocation\n", __func__,
                               format_command_binding(binding.binding).c_str());
                return false;
            }
            if (binding.binding.offset > allocation->size ||
                binding.binding.length > allocation->size - binding.binding.offset ||
                commands.transients.arena_size > transient_allocation.capacity) {
                GGML_LOG_ERROR("%s: %s is outside transient arena\n", __func__,
                               format_command_binding(binding.binding).c_str());
                return false;
            }
            binding.ref = {
                transient_allocation.buffer,
                allocation->arena_offset + binding.binding.offset,
                binding.binding.length,
            };
        }
    }
    prepared.bound_transient_arena_allocation_id = transient_allocation.allocation_id;
    return true;
}

bool bind_and_execute_prepared_command_program(const CommandProgramExecutionContext & context,
                                               const CommandProgram &                 commands,
                                               PreparedCommandProgram &               prepared) {
    if (!prepared.valid()) {
        return execute_prepared_command_program(context, prepared);
    }
    if (commands.transients.arena_size == 0) {
        return bind_prepared_command_program_transients(commands, {}, prepared) &&
               execute_prepared_command_program(context, prepared);
    }

    ErrorLog errors;
    if (!command_program_transient_context_valid(context, commands, errors)) {
        GGML_LOG_ERROR("%s: %s\n", __func__, errors.front().c_str());
        return false;
    }

    TransientArena::AllocationLease lease = context.transient_arena->acquire_allocation_lease();
    if (!lease.ensure_capacity(context.device, context.stream, commands.transients.arena_size, errors)) {
        GGML_LOG_ERROR("%s: %s\n", __func__, errors.front().c_str());
        return false;
    }
    return bind_prepared_command_program_transients(commands, lease.current_allocation(), prepared) &&
           execute_prepared_command_program(context, prepared);
}

bool execute_prepared_command_program(const CommandProgramExecutionContext & context,
                                      const PreparedCommandProgram &         commands) {
    if (!commands.valid()) {
        GGML_LOG_ERROR("%s: invalid HRX prepared command program: %s\n", __func__, commands.errors.front().c_str());
        return false;
    }
    if (!prepared_execution_context_valid(context)) {
        return false;
    }
    for (const PreparedCommand & command : commands.commands) {
        if (!execute_prepared_kernel_command(context, command)) {
            return false;
        }
    }
    return true;
}

bool execute_command_program(const CommandProgramExecutionContext & context,
                             const CommandProgram &                 commands,
                             const CommandProgramBindings &         bindings) {
    PreparedCommandProgram prepared = prepare_command_program(context, commands, bindings);
    return bind_and_execute_prepared_command_program(context, commands, prepared);
}

}  // namespace ggml::hrx
