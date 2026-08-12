#include "command-program-executor.h"

#include "dispatch/command-program-diagnostics.h"
#include "dispatch/command-program-resolver.h"
#include "ggml-impl.h"
#include "hrx-interop-utils.h"
#include "runtime/kernel-executable-cache.h"
#include "runtime/transient-arena.h"

#include <sstream>
#include <unordered_map>
#include <utility>

namespace ggml::hrx {
namespace {

static const char * status_first_error(const Status & status) {
    return status.errors().empty() ? "" : status.errors().front().c_str();
}

static Status command_program_metadata_context_valid(const CommandProgramExecutionContext & context) {
    Status status;
    if (context.target == nullptr) {
        status.log("missing HRX target");
        return status;
    }
    if (context.corpus == nullptr) {
        status.log("missing HRX kernel corpus");
        return status;
    }
    return status;
}

static Status command_program_preparation_context_valid(const CommandProgramExecutionContext & context) {
    Status status;
    if (context.device == nullptr) {
        status.log("missing HRX device");
        return status;
    }
    if (context.jit == nullptr) {
        status.log("missing HRX JIT storage");
        return status;
    }
    if (context.kernel_executables == nullptr) {
        status.log("missing HRX kernel executable cache");
        return status;
    }
    if (context.host_transfers == nullptr) {
        status.log("missing HRX host transfer manager");
        return status;
    }
    if (context.host_weights == nullptr) {
        status.log("missing HRX host weight cache");
        return status;
    }
    return status;
}

static Status command_program_transient_context_valid(const CommandProgramExecutionContext & context,
                                                      const CommandProgram &                 commands) {
    Status status;
    if (commands.transients.arena_size == 0) {
        return status;
    }
    if (context.transient_arena == nullptr) {
        status.log("missing HRX transient arena");
        return status;
    }
    if (context.stream == nullptr) {
        status.log("missing HRX stream for transient arena");
        return status;
    }
    return status;
}

static bool prepared_execution_context_valid(const CommandProgramExecutionContext & context) {
    if (context.stream == nullptr) {
        GGML_LOG_ERROR("%s: missing HRX stream\n", __func__);
        return false;
    }
    return true;
}

static Status ensure_transient_arena(const CommandProgramExecutionContext & context,
                                     const CommandProgram &                 commands,
                                     TransientArenaAllocationRef &          allocation) {
    allocation    = {};
    Status status = command_program_transient_context_valid(context, commands);
    if (!status.success()) {
        return status;
    }
    if (commands.transients.arena_size == 0) {
        return status;
    }
    status = context.transient_arena->ensure_capacity(context.device, context.stream, commands.transients.arena_size);
    if (!status.success()) {
        return status;
    }
    allocation = context.transient_arena->current_allocation();
    return status;
}

static Status initialize_command_program_constants(const CommandProgramExecutionContext & context,
                                                   const CommandProgram &                 commands,
                                                   const TransientArenaAllocationRef &    allocation) {
    Status status;
    if (commands.constant_initializations.empty()) {
        return status;
    }
    if (allocation.buffer == nullptr) {
        status.log("command program has constant initializations without a transient arena allocation");
        return status;
    }
    for (const ConstantInitialization & initialization : commands.constant_initializations) {
        const TransientAllocation * transient = find_transient_allocation(commands.transients, initialization.value);
        if (transient == nullptr) {
            status.log("constant initialization %s references missing transient value %d", initialization.name.c_str(),
                       initialization.value.value);
            continue;
        }
        if (initialization.offset > transient->size ||
            initialization.data.size() > transient->size - initialization.offset) {
            status.log("constant initialization %s is outside transient allocation length %zu",
                       initialization.name.c_str(), transient->size);
            continue;
        }
        // TODO: Track initialized transient arena allocation ids so constants are not transferred every invocation.
        if (ErrorResult error = take_status(
                hrx_stream_copy_h2d(context.stream, initialization.data.data(), allocation.buffer,
                                    transient->arena_offset + initialization.offset, initialization.data.size()))) {
            status.log("failed to upload constant initialization %s: %s", initialization.name.c_str(), error->c_str());
        }
    }
    return status;
}

static Status initialize_command_program_completion_counters(const CommandProgramExecutionContext & context,
                                                             const CommandProgram &                 commands,
                                                             const TransientArenaAllocationRef &    allocation) {
    Status status;
    if (commands.completion_counters.byte_count == 0) {
        return status;
    }
    if (allocation.buffer == nullptr) {
        status.log("command program has completion counters without a transient arena allocation");
        return status;
    }
    if (commands.completion_counters.arena_offset > commands.transients.arena_size ||
        commands.completion_counters.byte_count >
            commands.transients.arena_size - commands.completion_counters.arena_offset) {
        status.log("completion counter initialization is outside transient arena length %zu",
                   commands.transients.arena_size);
        return status;
    }
    const uint32_t zero_pattern = 0;
    if (ErrorResult error = take_status(
            hrx_stream_fill_buffer(context.stream, allocation.buffer, commands.completion_counters.arena_offset,
                                   commands.completion_counters.byte_count, &zero_pattern, sizeof(zero_pattern)))) {
        status.log("failed to initialize completion counters: %s", error->c_str());
    }
    return status;
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

struct GraphValueAccess {
    bool read  = false;
    bool write = false;
};

static std::unordered_map<int32_t, GraphValueAccess> collect_graph_value_access(const CommandProgram & commands) {
    std::unordered_map<int32_t, GraphValueAccess> access_by_value;
    for (const Command & command : commands.commands) {
        for (const CommandBinding & binding : command.bindings) {
            if (binding.origin != CommandBindingOrigin::GraphValue) {
                continue;
            }
            GraphValueAccess & access = access_by_value[binding.value.value];
            switch (binding.access) {
                case ResourceAccess::Read:
                    access.read = true;
                    break;
                case ResourceAccess::Write:
                    access.write = true;
                    break;
                case ResourceAccess::ReadWrite:
                    access.read  = true;
                    access.write = true;
                    break;
            }
        }
    }
    return access_by_value;
}

static CommandProgramBindings materialize_host_bindings(const CommandProgramExecutionContext & context,
                                                        const CommandProgram &                 commands,
                                                        const CommandProgramBindings &         bindings,
                                                        PreparedCommandProgram &               prepared) {
    std::vector<CommandProgramBinding> materialized;
    Status                             status;
    materialized.reserve(bindings.bindings().size());
    const std::unordered_map<int32_t, GraphValueAccess> access_by_value = collect_graph_value_access(commands);
    for (const CommandProgramBinding & binding : bindings.bindings()) {
        if (binding.host_data == nullptr) {
            materialized.push_back(binding);
            continue;
        }
        const auto             found_access = access_by_value.find(binding.value.value);
        const GraphValueAccess access =
            found_access != access_by_value.end() ? found_access->second : GraphValueAccess{};
        if (binding.weight && access.read && !access.write) {
            HostWeightSource source;
            source.host_data  = binding.host_data;
            source.identity   = binding.identity;
            source.generation = binding.generation;
            source.capacity   = binding.capacity;
            source.offset     = binding.offset;
            source.length     = binding.length;
            HostWeightAcquireResult resident =
                context.host_weights->acquire(context.device, context.stream, *context.host_transfers, source);
            if (!resident.valid()) {
                status.log("materialize host weight value %d failed", binding.value.value);
                status.append(resident.status);
                materialized.push_back(binding);
                continue;
            }
            CommandProgramBinding device_binding = binding;
            device_binding.buffer                = resident.lease.buffer();
            device_binding.host_data             = nullptr;
            device_binding.offset                = 0;
            device_binding.capacity              = binding.length;
            materialized.push_back(device_binding);
            prepared.resident_host_weights.push_back(std::move(resident.lease));
            continue;
        }

        HostStagingBuffer staging;
        Status            allocation_status = allocate_host_staging_buffer(context.device, binding.length, staging);
        if (!allocation_status.success()) {
            status.log("allocate host staging for value %d failed", binding.value.value);
            status.append(allocation_status);
            materialized.push_back(binding);
            continue;
        }
        staging.value                        = binding.value.value;
        staging.host_data                    = static_cast<uint8_t *>(binding.host_data) + binding.offset;
        staging.upload                       = access.read;
        staging.download                     = access.write;
        CommandProgramBinding device_binding = binding;
        device_binding.buffer                = staging.buffer;
        device_binding.host_data             = nullptr;
        device_binding.offset                = 0;
        device_binding.capacity              = binding.length;
        materialized.push_back(device_binding);
        prepared.host_staging.push_back(std::move(staging));
    }
    return CommandProgramBindings::from_bindings(std::move(materialized), status);
}

static Status rebind_prepared_host_staging(const CommandProgramBindings & bindings, PreparedCommandProgram & prepared) {
    Status status;
    for (HostStagingBuffer & staging : prepared.host_staging) {
        const CommandProgramBinding * binding = bindings.find(ValueId(staging.value));
        if (binding == nullptr || binding->host_data == nullptr || binding->length != staging.length ||
            binding->offset > binding->capacity || binding->length > binding->capacity - binding->offset) {
            status.log("live host binding does not match prepared value %d", staging.value);
            continue;
        }
        staging.host_data = static_cast<uint8_t *>(binding->host_data) + binding->offset;
    }
    return status;
}

static Status upload_prepared_host_staging(const CommandProgramExecutionContext & context,
                                           const PreparedCommandProgram &         prepared) {
    Status status;
    if (prepared.host_staging.empty()) {
        return status;
    }
    if (context.host_transfers == nullptr) {
        status.log("missing HRX host transfer manager");
        return status;
    }
    for (const HostStagingBuffer & staging : prepared.host_staging) {
        if (!staging.upload) {
            continue;
        }
        Status upload_status =
            context.host_transfers->upload(context.stream, staging.host_data, staging.buffer, 0, staging.length);
        status.append(upload_status);
    }
    return status;
}

static Status download_prepared_host_staging(const CommandProgramExecutionContext & context,
                                             const PreparedCommandProgram &         prepared) {
    Status status;
    if (prepared.host_staging.empty()) {
        return status;
    }
    if (context.host_transfers == nullptr) {
        status.log("missing HRX host transfer manager");
        return status;
    }
    for (const HostStagingBuffer & staging : prepared.host_staging) {
        if (!staging.download) {
            continue;
        }
        Status download_status =
            context.host_transfers->download(context.stream, staging.buffer, 0, staging.host_data, staging.length);
        status.append(download_status);
    }
    return status;
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

static Status prepare_kernel_command(const CommandProgramExecutionContext & context,
                                     const ResolvedCommand &                command,
                                     PreparedCommand &                      prepared) {
    Status            status;
    const std::string command_context = format_resolved_command_context(command);
    if (command.kind != CommandKind::Kernel) {
        status.log("unsupported command kind in %s", command_context.c_str());
        return status;
    }
    Dispatch dispatch = build_dispatch(command);

    KernelResolveResult resolved =
        resolve_kernel_definition(*context.corpus, context.target, dispatch.kernel.kernel_id);
    if (!resolved.found()) {
        status.log("%s: %s", command_context.c_str(),
                   format_kernel_resolve_error(resolved, dispatch.kernel.kernel_id).c_str());
        return status;
    }

    prepared                   = make_prepared_command_shape(command);
    prepared.kernel.executable = context.kernel_executables->prepare(
        { context.device, context.target, context.jit }, *resolved.definition, dispatch, prepared.kernel.constants);
    if (prepared.kernel.executable == nullptr) {
        status.log("failed to prepare %s", command_context.c_str());
        return status;
    }
    return status;
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
    prepared.status = command_program_metadata_context_valid(context);
    if (!prepared.status.success()) {
        return prepared;
    }

    const VerificationResult verification = verify_command_program(commands, *context.corpus, context.target);
    if (!verification.valid()) {
        prepared.status.append(verification.status);
        return prepared;
    }
    if (!bindings.valid()) {
        prepared.status.append(bindings.status);
        return prepared;
    }

    TransientArenaAllocationRef transient_allocation;
    prepared.status = ensure_transient_arena(context, commands, transient_allocation);
    if (!prepared.status.success()) {
        return prepared;
    }

    prepared.status = command_program_preparation_context_valid(context);
    if (!prepared.status.success()) {
        return prepared;
    }

    const CommandProgramBindings materialized_bindings =
        materialize_host_bindings(context, commands, bindings, prepared);
    if (!materialized_bindings.valid()) {
        prepared.status.append(materialized_bindings.status);
        return prepared;
    }

    const TransientArenaAllocationRef * transient_allocation_ptr =
        commands.transients.arena_size == 0 ? nullptr : &transient_allocation;
    const ResolvedCommandProgram resolved =
        resolve_command_program_bindings(commands, materialized_bindings, transient_allocation_ptr);
    if (!resolved.valid()) {
        prepared.status.append(resolved.status);
        return prepared;
    }

    prepared.commands.reserve(resolved.commands.size());
    for (const ResolvedCommand & command : resolved.commands) {
        PreparedCommand prepared_command;
        Status          status = prepare_kernel_command(context, command, prepared_command);
        if (status.success()) {
            prepared.commands.push_back(std::move(prepared_command));
        } else {
            prepared.status.append(status);
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
                                               const CommandProgramBindings &         bindings,
                                               PreparedCommandProgram &               prepared) {
    if (!prepared.valid()) {
        return execute_prepared_command_program(context, prepared);
    }
    Status rebind_status = rebind_prepared_host_staging(bindings, prepared);
    if (!rebind_status.success()) {
        GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(rebind_status));
        return false;
    }
    if (commands.transients.arena_size == 0) {
        Status status = initialize_command_program_constants(context, commands, {});
        if (!status.success()) {
            GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(status));
            return false;
        }
        status = initialize_command_program_completion_counters(context, commands, {});
        if (!status.success()) {
            GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(status));
            return false;
        }
        return bind_prepared_command_program_transients(commands, {}, prepared) &&
               execute_prepared_command_program(context, prepared);
    }

    Status status = command_program_transient_context_valid(context, commands);
    if (!status.success()) {
        GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(status));
        return false;
    }

    TransientArena::AllocationLease lease = context.transient_arena->acquire_allocation_lease();
    status = lease.ensure_capacity(context.device, context.stream, commands.transients.arena_size);
    if (!status.success()) {
        GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(status));
        return false;
    }
    if (!bind_prepared_command_program_transients(commands, lease.current_allocation(), prepared)) {
        return false;
    }
    status = initialize_command_program_constants(context, commands, lease.current_allocation());
    if (!status.success()) {
        GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(status));
        return false;
    }
    status = initialize_command_program_completion_counters(context, commands, lease.current_allocation());
    if (!status.success()) {
        GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(status));
        return false;
    }
    return execute_prepared_command_program(context, prepared);
}

bool execute_prepared_command_program(const CommandProgramExecutionContext & context,
                                      const PreparedCommandProgram &         commands) {
    if (!commands.valid()) {
        GGML_LOG_ERROR("%s: invalid HRX prepared command program: %s\n", __func__, status_first_error(commands.status));
        return false;
    }
    if (!prepared_execution_context_valid(context)) {
        return false;
    }
    Status upload_status = upload_prepared_host_staging(context, commands);
    if (!upload_status.success()) {
        GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(upload_status));
        return false;
    }
    for (const PreparedCommand & command : commands.commands) {
        if (!execute_prepared_kernel_command(context, command)) {
            return false;
        }
    }
    Status download_status = download_prepared_host_staging(context, commands);
    if (!download_status.success()) {
        GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(download_status));
        return false;
    }
    return true;
}

bool execute_command_program(const CommandProgramExecutionContext & context,
                             const CommandProgram &                 commands,
                             const CommandProgramBindings &         bindings) {
    PreparedCommandProgram prepared = prepare_command_program(context, commands, bindings);
    return bind_and_execute_prepared_command_program(context, commands, bindings, prepared);
}

}  // namespace ggml::hrx
