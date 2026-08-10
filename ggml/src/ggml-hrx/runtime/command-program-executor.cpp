#include "command-program-executor.h"

#include "dispatch/command-program-diagnostics.h"
#include "dispatch/command-program-resolver.h"
#include "ggml-impl.h"
#include "hrx-interop-utils.h"
#include "runtime/kernel-executable-cache.h"

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

static bool prepared_execution_context_valid(const CommandProgramExecutionContext & context) {
    if (context.stream == nullptr) {
        GGML_LOG_ERROR("%s: missing HRX stream\n", __func__);
        return false;
    }
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

    const ResolvedCommandProgram resolved = resolve_command_program_bindings(commands, bindings);
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
    return prepared;
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
    const PreparedCommandProgram prepared = prepare_command_program(context, commands, bindings);
    return execute_prepared_command_program(context, prepared);
}

}  // namespace ggml::hrx
