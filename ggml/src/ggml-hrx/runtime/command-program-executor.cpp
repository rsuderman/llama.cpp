#include "command-program-executor.h"

#include "dispatch/command-program-diagnostics.h"
#include "dispatch/command-program-resolver.h"
#include "ggml-impl.h"
#include "hrx-interop-utils.h"

#include <sstream>

namespace ggml::hrx {
namespace {

static bool execution_context_valid(const CommandProgramExecutionContext & context) {
    if (context.device == nullptr) {
        GGML_LOG_ERROR("%s: missing HRX device\n", __func__);
        return false;
    }
    if (context.stream == nullptr) {
        GGML_LOG_ERROR("%s: missing HRX stream\n", __func__);
        return false;
    }
    if (context.target == nullptr) {
        GGML_LOG_ERROR("%s: missing HRX target\n", __func__);
        return false;
    }
    if (context.corpus == nullptr) {
        GGML_LOG_ERROR("%s: missing HRX kernel corpus\n", __func__);
        return false;
    }
    if (context.jit == nullptr) {
        GGML_LOG_ERROR("%s: missing HRX JIT storage\n", __func__);
        return false;
    }
    if (context.kernel_executables == nullptr) {
        GGML_LOG_ERROR("%s: missing HRX kernel executable cache\n", __func__);
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

static bool dispatch_request(const CommandProgramExecutionContext & context,
                             const Dispatch &                       dispatch,
                             const std::vector<hrx_buffer_ref_t> &  refs) {
    KernelResolveResult resolved =
        resolve_kernel_definition(*context.corpus, context.target, dispatch.kernel.kernel_id);
    if (!resolved.found()) {
        GGML_LOG_ERROR("%s: %s\n", __func__, format_kernel_resolve_error(resolved, dispatch.kernel.kernel_id).c_str());
        return false;
    }
    std::vector<uint8_t> constants;
    auto                 artifact = context.kernel_executables->prepare({ context.device, context.target, context.jit },
                                                                        *resolved.definition, dispatch, constants);
    if (artifact == nullptr) {
        return false;
    }
    if (refs.size() != dispatch.bindings.size()) {
        GGML_LOG_ERROR("%s: dispatch binding refs do not match dispatch bindings\n", __func__);
        return false;
    }
    hrx_dispatch_config_t config = {
        { artifact->launch.workgroup_count[0], artifact->launch.workgroup_count[1],
         artifact->launch.workgroup_count[2]                                                                           },
        { artifact->launch.workgroup_size[0],  artifact->launch.workgroup_size[1],  artifact->launch.workgroup_size[2] },
        artifact->launch.subgroup_size,
    };
    if (ErrorResult error =
            take_status(hrx_stream_dispatch(context.stream, artifact->executable, artifact->export_ordinal, &config,
                                            constants.data(), constants.size(), refs.data(), refs.size(), 0))) {
        GGML_LOG_ERROR("%s: dispatch %s: %s\n", __func__, kernel_definition_name(*resolved.definition).c_str(),
                       error->c_str());
        return false;
    }
    return true;
}

static bool execute_kernel_command(const CommandProgramExecutionContext & context, const ResolvedCommand & command) {
    const std::string command_context = format_resolved_command_context(command);
    if (command.kind != CommandKind::Kernel) {
        GGML_LOG_ERROR("%s: unsupported command kind in %s\n", __func__, command_context.c_str());
        return false;
    }

    Dispatch                      dispatch;
    std::vector<hrx_buffer_ref_t> refs;
    dispatch.kernel = command.kernel;
    dispatch.bindings.reserve(command.bindings.size());
    refs.reserve(command.bindings.size());
    for (const ResolvedCommandBinding & binding : command.bindings) {
        dispatch.bindings.push_back({ binding.binding.value, binding.binding.offset, binding.binding.length });
        refs.push_back({ binding.ref.buffer, binding.ref.offset, binding.ref.length });
    }
    if (!dispatch_request(context, dispatch, refs)) {
        GGML_LOG_ERROR("%s: failed to execute %s\n", __func__, command_context.c_str());
        return false;
    }
    return true;
}

}  // namespace

bool execute_command_program(const CommandProgramExecutionContext & context,
                             const CommandProgram &                 commands,
                             const CommandProgramBindings &         bindings) {
    if (!execution_context_valid(context)) {
        return false;
    }
    const VerificationResult verification = verify_command_program(commands, *context.corpus, context.target);
    if (!verification.valid()) {
        GGML_LOG_ERROR("%s: invalid HRX command program: %s\n", __func__, verification.errors.front().c_str());
        return false;
    }
    if (!bindings.valid()) {
        GGML_LOG_ERROR("%s: invalid HRX command program bindings: %s\n", __func__, bindings.errors.front().c_str());
        return false;
    }
    const ResolvedCommandProgram resolved = resolve_command_program_bindings(commands, bindings);
    if (!resolved.valid()) {
        GGML_LOG_ERROR("%s: failed to resolve HRX command program bindings: %s\n", __func__,
                       resolved.errors.front().c_str());
        return false;
    }
    for (const ResolvedCommand & command : resolved.commands) {
        if (!execute_kernel_command(context, command)) {
            return false;
        }
    }
    return true;
}

}  // namespace ggml::hrx
