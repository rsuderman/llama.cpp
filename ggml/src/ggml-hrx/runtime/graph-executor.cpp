#include "graph-executor.h"

#include "backend-buffer-binding.h"
#include "ggml-impl.h"
#include "runtime/kernel-executable-cache.h"
#include "runtime/prepared-command-program-cache.h"
#include "runtime/transient-arena.h"

#include <utility>
#include <vector>

namespace ggml::hrx {

GraphExecutor::GraphExecutor(ggml_backend_hrx_context & context) : context_(context) {}

bool GraphExecutor::context_valid_for_graph_programs(ErrorLog & errors) const {
    if (context_.device == nullptr) {
        errors.log("missing HRX device context");
    } else if (context_.device->architecture.empty()) {
        errors.log("missing HRX target");
    }
    return errors.success();
}

bool GraphExecutor::context_valid_for_execution(ErrorLog & errors) const {
    context_valid_for_graph_programs(errors);
    return errors.success();
}

GraphSupportResult GraphExecutor::can_execute(const ggml_cgraph & graph) const {
    GraphSupportResult result;
    if (graph.n_nodes == 0) {
        result.supported = true;
        return result;
    }
    if (!context_valid_for_graph_programs(result.errors)) {
        return result;
    }
    const KernelCorpus &            corpus = get_qwen_kernel_corpus();
    const GraphProgramSupportResult support =
        context_.graph_programs.check_support(graph, corpus, context_.device->architecture);
    result.supported = support.supported;
    result.errors.append(support.errors);
    return result;
}

CommandProgramBindings GraphExecutor::bind_external_value_buffers(const GraphProgramMatch & match) const {
    std::vector<CommandProgramBinding> bindings;
    ErrorLog                           errors;
    bindings.reserve(match.external_bindings.size());
    for (const GraphProgramExternalBinding & external : match.external_bindings) {
        ValueBufferBinding    value_binding;
        CommandProgramBinding binding;
        binding.value = external.value;
        if (ggml_backend_hrx_resolve_value_buffer(external.tensor, value_binding)) {
            binding.buffer     = value_binding.buffer;
            binding.offset     = value_binding.offset;
            binding.length     = value_binding.length;
            binding.identity   = value_binding.identity;
            binding.generation = value_binding.generation;
            binding.capacity   = value_binding.capacity;
        } else {
            errors.log("external value %d is not bound", external.value.value);
        }
        bindings.push_back(binding);
    }
    return CommandProgramBindings::from_bindings(std::move(bindings), errors);
}

GraphExecutionResult GraphExecutor::execute(const ggml_cgraph & graph) const {
    GraphExecutionResult result;
    if (graph.n_nodes == 0) {
        result.status = GGML_STATUS_SUCCESS;
        return result;
    }
    if (!context_valid_for_execution(result.errors)) {
        return result;
    }

    const KernelCorpus & corpus = get_qwen_kernel_corpus();
    GraphProgramLookup   lookup = context_.graph_programs.get_or_build(graph, corpus, context_.device->architecture);
    if (!lookup.valid()) {
        result.errors.log("build HRX graph program failed");
        result.errors.append(lookup.errors);
        result.errors.append(lookup.match.errors);
        return result;
    }

    CommandProgramBindings bindings = bind_external_value_buffers(lookup.match);
    if (!bindings.valid()) {
        result.errors.append(bindings.errors);
        return result;
    }
    const CommandProgramExecutionContext execution_context = {
        context_.device->device,
        context_.stream,
        context_.device->architecture.c_str(),
        &corpus,
        &context_.jit,
        &context_.kernel_executables,
        &context_.transient_arena,
    };
    const PreparedCommandProgramCacheExecutionResult execution = context_.prepared_programs.execute_with_result(
        execution_context, lookup.program->uid(), lookup.program->command_shape(), lookup.program->commands(),
        bindings);
    if (!execution.success) {
        result.errors.append(execution.errors);
        if (result.errors.success()) {
            result.errors.log("execute HRX command program failed");
        }
        return result;
    }

    result.status = GGML_STATUS_SUCCESS;
    return result;
}

}  // namespace ggml::hrx
