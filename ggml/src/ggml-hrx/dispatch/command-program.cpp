#include "command-program.h"

#include "command-program-diagnostics.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

static bool string_equal(const char * lhs, const char * rhs) {
    return std::strcmp(lhs != nullptr ? lhs : "", rhs != nullptr ? rhs : "") == 0;
}

static std::string string_value(const char * value) {
    return value != nullptr ? value : "";
}

static size_t align_up(size_t value, size_t alignment) {
    return alignment == 0 ? value : (value + alignment - 1) / alignment * alignment;
}

static CommandBindingOrigin command_binding_origin(ValueKind kind) {
    switch (kind) {
        case ValueKind::External:
            return CommandBindingOrigin::GraphValue;
        case ValueKind::Transient:
            return CommandBindingOrigin::Transient;
    }
    return CommandBindingOrigin::GraphValue;
}

struct TransientAllocationRequest {
    ValueId value;
    size_t  required_size = 0;
};

struct TransientBindingTarget {
    ValueId value;
    size_t  offset = 0;
};

static TransientBindingTarget transient_binding_target(const Graph & graph, ValueId value) {
    TransientBindingTarget target;
    target.value              = value;
    const Value * graph_value = graph.values().find(value);
    if (graph_value == nullptr || graph_value->kind != ValueKind::Transient) {
        return target;
    }
    const Value * root = graph.values().find(graph_value->storage_root);
    if (root == nullptr || root->kind != ValueKind::Transient) {
        return target;
    }
    target.value  = root->id;
    target.offset = graph_value->storage_offset;
    return target;
}

static const CommandPlanTransient * find_plan_transient(const CommandPlan & plan, ValueId value) {
    const auto found = std::find_if(plan.transients.begin(), plan.transients.end(),
                                    [&](const CommandPlanTransient & transient) { return transient.value == value; });
    return found == plan.transients.end() ? nullptr : &*found;
}

static const CommandPlanCompletionCounterRequest * find_plan_completion_counter_request(const CommandPlan & plan,
                                                                                        ValueId             value) {
    const auto found =
        std::find_if(plan.completion_counter_requests.begin(), plan.completion_counter_requests.end(),
                     [&](const CommandPlanCompletionCounterRequest & request) { return request.value == value; });
    return found == plan.completion_counter_requests.end() ? nullptr : &*found;
}

static void add_transient_allocation_request(std::vector<TransientAllocationRequest> & requests,
                                             ValueId                                   value,
                                             size_t                                    required_size) {
    for (TransientAllocationRequest & request : requests) {
        if (request.value == value) {
            request.required_size = std::max(request.required_size, required_size);
            return;
        }
    }
    requests.push_back({ value, required_size });
}

static const TransientAllocationRequest * find_transient_allocation_request(
    const std::vector<TransientAllocationRequest> & requests,
    ValueId                                         value) {
    const auto found = std::find_if(requests.begin(), requests.end(),
                                    [&](const TransientAllocationRequest & request) { return request.value == value; });
    return found == requests.end() ? nullptr : &*found;
}

static void add_transient_allocation(const Graph &                      graph,
                                     const CommandPlan &                command_plan,
                                     const TransientAllocationRequest & request,
                                     TransientPlan &                    plan,
                                     Status &                           errors) {
    const Value *                graph_value    = graph.values().find(request.value);
    const CommandPlanTransient * plan_transient = find_plan_transient(command_plan, request.value);
    if (graph_value == nullptr && plan_transient == nullptr) {
        errors.log("transient value %d is missing from graph values and command plan transients", request.value.value);
        return;
    }
    if (graph_value != nullptr && graph_value->kind != ValueKind::Transient) {
        errors.log("transient value %d aliases a non-transient graph value", request.value.value);
        return;
    }

    TransientAllocation allocation;
    allocation.value = request.value;
    allocation.size =
        std::max(graph_value != nullptr ? graph_value->byte_count : plan_transient->size, request.required_size);
    allocation.alignment    = plan_transient != nullptr ? plan_transient->alignment : 256;
    allocation.arena_offset = align_up(plan.arena_size, allocation.alignment);
    plan.arena_size         = allocation.arena_offset + allocation.size;
    plan.allocations.push_back(allocation);
}

static void add_completion_counter_allocations(const CommandPlan &                             command_plan,
                                               const std::vector<TransientAllocationRequest> & binding_requests,
                                               TransientPlan &                                 plan,
                                               CompletionCounterPlan &                         completion_counters,
                                               Status &                                        errors) {
    for (size_t i = 0; i < command_plan.completion_counter_requests.size(); ++i) {
        const CommandPlanCompletionCounterRequest & request = command_plan.completion_counter_requests[i];
        if (request.value.value < 0) {
            errors.log("completion counter request %s has invalid value %d", request.name.c_str(), request.value.value);
            continue;
        }
        if (request.count == 0) {
            errors.log("completion counter request %s has zero counters", request.name.c_str());
            continue;
        }
        for (size_t j = i + 1; j < command_plan.completion_counter_requests.size(); ++j) {
            if (request.value == command_plan.completion_counter_requests[j].value) {
                errors.log("duplicate completion counter request value %d", request.value.value);
            }
        }
        const size_t                       byte_count = static_cast<size_t>(request.count) * sizeof(int32_t);
        const TransientAllocationRequest * binding_request =
            find_transient_allocation_request(binding_requests, request.value);
        if (binding_request != nullptr && binding_request->required_size > byte_count) {
            errors.log("completion counter request %s requires %zu bytes but binding uses %zu bytes",
                       request.name.c_str(), byte_count, binding_request->required_size);
            continue;
        }
        if (completion_counters.count > std::numeric_limits<uint32_t>::max() - request.count) {
            errors.log("completion counter count overflows");
            continue;
        }

        TransientAllocation allocation;
        allocation.value        = request.value;
        allocation.size         = byte_count;
        allocation.alignment    = 16;
        allocation.arena_offset = align_up(plan.arena_size, allocation.alignment);
        if (completion_counters.byte_count == 0) {
            completion_counters.arena_offset = allocation.arena_offset;
        }
        plan.arena_size = allocation.arena_offset + allocation.size;
        completion_counters.byte_count =
            plan.arena_size > completion_counters.arena_offset ? plan.arena_size - completion_counters.arena_offset : 0;
        completion_counters.count += request.count;
        plan.allocations.push_back(allocation);
    }
}

static TransientPlan build_transient_plan(const Graph &                graph,
                                          const CommandPlan &          command_plan,
                                          const std::vector<Command> & initialization_commands,
                                          const std::vector<Command> & commands,
                                          CompletionCounterPlan &      completion_counters,
                                          Status &                     errors) {
    TransientPlan plan;
    plan.arena_alignment = 256;
    std::vector<TransientAllocationRequest> transient_requests;
    std::vector<TransientAllocationRequest> completion_counter_binding_requests;
    auto                                    append_command_bindings = [&](const std::vector<Command> & command_list) {
        for (const Command & command : command_list) {
            for (const CommandBinding & binding : command.bindings) {
                if (binding.origin != CommandBindingOrigin::Transient) {
                    continue;
                }
                if (binding.offset > std::numeric_limits<size_t>::max() - binding.length) {
                    errors.log("transient value %d binding range overflows", binding.value.value);
                    continue;
                }
                if (find_plan_completion_counter_request(command_plan, binding.value) != nullptr) {
                    add_transient_allocation_request(completion_counter_binding_requests, binding.value,
                                                                                        binding.offset + binding.length);
                } else {
                    add_transient_allocation_request(transient_requests, binding.value,
                                                                                        binding.offset + binding.length);
                }
            }
        }
    };
    append_command_bindings(initialization_commands);
    append_command_bindings(commands);
    add_completion_counter_allocations(command_plan, completion_counter_binding_requests, plan, completion_counters,
                                       errors);
    for (const TransientAllocationRequest & request : transient_requests) {
        add_transient_allocation(graph, command_plan, request, plan, errors);
    }
    plan.arena_size = align_up(plan.arena_size, plan.arena_alignment);
    return plan;
}

static void append_command(const Graph &          graph,
                           const CommandPlan &    plan,
                           const KernelCorpus &   corpus,
                           const std::string &    target,
                           const Dispatch &       dispatch,
                           bool                   linear_dependency,
                           std::vector<Command> & commands,
                           Status &               status) {
    Command command;
    command.ordinal = static_cast<uint32_t>(commands.size());
    command.kind    = CommandKind::Kernel;
    command.kernel  = dispatch.kernel;
    // TODO: replace this linear ordinal dependency with real graph/resource dependency analysis.
    if (linear_dependency && command.ordinal > 0) {
        command.dependencies.push_back(command.ordinal - 1);
    }
    const KernelResolveResult resolved   = resolve_kernel_definition(corpus, target, command.kernel.kernel_id);
    const KernelDefinition *  definition = resolved.definition;
    if (!resolved.found()) {
        status.log("%s", format_kernel_resolve_error(resolved, command.kernel.kernel_id).c_str());
        definition = nullptr;
    } else if (dispatch.bindings.size() != definition->bindings.size()) {
        status.log("command %u kernel %s has %zu bindings but its ABI requires %zu", command.ordinal,
                   kernel_definition_name(*definition).c_str(), dispatch.bindings.size(), definition->bindings.size());
    }
    command.bindings.reserve(dispatch.bindings.size());
    for (size_t binding_index = 0; binding_index < dispatch.bindings.size(); ++binding_index) {
        const DispatchBinding & binding = dispatch.bindings[binding_index];
        CommandBinding          command_binding;
        command_binding.value                                      = binding.value;
        command_binding.offset                                     = binding.offset;
        command_binding.length                                     = binding.length;
        const Value *                               value          = graph.values().find(binding.value);
        const CommandPlanTransient *                plan_transient = find_plan_transient(plan, binding.value);
        const CommandPlanCompletionCounterRequest * completion_counter =
            find_plan_completion_counter_request(plan, binding.value);
        if (value == nullptr && plan_transient == nullptr && completion_counter == nullptr) {
            status.log("command %u binding %zu references missing value %d", command.ordinal, binding_index,
                       binding.value.value);
        } else if (value != nullptr) {
            command_binding.origin = command_binding_origin(value->kind);
        } else {
            command_binding.origin = CommandBindingOrigin::Transient;
        }
        if (command_binding.origin == CommandBindingOrigin::Transient) {
            const TransientBindingTarget binding_target = transient_binding_target(graph, binding.value);
            command_binding.value                       = binding_target.value;
            if (binding_target.offset > std::numeric_limits<size_t>::max() - command_binding.offset) {
                status.log("command %u binding %zu transient alias offset overflows", command.ordinal, binding_index);
            } else {
                command_binding.offset += binding_target.offset;
            }
        }
        if (definition != nullptr && binding_index < definition->bindings.size()) {
            command_binding.name   = string_value(definition->bindings[binding_index].name);
            command_binding.access = definition->bindings[binding_index].access;
        }
        command.bindings.push_back(std::move(command_binding));
    }
    commands.push_back(std::move(command));
}

static void verify_command_list(const std::vector<Command> & commands,
                                const TransientPlan &        transients,
                                const KernelCorpus &         corpus,
                                const std::string &          target,
                                Status &                     status) {
    for (size_t i = 0; i < commands.size(); ++i) {
        const Command &   command         = commands[i];
        const std::string command_context = format_command(command);
        if (command.ordinal != i) {
            status.log("%s has non-contiguous ordinal at index %zu", command_context.c_str(), i);
        }
        if (command.kind != CommandKind::Kernel) {
            status.log("%s is not a kernel command", command_context.c_str());
        }
        KernelResolveResult      resolved;
        const KernelDefinition * definition = nullptr;
        if (command.kind == CommandKind::Kernel) {
            resolved   = resolve_kernel_definition(corpus, target, command.kernel.kernel_id);
            definition = resolved.definition;
        }
        if (command.kind == CommandKind::Kernel && !resolved.found()) {
            status.log("%s: %s", command_context.c_str(),
                       format_kernel_resolve_error(resolved, command.kernel.kernel_id).c_str());
        } else if (definition != nullptr) {
            if (command.bindings.size() != definition->bindings.size()) {
                status.log("%s kernel %s has %zu bindings but its ABI requires %zu", command_context.c_str(),
                           kernel_definition_name(*definition).c_str(), command.bindings.size(),
                           definition->bindings.size());
            }
            const size_t shared_count = std::min(command.bindings.size(), definition->bindings.size());
            for (size_t binding_index = 0; binding_index < shared_count; ++binding_index) {
                const CommandBinding &          binding = command.bindings[binding_index];
                const KernelBindingDefinition & abi     = definition->bindings[binding_index];
                if (!string_equal(binding.name.c_str(), abi.name) || binding.access != abi.access) {
                    status.log("%s %s does not match ABI binding %zu", command_context.c_str(),
                               format_command_binding(binding).c_str(), binding_index);
                }
            }
        }
        if (command.bindings.empty()) {
            status.log("%s has no bindings", command_context.c_str());
        }
        for (uint32_t dependency : command.dependencies) {
            if (dependency >= command.ordinal) {
                status.log("%s has forward dependency %u", command_context.c_str(), dependency);
            }
        }
        for (const CommandBinding & binding : command.bindings) {
            const std::string binding_context = format_command_binding(binding);
            if (binding.origin != CommandBindingOrigin::GraphValue &&
                binding.origin != CommandBindingOrigin::Transient) {
                status.log("%s %s has an unsupported binding origin", command_context.c_str(), binding_context.c_str());
            }
            if (binding.origin == CommandBindingOrigin::Transient) {
                const TransientAllocation * allocation = find_transient_allocation(transients, binding.value);
                if (allocation == nullptr) {
                    status.log("%s %s has no transient allocation", command_context.c_str(), binding_context.c_str());
                } else if (binding.offset > allocation->size || binding.length > allocation->size - binding.offset) {
                    status.log("%s %s is outside transient allocation length %zu", command_context.c_str(),
                               binding_context.c_str(), allocation->size);
                }
            }
            if (binding.value.value < 0) {
                status.log("%s %s has an invalid value id", command_context.c_str(), binding_context.c_str());
            }
            if (binding.length == 0) {
                status.log("%s %s has an empty binding", command_context.c_str(), binding_context.c_str());
            }
        }
    }
}

}  // namespace

const TransientAllocation * find_transient_allocation(const TransientPlan & plan, ValueId value) {
    const auto found = std::find_if(plan.allocations.begin(), plan.allocations.end(),
                                    [&](const TransientAllocation & allocation) { return allocation.value == value; });
    return found == plan.allocations.end() ? nullptr : &*found;
}

CommandProgram build_command_program(const Graph &        graph,
                                     const CommandPlan &  plan,
                                     const KernelCorpus & corpus,
                                     const std::string &  target) {
    CommandProgram result;
    if (!plan.valid()) {
        result.status.append(plan.status);
        return result;
    }

    result.initialization_commands.reserve(plan.initialization_dispatches.size());
    for (const Dispatch & dispatch : plan.initialization_dispatches) {
        append_command(graph, plan, corpus, target, dispatch, false, result.initialization_commands, result.status);
    }
    for (const Dispatch & dispatch : plan.dispatches) {
        append_command(graph, plan, corpus, target, dispatch, true, result.commands, result.status);
    }
    result.transients = build_transient_plan(graph, plan, result.initialization_commands, result.commands,
                                             result.completion_counters, result.status);
    result.constant_initializations.reserve(plan.constant_initializations.size());
    for (const CommandPlanConstantInitialization & initialization : plan.constant_initializations) {
        result.constant_initializations.push_back({
            initialization.value,
            initialization.name,
            initialization.offset,
            initialization.data,
        });
    }
    return result;
}

VerificationResult verify_command_program(const CommandProgram & program,
                                          const KernelCorpus &   corpus,
                                          const std::string &    target) {
    VerificationResult result;
    if (!program.valid()) {
        result.status.append(program.status);
    }
    verify_command_list(program.initialization_commands, program.transients, corpus, target, result.status);
    verify_command_list(program.commands, program.transients, corpus, target, result.status);
    if (program.transients.arena_alignment == 0) {
        result.status.log("transient arena has zero alignment");
    }
    if (program.completion_counters.count == 0) {
        if (program.completion_counters.byte_count != 0) {
            result.status.log("completion counter region has bytes but no counters");
        }
    } else {
        if (program.completion_counters.byte_count == 0) {
            result.status.log("completion counter region has counters but no bytes");
        }
        if (program.completion_counters.byte_count % sizeof(int32_t) != 0) {
            result.status.log("completion counter region byte count is not i32 aligned");
        }
        if (program.completion_counters.arena_offset % 16 != 0) {
            result.status.log("completion counter region is not 16-byte aligned");
        }
        if (program.completion_counters.arena_offset > program.transients.arena_size ||
            program.completion_counters.byte_count >
                program.transients.arena_size - program.completion_counters.arena_offset) {
            result.status.log("completion counter region is outside transient arena");
        }
    }
    for (const TransientAllocation & allocation : program.transients.allocations) {
        if (allocation.value.value < 0 || allocation.size == 0 || allocation.alignment == 0 ||
            allocation.arena_offset % allocation.alignment != 0 ||
            allocation.arena_offset + allocation.size > program.transients.arena_size) {
            result.status.log("invalid transient allocation for value %d", allocation.value.value);
        }
        for (const TransientAllocation & other : program.transients.allocations) {
            if (allocation.value.value >= other.value.value) {
                continue;
            }
            const bool overlap = allocation.arena_offset < other.arena_offset + other.size &&
                                 other.arena_offset < allocation.arena_offset + allocation.size;
            if (overlap) {
                result.status.log("transient allocations overlap");
            }
        }
    }
    for (const ConstantInitialization & initialization : program.constant_initializations) {
        const TransientAllocation * allocation = find_transient_allocation(program.transients, initialization.value);
        if (allocation == nullptr) {
            result.status.log("constant initialization %s references missing transient value %d",
                              initialization.name.c_str(), initialization.value.value);
            continue;
        }
        if (initialization.data.empty()) {
            result.status.log("constant initialization %s has no data", initialization.name.c_str());
        }
        if (initialization.offset > allocation->size ||
            initialization.data.size() > allocation->size - initialization.offset) {
            result.status.log("constant initialization %s is outside transient allocation length %zu",
                              initialization.name.c_str(), allocation->size);
        }
    }
    return result;
}

}  // namespace ggml::hrx
