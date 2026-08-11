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

static const CommandPlanTransient * find_plan_transient(const CommandPlan & plan, ValueId value) {
    const auto found = std::find_if(plan.transients.begin(), plan.transients.end(),
                                    [&](const CommandPlanTransient & transient) { return transient.value == value; });
    return found == plan.transients.end() ? nullptr : &*found;
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

static TransientPlan build_transient_plan(const Graph &                graph,
                                          const CommandPlan &          command_plan,
                                          const std::vector<Command> & commands,
                                          Status &                     errors) {
    TransientPlan plan;
    plan.arena_alignment = 256;
    std::vector<TransientAllocationRequest> requests;
    for (const Command & command : commands) {
        for (const CommandBinding & binding : command.bindings) {
            if (binding.origin == CommandBindingOrigin::Transient) {
                if (binding.offset > std::numeric_limits<size_t>::max() - binding.length) {
                    errors.log("transient value %d binding range overflows", binding.value.value);
                    continue;
                }
                add_transient_allocation_request(requests, binding.value, binding.offset + binding.length);
            }
        }
    }
    for (const TransientAllocationRequest & request : requests) {
        add_transient_allocation(graph, command_plan, request, plan, errors);
    }
    plan.arena_size = align_up(plan.arena_size, plan.arena_alignment);
    return plan;
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

    result.commands.reserve(plan.dispatches.size());
    for (const Dispatch & dispatch : plan.dispatches) {
        Command command;
        command.ordinal = static_cast<uint32_t>(result.commands.size());
        command.kind    = CommandKind::Kernel;
        command.kernel  = dispatch.kernel;
        // TODO: replace this linear ordinal dependency with real graph/resource dependency analysis.
        if (command.ordinal > 0) {
            command.dependencies.push_back(command.ordinal - 1);
        }
        const KernelResolveResult resolved   = resolve_kernel_definition(corpus, target, command.kernel.kernel_id);
        const KernelDefinition *  definition = resolved.definition;
        if (!resolved.found()) {
            result.status.log("%s", format_kernel_resolve_error(resolved, command.kernel.kernel_id).c_str());
            definition = nullptr;
        } else if (dispatch.bindings.size() != definition->bindings.size()) {
            result.status.log("command %u kernel %s has %zu bindings but its ABI requires %zu", command.ordinal,
                              kernel_definition_name(*definition).c_str(), dispatch.bindings.size(),
                              definition->bindings.size());
        }
        command.bindings.reserve(dispatch.bindings.size());
        for (size_t binding_index = 0; binding_index < dispatch.bindings.size(); ++binding_index) {
            const DispatchBinding & binding = dispatch.bindings[binding_index];
            CommandBinding          command_binding;
            command_binding.value                       = binding.value;
            command_binding.offset                      = binding.offset;
            command_binding.length                      = binding.length;
            const Value *                value          = graph.values().find(binding.value);
            const CommandPlanTransient * plan_transient = find_plan_transient(plan, binding.value);
            if (value == nullptr && plan_transient == nullptr) {
                result.status.log("command %u binding %zu references missing value %d", command.ordinal, binding_index,
                                  binding.value.value);
            } else if (value != nullptr) {
                command_binding.origin = command_binding_origin(value->kind);
            } else {
                command_binding.origin = CommandBindingOrigin::Transient;
            }
            if (definition != nullptr && binding_index < definition->bindings.size()) {
                command_binding.name   = string_value(definition->bindings[binding_index].name);
                command_binding.access = definition->bindings[binding_index].access;
            }
            command.bindings.push_back(std::move(command_binding));
        }
        result.commands.push_back(std::move(command));
    }
    result.transients = build_transient_plan(graph, plan, result.commands, result.status);
    return result;
}

VerificationResult verify_command_program(const CommandProgram & program,
                                          const KernelCorpus &   corpus,
                                          const std::string &    target) {
    VerificationResult result;
    if (!program.valid()) {
        result.status.append(program.status);
    }
    for (size_t i = 0; i < program.commands.size(); ++i) {
        const Command &   command         = program.commands[i];
        const std::string command_context = format_command(command);
        if (command.ordinal != i) {
            result.status.log("%s has non-contiguous ordinal at index %zu", command_context.c_str(), i);
        }
        if (command.kind != CommandKind::Kernel) {
            result.status.log("%s is not a kernel command", command_context.c_str());
        }
        KernelResolveResult      resolved;
        const KernelDefinition * definition = nullptr;
        if (command.kind == CommandKind::Kernel) {
            resolved   = resolve_kernel_definition(corpus, target, command.kernel.kernel_id);
            definition = resolved.definition;
        }
        if (command.kind == CommandKind::Kernel && !resolved.found()) {
            result.status.log("%s: %s", command_context.c_str(),
                              format_kernel_resolve_error(resolved, command.kernel.kernel_id).c_str());
        } else if (definition != nullptr) {
            if (command.bindings.size() != definition->bindings.size()) {
                result.status.log("%s kernel %s has %zu bindings but its ABI requires %zu", command_context.c_str(),
                                  kernel_definition_name(*definition).c_str(), command.bindings.size(),
                                  definition->bindings.size());
            }
            const size_t shared_count = std::min(command.bindings.size(), definition->bindings.size());
            for (size_t binding_index = 0; binding_index < shared_count; ++binding_index) {
                const CommandBinding &          binding = command.bindings[binding_index];
                const KernelBindingDefinition & abi     = definition->bindings[binding_index];
                if (!string_equal(binding.name.c_str(), abi.name) || binding.access != abi.access) {
                    result.status.log("%s %s does not match ABI binding %zu", command_context.c_str(),
                                      format_command_binding(binding).c_str(), binding_index);
                }
            }
        }
        if (command.bindings.empty()) {
            result.status.log("%s has no bindings", command_context.c_str());
        }
        for (uint32_t dependency : command.dependencies) {
            if (dependency >= command.ordinal) {
                result.status.log("%s has forward dependency %u", command_context.c_str(), dependency);
            }
        }
        for (const CommandBinding & binding : command.bindings) {
            const std::string binding_context = format_command_binding(binding);
            if (binding.origin != CommandBindingOrigin::GraphValue &&
                binding.origin != CommandBindingOrigin::Transient) {
                result.status.log("%s %s has an unsupported binding origin", command_context.c_str(),
                                  binding_context.c_str());
            }
            if (binding.origin == CommandBindingOrigin::Transient) {
                const TransientAllocation * allocation = find_transient_allocation(program.transients, binding.value);
                if (allocation == nullptr) {
                    result.status.log("%s %s has no transient allocation", command_context.c_str(),
                                      binding_context.c_str());
                } else if (binding.offset > allocation->size || binding.length > allocation->size - binding.offset) {
                    result.status.log("%s %s is outside transient allocation length %zu", command_context.c_str(),
                                      binding_context.c_str(), allocation->size);
                }
            }
            if (binding.value.value < 0) {
                result.status.log("%s %s has an invalid value id", command_context.c_str(), binding_context.c_str());
            }
            if (binding.length == 0) {
                result.status.log("%s %s has an empty binding", command_context.c_str(), binding_context.c_str());
            }
        }
    }
    if (program.transients.arena_alignment == 0) {
        result.status.log("transient arena has zero alignment");
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
    return result;
}

}  // namespace ggml::hrx
