#include "command-program.h"

#include "command-program-diagnostics.h"

#include <algorithm>
#include <cstring>
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

static bool has_transient_allocation(const TransientPlan & plan, ValueId value) {
    return find_transient_allocation(plan, value) != nullptr;
}

static void add_transient_allocation(const Graph & graph, ValueId value, TransientPlan & plan, ErrorLog & errors) {
    if (has_transient_allocation(plan, value)) {
        return;
    }
    const Value * graph_value = graph.values().find(value);
    if (graph_value == nullptr) {
        errors.log("transient value %d is missing from graph values", value.value);
        return;
    }
    if (graph_value->kind != ValueKind::Transient) {
        return;
    }

    TransientAllocation allocation;
    allocation.value        = value;
    allocation.size         = graph_value->byte_count;
    allocation.alignment    = 256;
    allocation.arena_offset = align_up(plan.arena_size, allocation.alignment);
    plan.arena_size         = allocation.arena_offset + allocation.size;
    plan.allocations.push_back(allocation);
}

static TransientPlan build_transient_plan(const Graph &                graph,
                                          const std::vector<Command> & commands,
                                          ErrorLog &                   errors) {
    TransientPlan plan;
    plan.arena_alignment = 256;
    for (const Command & command : commands) {
        for (const CommandBinding & binding : command.bindings) {
            if (binding.origin == CommandBindingOrigin::Transient) {
                add_transient_allocation(graph, binding.value, plan, errors);
            }
        }
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
        result.errors.append(plan.errors);
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
            result.errors.log("%s", format_kernel_resolve_error(resolved, command.kernel.kernel_id).c_str());
            definition = nullptr;
        } else if (dispatch.bindings.size() != definition->bindings.size()) {
            result.errors.log("command %u kernel %s has %zu bindings but its ABI requires %zu", command.ordinal,
                              kernel_definition_name(*definition).c_str(), dispatch.bindings.size(),
                              definition->bindings.size());
        }
        command.bindings.reserve(dispatch.bindings.size());
        for (size_t binding_index = 0; binding_index < dispatch.bindings.size(); ++binding_index) {
            const DispatchBinding & binding = dispatch.bindings[binding_index];
            CommandBinding          command_binding;
            command_binding.value  = binding.value;
            command_binding.offset = binding.offset;
            command_binding.length = binding.length;
            const Value * value    = graph.values().find(binding.value);
            if (value == nullptr) {
                result.errors.log("command %u binding %zu references missing graph value %d", command.ordinal,
                                  binding_index, binding.value.value);
            } else {
                command_binding.origin = command_binding_origin(value->kind);
            }
            if (definition != nullptr && binding_index < definition->bindings.size()) {
                command_binding.name   = string_value(definition->bindings[binding_index].name);
                command_binding.access = definition->bindings[binding_index].access;
            }
            command.bindings.push_back(std::move(command_binding));
        }
        result.commands.push_back(std::move(command));
    }
    result.transients = build_transient_plan(graph, result.commands, result.errors);
    return result;
}

VerificationResult verify_command_program(const CommandProgram & program,
                                          const KernelCorpus &   corpus,
                                          const std::string &    target) {
    VerificationResult result;
    if (!program.valid()) {
        result.errors.append(program.errors);
    }
    for (size_t i = 0; i < program.commands.size(); ++i) {
        const Command &   command         = program.commands[i];
        const std::string command_context = format_command(command);
        if (command.ordinal != i) {
            result.errors.log("%s has non-contiguous ordinal at index %zu", command_context.c_str(), i);
        }
        if (command.kind != CommandKind::Kernel) {
            result.errors.log("%s is not a kernel command", command_context.c_str());
        }
        KernelResolveResult      resolved;
        const KernelDefinition * definition = nullptr;
        if (command.kind == CommandKind::Kernel) {
            resolved   = resolve_kernel_definition(corpus, target, command.kernel.kernel_id);
            definition = resolved.definition;
        }
        if (command.kind == CommandKind::Kernel && !resolved.found()) {
            result.errors.log("%s: %s", command_context.c_str(),
                              format_kernel_resolve_error(resolved, command.kernel.kernel_id).c_str());
        } else if (definition != nullptr) {
            if (command.bindings.size() != definition->bindings.size()) {
                result.errors.log("%s kernel %s has %zu bindings but its ABI requires %zu", command_context.c_str(),
                                  kernel_definition_name(*definition).c_str(), command.bindings.size(),
                                  definition->bindings.size());
            }
            const size_t shared_count = std::min(command.bindings.size(), definition->bindings.size());
            for (size_t binding_index = 0; binding_index < shared_count; ++binding_index) {
                const CommandBinding &          binding = command.bindings[binding_index];
                const KernelBindingDefinition & abi     = definition->bindings[binding_index];
                if (!string_equal(binding.name.c_str(), abi.name) || binding.access != abi.access) {
                    result.errors.log("%s %s does not match ABI binding %zu", command_context.c_str(),
                                      format_command_binding(binding).c_str(), binding_index);
                }
            }
        }
        if (command.bindings.empty()) {
            result.errors.log("%s has no bindings", command_context.c_str());
        }
        for (uint32_t dependency : command.dependencies) {
            if (dependency >= command.ordinal) {
                result.errors.log("%s has forward dependency %u", command_context.c_str(), dependency);
            }
        }
        for (const CommandBinding & binding : command.bindings) {
            const std::string binding_context = format_command_binding(binding);
            if (binding.origin != CommandBindingOrigin::GraphValue &&
                binding.origin != CommandBindingOrigin::Transient) {
                result.errors.log("%s %s has an unsupported binding origin", command_context.c_str(),
                                  binding_context.c_str());
            }
            if (binding.origin == CommandBindingOrigin::Transient) {
                const TransientAllocation * allocation = find_transient_allocation(program.transients, binding.value);
                if (allocation == nullptr) {
                    result.errors.log("%s %s has no transient allocation", command_context.c_str(),
                                      binding_context.c_str());
                } else if (binding.offset > allocation->size || binding.length > allocation->size - binding.offset) {
                    result.errors.log("%s %s is outside transient allocation length %zu", command_context.c_str(),
                                      binding_context.c_str(), allocation->size);
                }
            }
            if (binding.value.value < 0) {
                result.errors.log("%s %s has an invalid value id", command_context.c_str(), binding_context.c_str());
            }
            if (binding.length == 0) {
                result.errors.log("%s %s has an empty binding", command_context.c_str(), binding_context.c_str());
            }
        }
    }
    if (program.transients.arena_alignment == 0) {
        result.errors.log("transient arena has zero alignment");
    }
    for (const TransientAllocation & allocation : program.transients.allocations) {
        if (allocation.value.value < 0 || allocation.size == 0 || allocation.alignment == 0 ||
            allocation.arena_offset % allocation.alignment != 0 ||
            allocation.arena_offset + allocation.size > program.transients.arena_size) {
            result.errors.log("invalid transient allocation for value %d", allocation.value.value);
        }
        for (const TransientAllocation & other : program.transients.allocations) {
            if (allocation.value.value >= other.value.value) {
                continue;
            }
            const bool overlap = allocation.arena_offset < other.arena_offset + other.size &&
                                 other.arena_offset < allocation.arena_offset + allocation.size;
            if (overlap) {
                result.errors.log("transient allocations overlap");
            }
        }
    }
    return result;
}

}  // namespace ggml::hrx
