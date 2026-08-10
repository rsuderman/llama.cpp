#include "graph-program-cache.h"

#include "dispatch/dispatch-scheduler.h"
#include "ggml-impl.h"
#include "ggml.h"

#include <cstddef>
#include <sstream>
#include <utility>

namespace ggml::hrx {
namespace {

static bool tensor_metadata_matches(const Value & value, const ggml_tensor * tensor) {
    if (tensor == nullptr || value.type != tensor->type || value.element_count != ggml_nelements(tensor) ||
        value.byte_count != ggml_nbytes(tensor) || value.contiguous != ggml_is_contiguous(tensor)) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (value.ne[i] != tensor->ne[i] || value.nb[i] != tensor->nb[i]) {
            return false;
        }
    }
    return true;
}

static bool bind_current_value(const ValueMap &                                   values,
                               ValueId                                            expected,
                               const ggml_tensor *                                tensor,
                               std::vector<const ggml_tensor *> &                 tensor_by_value,
                               std::unordered_map<const ggml_tensor *, int32_t> & value_by_tensor,
                               const char *                                       role,
                               size_t                                             node_index,
                               ErrorLog &                                         errors) {
    const Value * value = values.find(expected);
    if (value == nullptr || expected.value < 0 || static_cast<size_t>(expected.value) >= tensor_by_value.size()) {
        errors.log("node %zu %s references missing cached value %d", node_index, role, expected.value);
        return false;
    }
    if (tensor == nullptr) {
        errors.log("node %zu %s value %d maps to a null tensor", node_index, role, expected.value);
        return false;
    }

    const ggml_tensor * existing_tensor = tensor_by_value[static_cast<size_t>(expected.value)];
    if (existing_tensor != nullptr && existing_tensor != tensor) {
        errors.log("node %zu %s value %d maps to multiple current tensors", node_index, role, expected.value);
        return false;
    }

    const auto existing_value = value_by_tensor.find(tensor);
    if (existing_value != value_by_tensor.end() && existing_value->second != expected.value) {
        errors.log("node %zu %s tensor maps to cached values %d and %d", node_index, role, existing_value->second,
                   expected.value);
        return false;
    }

    if (existing_tensor == nullptr && existing_value == value_by_tensor.end() &&
        !tensor_metadata_matches(*value, tensor)) {
        errors.log("node %zu %s value %d metadata does not match current tensor", node_index, role, expected.value);
        return false;
    }

    tensor_by_value[static_cast<size_t>(expected.value)] = tensor;
    value_by_tensor.emplace(tensor, expected.value);
    return true;
}

static std::string command_program_shape_key(const CommandProgram & commands) {
    std::ostringstream out;
    out << "hrx-command-program-v1|commands=" << commands.commands.size();
    for (const Command & command : commands.commands) {
        out << "|ordinal=" << command.ordinal << "|kind=" << static_cast<int>(command.kind)
            << "|kernel=" << command.kernel.kernel_id;
        for (const auto & parameter : command.kernel.integer_parameters) {
            out << "|ip:" << parameter.first << '=' << parameter.second;
        }
        for (const auto & parameter : command.kernel.compile_parameters) {
            out << "|cp:" << parameter.first << '=' << parameter.second;
        }
        out << "|bindings=" << command.bindings.size();
        for (const CommandBinding & binding : command.bindings) {
            out << "|b:" << binding.name << ':' << binding.value.value << ':' << static_cast<int>(binding.origin) << ':'
                << binding.offset << ':' << binding.length << ':' << static_cast<int>(binding.access);
        }
        out << "|deps=" << command.dependencies.size();
        for (const uint32_t dependency : command.dependencies) {
            out << ':' << dependency;
        }
    }
    return out.str();
}

}  // namespace

GraphProgram::GraphProgram(uint64_t                        uid,
                           std::string                     target,
                           std::unique_ptr<Graph>          graph,
                           std::unique_ptr<CommandProgram> commands,
                           std::string                     command_shape) :
    uid_(uid),
    target_(std::move(target)),
    graph_(std::move(graph)),
    commands_(std::move(commands)),
    command_shape_(std::move(command_shape)) {}

GraphProgramMatch GraphProgram::match_current_graph(const ggml_cgraph & current_graph) const {
    GraphProgramMatch result;
    if (graph_ == nullptr) {
        result.errors.log("missing cached HRX graph");
        return result;
    }
    if (commands_ == nullptr) {
        result.errors.log("missing cached HRX command program");
        return result;
    }
    if (graph_->nodes().size() != static_cast<size_t>(current_graph.n_nodes)) {
        result.errors.log("cached graph has %zu nodes but current graph has %d", graph_->nodes().size(),
                          current_graph.n_nodes);
        return result;
    }

    const ValueMap &                                 values = graph_->values();
    std::vector<const ggml_tensor *>                 tensor_by_value(values.size(), nullptr);
    std::unordered_map<const ggml_tensor *, int32_t> value_by_tensor;

    for (size_t node_index = 0; node_index < graph_->nodes().size(); ++node_index) {
        const GraphNode &   cached_node  = graph_->nodes()[node_index];
        const ggml_tensor * current_node = current_graph.nodes[node_index];
        if (current_node == nullptr) {
            result.errors.log("current graph node %zu is null", node_index);
            return result;
        }
        if (cached_node.op != current_node->op) {
            result.errors.log("node %zu cached op %s does not match current op %s", node_index,
                              ggml_op_name(cached_node.op), ggml_op_name(current_node->op));
            return result;
        }

        size_t input_index = 0;
        for (const ggml_tensor * source : current_node->src) {
            if (source == nullptr) {
                continue;
            }
            if (input_index >= cached_node.inputs.size()) {
                result.errors.log("node %zu has more inputs than the cached graph", node_index);
                return result;
            }
            if (!bind_current_value(values, cached_node.inputs[input_index], source, tensor_by_value, value_by_tensor,
                                    "input", node_index, result.errors)) {
                return result;
            }
            ++input_index;
        }
        if (input_index != cached_node.inputs.size()) {
            result.errors.log("node %zu has %zu inputs but cached graph has %zu", node_index, input_index,
                              cached_node.inputs.size());
            return result;
        }
        if (!bind_current_value(values, cached_node.output, current_node, tensor_by_value, value_by_tensor, "output",
                                node_index, result.errors)) {
            return result;
        }
    }

    for (const ValueId id : values.external_value_ids()) {
        if (id.value < 0 || static_cast<size_t>(id.value) >= tensor_by_value.size() ||
            tensor_by_value[static_cast<size_t>(id.value)] == nullptr) {
            result.errors.log("external value %d is missing from the current graph", id.value);
            return result;
        }
        result.external_bindings.push_back({ id, tensor_by_value[static_cast<size_t>(id.value)] });
    }
    return result;
}

bool GraphProgramCache::can_execute(const ggml_cgraph &  graph,
                                    const KernelCorpus & corpus,
                                    const std::string &  target) const {
    if (graph.n_nodes == 0) {
        return true;
    }
    ErrorLog errors;
    return build_program(graph, corpus, target, errors) != nullptr;
}

GraphProgramLookup GraphProgramCache::get_or_build(const ggml_cgraph &  graph,
                                                   const KernelCorpus & corpus,
                                                   const std::string &  target) {
    GraphProgramLookup result;
    if (graph.uid != 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto                  found = programs_.find(graph.uid);
        if (found != programs_.end() && found->second->target() == target) {
            GraphProgramMatch match = found->second->match_current_graph(graph);
            if (match.valid()) {
                result.program = found->second.get();
                result.match   = std::move(match);
                ++stats_.hits;
                return result;
            }
        }
    }

    std::unique_ptr<GraphProgram> program = build_program(graph, corpus, target, result.errors);
    if (program == nullptr) {
        return result;
    }

    GraphProgramMatch match = program->match_current_graph(graph);
    if (!match.valid()) {
        result.errors.append(match.errors);
        return result;
    }

    if (graph.uid == 0) {
        result.uncached_program = std::move(program);
        result.program          = result.uncached_program.get();
        result.match            = std::move(match);
        return result;
    }

    GraphProgram * cached_program = program.get();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        programs_[graph.uid] = std::move(program);
        cached_program       = programs_[graph.uid].get();
        ++stats_.builds;
    }
    result.program = cached_program;
    result.match   = std::move(match);
    return result;
}

GraphProgramCacheStats GraphProgramCache::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void GraphProgramCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    programs_.clear();
}

std::unique_ptr<GraphProgram> GraphProgramCache::build_program(const ggml_cgraph &  graph,
                                                               const KernelCorpus & corpus,
                                                               const std::string &  target,
                                                               ErrorLog &           errors) const {
    GraphImportResult imported = import_ggml_graph(graph);
    if (!imported.valid()) {
        errors.append(imported.errors);
        return nullptr;
    }

    DispatchScheduler scheduler;
    if (!scheduler.schedule_graph(imported.graph)) {
        errors.append(scheduler.plan().errors);
        return nullptr;
    }

    CommandProgram commands = build_command_program(scheduler.plan(), corpus, target);
    if (!commands.valid()) {
        errors.append(commands.errors);
        return nullptr;
    }

    std::string command_shape = command_program_shape_key(commands);
    return std::make_unique<GraphProgram>(graph.uid, target, std::make_unique<Graph>(std::move(imported.graph)),
                                          std::make_unique<CommandProgram>(std::move(commands)),
                                          std::move(command_shape));
}

bool can_execute_standalone_op_as_graph(const ggml_tensor * op) {
    if (op == nullptr) {
        return false;
    }
    Graph                graph;
    std::vector<ValueId> inputs;
    for (const ggml_tensor * source : op->src) {
        if (source == nullptr) {
            continue;
        }
        inputs.push_back(graph.values().get_or_add_tensor_value(source, ValueKind::External));
    }
    const ValueId     output = graph.values().get_or_add_tensor_value(op, ValueKind::External);
    const GraphNode & node   = graph.add_node(op->op, output, std::move(inputs));
    return DispatchScheduler::supports_node(graph, &node);
}

}  // namespace ggml::hrx
