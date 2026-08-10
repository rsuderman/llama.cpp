#include "graph.h"

#include "ggml-impl.h"

#include <optional>
#include <unordered_map>
#include <utility>

namespace ggml::hrx {
namespace {

static bool tensor_is_external(const ggml_tensor *                                  tensor,
                               const std::unordered_map<const ggml_tensor *, int> & use_counts) {
    if (tensor->op == GGML_OP_NONE) {
        return true;
    }
    const auto found = use_counts.find(tensor);
    return found == use_counts.end() || found->second == 0;
}

static std::optional<ValueBufferBinding> resolve_external_buffer(const ggml_tensor * tensor,
                                                                 ValueBufferResolver resolver,
                                                                 void *              user_data) {
    if (resolver == nullptr) {
        return std::nullopt;
    }
    ValueBufferBinding binding;
    if (!resolver(tensor, binding, user_data)) {
        return std::nullopt;
    }
    return binding;
}

}  // namespace

GraphNode & Graph::add_node(ggml_op op, ValueId output, std::vector<ValueId> inputs, const ggml_tensor * tensor) {
    nodes_.push_back({ op, output, std::move(inputs), tensor });
    return nodes_.back();
}

GraphImportResult import_ggml_graph(const ggml_cgraph & graph,
                                    ValueBufferResolver resolver,
                                    void *              resolver_user_data) {
    GraphImportResult                            result;
    std::unordered_map<const ggml_tensor *, int> use_counts;
    for (int i = 0; i < graph.n_nodes; ++i) {
        const ggml_tensor * node = graph.nodes[i];
        if (node == nullptr) {
            result.errors.push_back("ggml graph contains a null node");
            return result;
        }
        for (const ggml_tensor * source : node->src) {
            if (source != nullptr) {
                ++use_counts[source];
            }
        }
    }

    ValueMap & values = result.graph.values();
    for (int i = 0; i < graph.n_nodes; ++i) {
        const ggml_tensor *  node = graph.nodes[i];
        std::vector<ValueId> inputs;
        for (const ggml_tensor * source : node->src) {
            if (source == nullptr) {
                continue;
            }
            const ValueKind kind = tensor_is_external(source, use_counts) ? ValueKind::External : ValueKind::Transient;
            inputs.push_back(values.get_or_add_tensor_value(
                source, kind,
                kind == ValueKind::External ? resolve_external_buffer(source, resolver, resolver_user_data) :
                                              std::nullopt));
        }

        const ValueKind output_kind = tensor_is_external(node, use_counts) ? ValueKind::External : ValueKind::Transient;
        const ValueId   output      = values.get_or_add_tensor_value(
            node, output_kind,
            output_kind == ValueKind::External ? resolve_external_buffer(node, resolver, resolver_user_data) :
                                                        std::nullopt);
        result.graph.add_node(node->op, output, std::move(inputs), node);
    }

    return result;
}

}  // namespace ggml::hrx
