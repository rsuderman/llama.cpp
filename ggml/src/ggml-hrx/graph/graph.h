#pragma once

#include "ggml.h"
#include "value-map.h"

#include <string>
#include <vector>

struct ggml_cgraph;
struct ggml_tensor;

namespace ggml::hrx {

struct GraphNode {
    ggml_op              op;
    ValueId              output;
    std::vector<ValueId> inputs;
    const ggml_tensor *  tensor = nullptr;
};

class Graph {
  public:
    GraphNode & add_node(ggml_op op, ValueId output, std::vector<ValueId> inputs, const ggml_tensor * tensor);

    const std::vector<GraphNode> & nodes() const { return nodes_; }

    const ValueMap & values() const { return values_; }

    ValueMap & values() { return values_; }

  private:
    ValueMap               values_;
    std::vector<GraphNode> nodes_;
};

using ValueBufferResolver = bool (*)(const ggml_tensor * tensor, ValueBufferBinding & binding, void * user_data);

struct GraphImportResult {
    Graph                    graph;
    std::vector<std::string> errors;

    bool valid() const { return errors.empty(); }
};

GraphImportResult import_ggml_graph(const ggml_cgraph & graph, ValueBufferResolver resolver, void * resolver_user_data);

}  // namespace ggml::hrx
