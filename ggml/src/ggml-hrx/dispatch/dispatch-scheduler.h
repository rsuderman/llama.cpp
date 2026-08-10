#pragma once

#include "dispatch.h"
#include "graph/graph.h"

#include <string>
#include <vector>

namespace ggml::hrx {

class DispatchScheduler {
  public:
    void enqueue(Dispatch dispatch);
    bool schedule_graph(const Graph & graph);

    const std::vector<Dispatch> & dispatches() const { return dispatches_; }

    const std::string & error() const { return error_; }

    static bool supports_node(const Graph & graph, const GraphNode * node);
    static bool can_schedule_graph(const Graph & graph);

  private:
    std::vector<Dispatch> dispatches_;
    std::string           error_;
};

}  // namespace ggml::hrx
