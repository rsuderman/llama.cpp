#pragma once

#include "command-plan.h"
#include "graph/graph.h"

#include <string>
#include <vector>

namespace ggml::hrx {

class DispatchScheduler {
  public:
    void enqueue(Dispatch dispatch);
    bool schedule_graph(const Graph & graph);

    const CommandPlan & plan() const { return plan_; }

    const std::vector<Dispatch> & dispatches() const { return plan_.dispatches; }

    const std::string & error() const {
        static const std::string empty;
        return plan_.status.errors().empty() ? empty : plan_.status.errors().front();
    }

    static bool supports_node(const Graph & graph, const GraphNode * node);
    static bool can_schedule_graph(const Graph & graph);

  private:
    CommandPlan plan_;
};

}  // namespace ggml::hrx
