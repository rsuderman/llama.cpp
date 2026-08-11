#include "dispatch-scheduler.h"

#include "dispatch-add.h"
#include "ggml.h"

#include <cstddef>
#include <string>
#include <utility>

namespace ggml::hrx {

void DispatchScheduler::enqueue(Dispatch dispatch) {
    plan_.dispatches.push_back(std::move(dispatch));
}

bool DispatchScheduler::schedule_graph(const Graph & graph) {
    plan_                                = {};
    const std::vector<GraphNode> & nodes = graph.nodes();
    for (size_t i = 0; i < nodes.size(); ++i) {
        const GraphNode * node = &nodes[i];
        if (try_match_add_f32_dispatch(graph, node, *this)) {
            continue;
        }
        plan_.status.log("unsupported HRX node %zu: %s", i, ggml_op_name(node->op));
        plan_.dispatches.clear();
        return false;
    }
    return true;
}

bool DispatchScheduler::supports_node(const Graph & graph, const GraphNode * node) {
    return supports_add_f32_dispatch(graph, node);
}

bool DispatchScheduler::can_schedule_graph(const Graph & graph) {
    for (const GraphNode & node : graph.nodes()) {
        if (!supports_node(graph, &node)) {
            return false;
        }
    }
    return true;
}

}  // namespace ggml::hrx
