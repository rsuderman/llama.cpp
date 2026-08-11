#include "dispatch-scheduler.h"

#include "dispatch-add.h"
#include "dispatch-rmsnorm.h"
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
    if (!graph.has_index()) {
        plan_.status.log("HRX graph is missing graph index");
        return false;
    }
    std::vector<bool> covered_nodes(nodes.size(), false);
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (covered_nodes[i]) {
            continue;
        }
        const GraphNode * node = &nodes[i];
        if (try_match_add_f32_dispatch(graph, node, *this)) {
            covered_nodes[i] = true;
            continue;
        }
        if (try_match_qwen_rmsnorm_f32_dispatch(graph, i, covered_nodes, *this)) {
            continue;
        }
        plan_.status.log("unsupported HRX node %zu: %s", i, ggml_op_name(node->op));
        plan_.dispatches.clear();
        return false;
    }
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (!covered_nodes[i]) {
            plan_.status.log("unsupported HRX node %zu: %s", i, ggml_op_name(nodes[i].op));
            plan_.dispatches.clear();
            return false;
        }
    }
    return true;
}

bool DispatchScheduler::supports_node(const Graph & graph, const GraphNode * node) {
    return supports_add_f32_dispatch(graph, node) || supports_qwen_rmsnorm_f32_dispatch(graph, node);
}

bool DispatchScheduler::can_schedule_graph(const Graph & graph) {
    DispatchScheduler scheduler;
    return scheduler.schedule_graph(graph);
}

}  // namespace ggml::hrx
