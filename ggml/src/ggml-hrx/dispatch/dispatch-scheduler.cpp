#include "dispatch-scheduler.h"

#include "ggml.h"
#include "graph/graph-traversal.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

static bool match_covers_root(const DispatchMatch & match, size_t root_index) {
    return std::find(match.covered_nodes.begin(), match.covered_nodes.end(), root_index) != match.covered_nodes.end();
}

static bool match_overlaps_covered_nodes(const DispatchMatch & match, const std::vector<bool> & covered_nodes) {
    for (const size_t node_index : match.covered_nodes) {
        if (node_index >= covered_nodes.size() || covered_nodes[node_index]) {
            return true;
        }
    }
    return false;
}

static bool try_match_registration(const Graph &             graph,
                                   const GraphNode *         node,
                                   size_t                    node_index,
                                   const std::vector<bool> & covered_nodes,
                                   const DispatchRegistry &  registry,
                                   DispatchMatch &           match) {
    const DispatchMatchContext context = {
        graph,
        node,
        node_index,
        covered_nodes,
    };
    return registry.match(context, match);
}

}  // namespace

bool DispatchScheduler::schedule_graph(const Graph & graph, const DispatchTarget & target) {
    plan_                             = {};
    const DispatchRegistry * registry = find_dispatch_registry(target);
    if (registry == nullptr) {
        plan_.status.log("no HRX dispatch registry for target %s", target.architecture.c_str());
        return false;
    }
    const std::vector<GraphNode> & nodes = graph.nodes();
    if (!graph.has_index()) {
        plan_.status.log("HRX graph is missing graph index");
        return false;
    }
    std::vector<bool>         covered_nodes(nodes.size(), false);
    const GraphTraversalOrder traversal = GraphTraversalOrder::build(graph);
    for (const GraphNode * node : traversal.nodes()) {
        size_t i = 0;
        if (node == nullptr || !graph.index().node_index(node, i)) {
            plan_.status.log("HRX traversal references a node outside the graph");
            plan_.dispatches.clear();
            return false;
        }
        if (covered_nodes[i]) {
            continue;
        }
        DispatchMatch match;
        if (!try_match_registration(graph, node, i, covered_nodes, *registry, match)) {
            plan_.status.log("unsupported HRX node %zu: %s", i, ggml_op_name(node->op));
            plan_.dispatches.clear();
            return false;
        }
        if (match.covered_nodes.empty() || match.dispatches.empty() || !match_covers_root(match, i) ||
            match_overlaps_covered_nodes(match, covered_nodes)) {
            plan_.status.log("invalid HRX dispatch match for node %zu: %s", i, ggml_op_name(node->op));
            plan_.dispatches.clear();
            return false;
        }
        for (Dispatch & dispatch : match.dispatches) {
            plan_.dispatches.push_back(std::move(dispatch));
        }
        for (const size_t covered_node : match.covered_nodes) {
            covered_nodes[covered_node] = true;
        }
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

bool DispatchScheduler::supports_node(const Graph & graph, const GraphNode * node, const DispatchTarget & target) {
    const DispatchRegistry * registry = find_dispatch_registry(target);
    if (registry == nullptr) {
        return false;
    }
    if (node == nullptr || !graph.has_index()) {
        return false;
    }
    size_t node_index = 0;
    if (!graph.index().node_index(node, node_index)) {
        return false;
    }
    const std::vector<bool> covered_nodes(graph.nodes().size(), false);
    DispatchMatch           match;
    return try_match_registration(graph, node, node_index, covered_nodes, *registry, match);
}

bool DispatchScheduler::can_schedule_graph(const Graph & graph, const DispatchTarget & target) {
    DispatchScheduler scheduler;
    return scheduler.schedule_graph(graph, target);
}

}  // namespace ggml::hrx
