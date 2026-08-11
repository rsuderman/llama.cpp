#pragma once

#include "graph/graph.h"

#include <cstddef>
#include <vector>

namespace ggml::hrx {

class DispatchScheduler;

bool supports_qwen_rmsnorm_f32_dispatch(const Graph & graph, const GraphNode * node);
bool try_match_qwen_rmsnorm_f32_dispatch(const Graph &       graph,
                                         size_t              node_index,
                                         std::vector<bool> & covered_nodes,
                                         DispatchScheduler & scheduler);

}  // namespace ggml::hrx
