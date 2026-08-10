#pragma once

#include "graph/graph.h"

namespace ggml::hrx {

class DispatchScheduler;

bool supports_add_f32_dispatch(const Graph & graph, const GraphNode * node);
bool try_match_add_f32_dispatch(const Graph & graph, const GraphNode * node, DispatchScheduler & scheduler);

}  // namespace ggml::hrx
