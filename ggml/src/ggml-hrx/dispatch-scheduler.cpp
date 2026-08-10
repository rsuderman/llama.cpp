#include "dispatch-scheduler.h"

#include "dispatch-add.h"
#include "ggml-impl.h"
#include "ggml.h"

#include <string>
#include <utility>

namespace ggml::hrx {

void DispatchScheduler::enqueue(Dispatch dispatch) {
    dispatches_.push_back(std::move(dispatch));
}

bool DispatchScheduler::schedule_graph(const ggml_cgraph & graph, const DispatchMatchContext & context) {
    dispatches_.clear();
    error_.clear();
    for (int i = 0; i < graph.n_nodes; ++i) {
        if (try_match_add_f32_dispatch(graph.nodes[i], context, *this)) {
            continue;
        }
        error_ = "unsupported HRX node " + std::to_string(i) + ": " + ggml_op_desc(graph.nodes[i]);
        dispatches_.clear();
        return false;
    }
    return true;
}

bool DispatchScheduler::supports_op(const ggml_tensor * op) {
    return supports_add_f32_dispatch(op);
}

bool DispatchScheduler::can_schedule_graph(const ggml_cgraph & graph) {
    for (int i = 0; i < graph.n_nodes; ++i) {
        if (!supports_op(graph.nodes[i])) {
            return false;
        }
    }
    return true;
}

}  // namespace ggml::hrx
