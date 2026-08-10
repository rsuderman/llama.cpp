#pragma once

#include "dispatch.h"

#include <string>
#include <vector>

struct ggml_cgraph;
struct ggml_tensor;

namespace ggml::hrx {

class DispatchScheduler {
  public:
    void enqueue(Dispatch dispatch);
    bool schedule_graph(const ggml_cgraph & graph, const DispatchMatchContext & context);

    const std::vector<Dispatch> & dispatches() const { return dispatches_; }

    const std::string & error() const { return error_; }

    static bool supports_op(const ggml_tensor * op);
    static bool can_schedule_graph(const ggml_cgraph & graph);

  private:
    std::vector<Dispatch> dispatches_;
    std::string           error_;
};

}  // namespace ggml::hrx
