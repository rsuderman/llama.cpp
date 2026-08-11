#pragma once

#include "backend-context.h"
#include "dispatch/command-program-bindings.h"
#include "error-log.h"
#include "ggml.h"
#include "runtime/graph-program-cache.h"

struct ggml_cgraph;

namespace ggml::hrx {

struct GraphSupportResult {
    bool     supported = false;
    ErrorLog errors;

    bool success() const { return supported && errors.success(); }
};

struct GraphExecutionResult {
    enum ggml_status status = GGML_STATUS_FAILED;
    ErrorLog         errors;

    bool success() const { return status == GGML_STATUS_SUCCESS && errors.success(); }
};

class GraphExecutor {
  public:
    explicit GraphExecutor(ggml_backend_hrx_context & context);

    GraphSupportResult   can_execute(const ggml_cgraph & graph) const;
    GraphExecutionResult execute(const ggml_cgraph & graph) const;

  private:
    bool context_valid_for_graph_programs(ErrorLog & errors) const;
    bool context_valid_for_execution(ErrorLog & errors) const;

    CommandProgramBindings bind_external_value_buffers(const GraphProgramMatch & match) const;

    ggml_backend_hrx_context & context_;
};

}  // namespace ggml::hrx
