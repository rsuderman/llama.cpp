#pragma once

#include "dispatch/command-plan.h"
#include "ggml.h"
#include "graph/graph.h"
#include "status.h"

#include <cstddef>
#include <string>
#include <vector>

namespace ggml::hrx {

struct DispatchTarget {
    std::string architecture;
};

enum class DispatchMatchKind {
    Fused,
    SingleOp,
};

enum class DispatchSource {
    Common,
    Qwen,
};

struct DispatchMatchContext {
    const Graph &             graph;
    const GraphNode *         root_node  = nullptr;
    size_t                    root_index = 0;
    const std::vector<bool> & covered_nodes;
    const CommandPlan &       plan;
    ValueId                   next_plan_value;
};

struct DispatchMatch {
    std::vector<size_t>                              covered_nodes;
    std::vector<Dispatch>                            dispatches;
    std::vector<CommandPlanTransient>                transients;
    std::vector<CommandPlanConstantInitialization>   constant_initializations;
    std::vector<CommandPlanCompletionCounterRequest> completion_counter_requests;
    CommandPlanMetadata                              metadata;
    Status                                           status;
};

using DispatchMatcher = bool (*)(const DispatchMatchContext & context, DispatchMatch & match);

struct DispatchRegistration {
    const char *      name     = "";
    ggml_op           root_op  = GGML_OP_NONE;
    DispatchMatchKind kind     = DispatchMatchKind::SingleOp;
    int               priority = 0;
    DispatchSource    source   = DispatchSource::Common;
    DispatchMatcher   matcher  = nullptr;
};

class DispatchRegistry {
  public:
    bool match(const DispatchMatchContext & context, DispatchMatch & match) const;

    const std::vector<DispatchRegistration> & registrations_for_root(ggml_op root_op) const;

    const std::vector<DispatchRegistration> & single_op_registrations() const { return single_op_registrations_; }

  private:
    friend class DispatchRegistryBuilder;

    struct RegistrationGroup {
        std::vector<DispatchRegistration> fused;
        std::vector<DispatchRegistration> single_op;
        std::vector<DispatchRegistration> ordered;
    };

    std::vector<RegistrationGroup>    registrations_by_root_;
    std::vector<DispatchRegistration> single_op_registrations_;
};

class DispatchRegistryBuilder {
  public:
    void             add(DispatchRegistration registration);
    DispatchRegistry build();

  private:
    DispatchRegistry registry_;
};

const DispatchRegistry * find_dispatch_registry(const DispatchTarget & target);

}  // namespace ggml::hrx
