#pragma once

#include "command-plan-metadata.h"
#include "dispatch.h"
#include "status.h"

#include <cstddef>
#include <string>
#include <vector>

namespace ggml::hrx {

struct CommandPlanTransient {
    ValueId     value;
    std::string name;
    size_t      size      = 0;
    size_t      alignment = 256;
};

struct CommandPlan {
    std::vector<Dispatch>             dispatches;
    std::vector<CommandPlanTransient> transients;
    CommandPlanMetadata               metadata;
    Status                            status;

    bool valid() const { return status.success(); }
};

inline const CommandPlanAlternateValue * find_alternate_value(const CommandPlan & plan, ValueId graph_value) {
    return plan.metadata.find_alternate_value(graph_value);
}

inline const CommandPlanAlternateValue * find_alternate_value(const CommandPlan & plan,
                                                              ValueId             graph_value,
                                                              ggml_type           type,
                                                              size_t              byte_count) {
    return plan.metadata.find_alternate_value(graph_value, type, byte_count);
}

}  // namespace ggml::hrx
