#pragma once

#include "command-plan-metadata.h"
#include "dispatch.h"
#include "status.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ggml::hrx {

struct CommandPlanTransient {
    ValueId     value;
    std::string name;
    size_t      size      = 0;
    size_t      alignment = 256;
};

struct CommandPlanConstantInitialization {
    ValueId              value;
    std::string          name;
    size_t               offset = 0;
    std::vector<uint8_t> data;
};

struct CommandPlanCompletionCounterRequest {
    ValueId     value;
    std::string name;
    uint32_t    count = 0;
};

struct CommandPlan {
    std::vector<Dispatch>                            dispatches;
    std::vector<CommandPlanTransient>                transients;
    std::vector<CommandPlanConstantInitialization>   constant_initializations;
    std::vector<CommandPlanCompletionCounterRequest> completion_counter_requests;
    CommandPlanMetadata                              metadata;
    Status                                           status;

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
