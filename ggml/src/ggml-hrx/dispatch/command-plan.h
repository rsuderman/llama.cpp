#pragma once

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
    Status                            status;

    bool valid() const { return status.success(); }
};

}  // namespace ggml::hrx
