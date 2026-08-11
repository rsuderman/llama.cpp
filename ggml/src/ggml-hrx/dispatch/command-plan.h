#pragma once

#include "dispatch.h"
#include "status.h"

#include <vector>

namespace ggml::hrx {

struct CommandPlan {
    std::vector<Dispatch> dispatches;
    Status                status;

    bool valid() const { return status.success(); }
};

}  // namespace ggml::hrx
