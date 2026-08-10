#pragma once

#include "dispatch.h"

#include <string>
#include <vector>

namespace ggml::hrx {

struct CommandPlan {
    std::vector<Dispatch> dispatches;
    std::string           error;

    bool valid() const { return error.empty(); }
};

}  // namespace ggml::hrx
