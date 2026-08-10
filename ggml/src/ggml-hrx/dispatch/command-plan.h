#pragma once

#include "dispatch.h"
#include "error-log.h"

#include <vector>

namespace ggml::hrx {

struct CommandPlan {
    std::vector<Dispatch> dispatches;
    ErrorLog              errors;

    bool valid() const { return errors.success(); }
};

}  // namespace ggml::hrx
