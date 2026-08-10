#pragma once

#include "dispatch.h"

struct ggml_tensor;

namespace ggml::hrx {

class DispatchScheduler;

bool supports_add_f32_dispatch(const ggml_tensor * op);
bool try_match_add_f32_dispatch(const ggml_tensor *          node,
                                const DispatchMatchContext & context,
                                DispatchScheduler &          scheduler);

}  // namespace ggml::hrx
