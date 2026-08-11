#pragma once

#include "dispatch-registry.h"

namespace ggml::hrx {

void register_qwen_moe_dispatches(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
