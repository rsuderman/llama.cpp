#pragma once

#include "reactive-plan.h"

namespace ggml::hrx {

// Replaces recovered invocation-boundary placeholders with the exact runtime
// ABI of every concrete Qwen dispatch. Backend-only scratch values are
// appended to |graph| and participate in ordinary resource/lifetime planning.
VerificationResult materialize_qwen3_moe_dispatch_bindings(Graph & graph, Schedule & schedule);

} // namespace ggml::hrx
