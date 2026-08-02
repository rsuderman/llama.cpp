#pragma once

#include "schedule.h"

#include <string>
#include <vector>

namespace ggml::hrx {

struct QwenProgramProof {
    Schedule schedule;
    std::vector<std::string> errors;
    std::vector<std::string> native_gaps;
    std::vector<std::string> root_seams;

    bool recognized() const { return errors.empty() && !schedule.invocations.empty(); }
    bool structurally_sufficient() const;
    bool natively_complete() const;
};

QwenProgramProof recover_owned_qwen3_moe_program(const Graph & graph);
VerificationResult verify_owned_qwen3_moe_program(const Graph & graph, const QwenProgramProof & proof);
std::string qwen_program_signature(const QwenProgramProof & proof);

} // namespace ggml::hrx
