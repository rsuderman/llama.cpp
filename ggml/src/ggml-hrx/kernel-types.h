#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ggml::hrx {

enum class ResourceAccess : uint8_t {
    Read,
    Write,
    ReadWrite,
};

struct VerificationResult {
    std::vector<std::string> errors;

    bool valid() const { return errors.empty(); }
};

}  // namespace ggml::hrx
