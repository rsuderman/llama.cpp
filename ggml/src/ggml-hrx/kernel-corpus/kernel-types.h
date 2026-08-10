#pragma once

#include "error-log.h"

#include <cstdint>
#include <string>

namespace ggml::hrx {

enum class ResourceAccess : uint8_t {
    Read,
    Write,
    ReadWrite,
};

struct VerificationResult {
    ErrorLog errors;

    bool valid() const { return errors.success(); }
};

}  // namespace ggml::hrx
