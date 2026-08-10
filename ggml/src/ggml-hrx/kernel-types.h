#pragma once

#include "kernel-corpus-catalog.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ggml::hrx {

enum class ResourceAccess : uint8_t {
    Read,
    Write,
    ReadWrite,
};

struct KernelSpecialization {
    enum class ExecutionKind : uint8_t {
        Native,
        NativeGap,
        NativeEager,
        CpuFallback,
    };

    std::string                        family;
    std::string                        variant;
    std::map<std::string, int64_t>     integer_parameters;
    ExecutionKind                      execution_kind = ExecutionKind::Native;
    std::map<std::string, std::string> compile_parameters;
    uint64_t                           kernel_id = kUncatalogedKernelId;
};

struct VerificationResult {
    std::vector<std::string> errors;

    bool valid() const { return errors.empty(); }
};

inline std::string kernel_specialization_name(const KernelSpecialization & kernel) {
    return kernel.family + ":" + kernel.variant;
}

}  // namespace ggml::hrx
