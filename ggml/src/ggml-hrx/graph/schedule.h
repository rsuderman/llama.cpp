#pragma once

#include "graph-ir.h"

#include <map>
#include <string>
#include <vector>

namespace ggml::hrx {

struct KernelSpecialization {
    enum class ExecutionKind : uint8_t {
        Native,
        CpuFallback,
    };

    std::string family;
    std::string variant;
    std::map<std::string, int64_t> integer_parameters;
    ExecutionKind execution_kind = ExecutionKind::Native;
};

struct TensorBinding {
    std::string role;
    ValueId value = kInvalidId;
};

struct Dispatch {
    KernelSpecialization kernel;
    std::vector<TensorBinding> bindings;
    std::vector<uint32_t> dependencies;
};

struct Invocation {
    KernelSpecialization kernel;
    std::vector<OperationId> covered_operations;
    std::vector<TensorBinding> inputs;
    std::vector<TensorBinding> outputs;
    std::vector<Dispatch> dispatches;
};

struct Schedule {
    std::string graph_fingerprint;
    std::string workload;
    size_t expected_dispatch_count = 0;
    std::vector<Invocation> invocations;
};

struct VerificationResult {
    std::vector<std::string> errors;

    bool valid() const { return errors.empty(); }
};

VerificationResult verify_schedule(const Graph & graph, const Schedule & schedule);
Schedule deserialize_schedule_json(const std::string & json, std::vector<std::string> & errors);
std::string serialize_schedule_json(const Schedule & schedule);
size_t schedule_dispatch_count(const Schedule & schedule);

} // namespace ggml::hrx
