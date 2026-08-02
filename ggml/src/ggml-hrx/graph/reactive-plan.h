#pragma once

#include "graph-ir.h"
#include "schedule.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace ggml::hrx {

enum class ResourceAccess : uint8_t {
    Read,
    Write,
    ReadWrite,
};

struct ResourceUse {
    uint32_t invocation = 0;
    StorageId storage = kInvalidId;
    uint32_t before_version = 0;
    uint32_t after_version = 0;
    ResourceAccess access = ResourceAccess::Read;
};

struct ResourceContract {
    StorageId storage = kInvalidId;
    size_t size = 0;
    bool imported = false;
    bool weight = false;
    bool mutable_state = false;
    bool exported = false;
    bool elidable = false;
    uint32_t final_version = 0;
    uint32_t first_invocation = UINT32_MAX;
    uint32_t last_invocation = 0;
    std::vector<ValueId> aliases;
};

struct ResourceProgram {
    std::vector<ResourceContract> resources;
    std::vector<ResourceUse> uses;
};

struct ProgramPlan {
    Graph graph;
    Schedule schedule;
    ResourceProgram resources;
    std::string semantic_witness;
    std::string target;
    std::vector<std::string> errors;

    bool valid() const { return errors.empty(); }
};

struct ExecutionFrame {
    std::shared_ptr<const ProgramPlan> plan;
    std::vector<const ggml_tensor *> values;
    std::vector<const ggml_tensor *> storage_roots;
    std::vector<std::string> errors;

    bool valid() const { return errors.empty() && plan != nullptr && plan->valid(); }
};

struct PlanCacheStats {
    uint64_t builds = 0;
    uint64_t hits = 0;
    uint64_t semantic_collisions = 0;
    uint64_t failures = 0;
};

bool eager_capability_declared(enum ggml_op op);
ResourceProgram build_resource_program(const Graph & graph, const Schedule & schedule);
VerificationResult verify_resource_program(const Graph & graph, const Schedule & schedule, const ResourceProgram & resources);
bool graph_semantically_equal(const Graph & lhs, const Graph & rhs);
std::string schedule_semantic_witness(const Graph & graph, const Schedule & schedule);
ProgramPlan build_reactive_plan(const Graph & graph, const std::string & target);

class ReactivePlanCache {
public:
    ExecutionFrame prepare(const ggml_cgraph * graph, const std::string & target);
    PlanCacheStats stats() const;

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::vector<std::shared_ptr<const ProgramPlan>>> plans_;
    PlanCacheStats stats_;
};

} // namespace ggml::hrx
