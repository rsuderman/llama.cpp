#pragma once

#include "reactive-plan.h"

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ggml::hrx {

enum class CommandKind : uint8_t {
    Kernel,
    Fill,
    Copy,
    Barrier,
};

enum class BindingOrigin : uint8_t {
    GraphValue,
    Transient,
};

struct KernelBindingDefinition {
    std::string name;
    ResourceAccess access = ResourceAccess::Read;
};

struct KernelScalarDefinition {
    std::string name;
    std::string type;
};

struct KernelCompileRecipe {
    std::string mode;
    std::string link_module;
    std::vector<std::string> primary_sources;
    std::vector<std::string> library_sources;
};

struct KernelDefinition {
    std::string id;
    std::string source;
    std::vector<std::string> dependencies;
    std::string symbol;
    std::string target;
    std::map<std::string, std::string> compile_config;
    std::vector<std::string> scalar_parameters;
    std::vector<KernelBindingDefinition> bindings;
    std::string source_digest;
    std::vector<KernelScalarDefinition> workload_parameters;
    std::vector<KernelScalarDefinition> launch_parameters;
    KernelCompileRecipe compile_recipe;
};

struct KernelCorpus {
    std::string schema = "ggml-hrx-kernel-corpus-v1";
    std::string upstream_revision;
    std::string corpus_digest;
    std::string recipe_digest;
    size_t plan_case_count = 0;
    std::vector<KernelDefinition> kernels;
};

struct CommandBinding {
    std::string name;
    BindingOrigin origin = BindingOrigin::GraphValue;
    ValueId value = kInvalidId;
    uint32_t transient = UINT32_MAX;
    StorageId storage = kInvalidId;
    size_t offset = 0;
    size_t length = 0;
    ResourceAccess access = ResourceAccess::Read;
};

struct Command {
    uint32_t ordinal = 0;
    CommandKind kind = CommandKind::Kernel;
    std::string label;
    std::string kernel_id;
    std::map<std::string, int64_t> scalar_parameters;
    std::map<std::string, std::string> compile_parameters;
    std::array<uint32_t, 3> workgroup_count = { 0, 0, 0 };
    std::array<uint32_t, 3> workgroup_size = { 0, 0, 0 };
    uint32_t subgroup_size = 0;
    std::vector<CommandBinding> bindings;
    std::vector<uint32_t> dependencies;
};

struct TransientAllocation {
    uint32_t id = 0;
    StorageId storage = kInvalidId;
    size_t size = 0;
    size_t alignment = 1;
    size_t arena_offset = 0;
    uint32_t first_command = UINT32_MAX;
    uint32_t last_command = 0;
};

struct TransientPlan {
    size_t arena_size = 0;
    size_t arena_alignment = 1;
    std::vector<TransientAllocation> allocations;
};

struct ConstantInitialization {
    std::string label;
    StorageId storage = kInvalidId;
    std::vector<uint8_t> data;
};

struct CommandProgram {
    std::string schema = "ggml-hrx-command-program-v1";
    std::string workload;
    std::string target;
    std::string graph_fingerprint;
    std::string recipe_revision;
    std::string corpus_digest;
    std::vector<Command> commands;
    TransientPlan transients;
    std::vector<ConstantInitialization> initializations;
    std::vector<RootContract> roots;
    std::vector<std::string> errors;

    bool valid() const { return errors.empty(); }
};

struct ConcreteBinding {
    StorageId storage = kInvalidId;
    uint64_t buffer_identity = 0;
    uint64_t generation = 0;
    size_t capacity = 0;
    size_t offset = 0;
    size_t length = 0;
};

struct BindingSnapshot {
    std::string device_identity;
    std::vector<ConcreteBinding> bindings;
};

struct AllocationFingerprint {
    std::string value;

    bool operator==(const AllocationFingerprint & other) const { return value == other.value; }
    bool operator!=(const AllocationFingerprint & other) const { return !(*this == other); }
};

CommandProgram build_command_program(const ProgramPlan & plan, const KernelCorpus & corpus);
KernelCorpus load_kernel_corpus_manifest(const std::string & path, const std::string & target,
                                         std::vector<std::string> & errors);
VerificationResult verify_kernel_corpus(const KernelCorpus & corpus);
VerificationResult verify_command_program(const ProgramPlan & plan, const KernelCorpus & corpus,
                                          const CommandProgram & commands);
VerificationResult verify_binding_snapshot(const ProgramPlan & plan, const BindingSnapshot & snapshot);
AllocationFingerprint fingerprint_bindings(const BindingSnapshot & snapshot);

const char * command_kind_name(CommandKind kind);
const char * resource_access_name(ResourceAccess access);
std::string format_kernel_corpus(const KernelCorpus & corpus);
std::string format_resource_program(const ResourceProgram & resources);
std::string format_command_program(const CommandProgram & program);
std::string format_verification_errors(const std::vector<std::string> & errors);
std::string format_verification_summary(const std::vector<std::string> & errors);
std::string format_binding_snapshot(const BindingSnapshot & snapshot, bool include_runtime_identities = false);
std::string serialize_kernel_corpus_json(const KernelCorpus & corpus);
std::string serialize_command_program_json(const CommandProgram & program);
std::string serialize_binding_snapshot_json(const BindingSnapshot & snapshot, bool include_runtime_identities = false);
std::string command_program_dot(const CommandProgram & program);

} // namespace ggml::hrx
