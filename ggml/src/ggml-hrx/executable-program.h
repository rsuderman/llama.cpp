#pragma once

#include "graph/command-program.h"

#include "hrx_runtime.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ggml::hrx {

struct PackedKernelConstants {
    std::vector<uint8_t> bytes;
    std::vector<std::string> errors;

    bool valid() const { return errors.empty(); }
};

PackedKernelConstants pack_kernel_constants(const KernelDefinition & definition,
                                             const Command & command);
std::string kernel_artifact_key(const KernelDefinition & definition,
                                const Command & command);

struct PreparedArtifactDiagnostic {
    std::string key;
    std::string kernel_id;
    std::array<uint32_t, 3> workgroup_count = { 0, 0, 0 };
    std::array<uint32_t, 3> workgroup_size = { 0, 0, 0 };
    uint32_t subgroup_size = 0;
    size_t constant_bytes = 0;
    size_t binding_count = 0;
    std::string manifest_json;
    std::string compile_report_json;
    std::string final_module_text;
};

struct PreparedCommandDiagnostic {
    uint32_t ordinal = 0;
    CommandKind kind = CommandKind::Kernel;
    std::string label;
    std::string artifact_key;
    size_t constant_bytes = 0;
    size_t binding_count = 0;
};

struct ExecutablePreparationOptions {
    std::string corpus_directory;
    std::string target;
    size_t dummy_buffer_limit = 256ull * 1024ull * 1024ull;
};

class PreparedExecutableProgram {
public:
    PreparedExecutableProgram();
    ~PreparedExecutableProgram();
    PreparedExecutableProgram(PreparedExecutableProgram &&) noexcept;
    PreparedExecutableProgram & operator=(PreparedExecutableProgram &&) noexcept;
    PreparedExecutableProgram(const PreparedExecutableProgram &) = delete;
    PreparedExecutableProgram & operator=(const PreparedExecutableProgram &) = delete;

    bool valid() const;
    size_t node_count() const;
    size_t artifact_count() const;
    const std::vector<std::string> & errors() const;
    const std::vector<PreparedArtifactDiagnostic> & artifacts() const;
    const std::vector<PreparedCommandDiagnostic> & commands() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend PreparedExecutableProgram prepare_executable_program(
        hrx_device_t, hrx_stream_t, const ProgramPlan &, const KernelCorpus &,
        const CommandProgram &, const ExecutablePreparationOptions &);
};

PreparedExecutableProgram prepare_executable_program(
    hrx_device_t device, hrx_stream_t stream, const ProgramPlan & plan,
    const KernelCorpus & corpus, const CommandProgram & commands,
    const ExecutablePreparationOptions & options);

std::string format_prepared_executable_program(const PreparedExecutableProgram & program);
std::string serialize_prepared_executable_program_json(const PreparedExecutableProgram & program);

} // namespace ggml::hrx
