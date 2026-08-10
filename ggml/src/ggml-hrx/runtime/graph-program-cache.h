#pragma once

#include "dispatch/command-program.h"
#include "error-log.h"
#include "graph/graph.h"
#include "kernel-corpus/kernel-corpus.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_cgraph;
struct ggml_tensor;

namespace ggml::hrx {

struct GraphProgramExternalBinding {
    ValueId             value;
    const ggml_tensor * tensor = nullptr;
};

struct GraphProgramMatch {
    std::vector<GraphProgramExternalBinding> external_bindings;
    ErrorLog                                 errors;

    bool valid() const { return errors.success(); }
};

class GraphProgram {
  public:
    GraphProgram(uint64_t                        uid,
                 std::string                     target,
                 std::unique_ptr<Graph>          graph,
                 std::unique_ptr<CommandProgram> commands,
                 std::string                     command_shape);

    uint64_t uid() const { return uid_; }

    const std::string & target() const { return target_; }

    const std::string & command_shape() const { return command_shape_; }

    const Graph & graph() const { return *graph_; }

    Graph & graph() { return *graph_; }

    const CommandProgram & commands() const { return *commands_; }

    CommandProgram & commands() { return *commands_; }

    GraphProgramMatch match_current_graph(const ggml_cgraph & graph) const;

  private:
    uint64_t                        uid_ = 0;
    std::string                     target_;
    std::unique_ptr<Graph>          graph_;
    std::unique_ptr<CommandProgram> commands_;
    std::string                     command_shape_;
};

struct GraphProgramCacheStats {
    uint64_t builds = 0;
    uint64_t hits   = 0;
};

struct GraphProgramLookup {
    GraphProgram *                program = nullptr;
    std::unique_ptr<GraphProgram> uncached_program;
    GraphProgramMatch             match;
    ErrorLog                      errors;

    bool valid() const { return program != nullptr && errors.success() && match.valid(); }
};

class GraphProgramCache {
  public:
    bool can_execute(const ggml_cgraph & graph, const KernelCorpus & corpus, const std::string & target) const;

    GraphProgramLookup get_or_build(const ggml_cgraph & graph, const KernelCorpus & corpus, const std::string & target);

    GraphProgramCacheStats stats() const;

    void clear();

  private:
    std::unique_ptr<GraphProgram> build_program(const ggml_cgraph &  graph,
                                                const KernelCorpus & corpus,
                                                const std::string &  target,
                                                ErrorLog &           errors) const;

    mutable std::mutex                                          mutex_;
    std::unordered_map<uint64_t, std::unique_ptr<GraphProgram>> programs_;
    GraphProgramCacheStats                                      stats_;
};

bool can_execute_standalone_op_as_graph(const ggml_tensor * op);

}  // namespace ggml::hrx
