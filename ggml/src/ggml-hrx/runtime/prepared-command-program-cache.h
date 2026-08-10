#pragma once

#include "command-program-executor.h"
#include "dispatch/command-program-bindings.h"
#include "dispatch/command-program.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

namespace ggml::hrx {

struct PreparedCommandProgramCacheStats {
    uint64_t builds = 0;
    uint64_t hits   = 0;
};

class PreparedCommandProgramCache {
  public:
    bool execute(const CommandProgramExecutionContext & context,
                 uint64_t                               graph_uid,
                 const std::string &                    command_shape,
                 const CommandProgram &                 commands,
                 const CommandProgramBindings &         bindings);

    PreparedCommandProgramCacheStats stats() const;

    void clear();

  private:
    std::string cache_key(uint64_t                               graph_uid,
                          const CommandProgramExecutionContext & context,
                          const std::string &                    command_shape,
                          const CommandProgramBindings &         bindings) const;

    mutable std::mutex                                      mutex_;
    std::unordered_map<std::string, PreparedCommandProgram> programs_;
    PreparedCommandProgramCacheStats                        stats_;
};

}  // namespace ggml::hrx
