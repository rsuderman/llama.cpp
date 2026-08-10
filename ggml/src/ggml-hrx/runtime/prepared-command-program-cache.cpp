#include "prepared-command-program-cache.h"

#include <optional>
#include <sstream>
#include <utility>

namespace ggml::hrx {

std::string PreparedCommandProgramCache::cache_key(uint64_t                               graph_uid,
                                                   const CommandProgramExecutionContext & context,
                                                   const std::string &                    command_shape,
                                                   const CommandProgramBindings &         bindings) const {
    const CommandProgramBindingsFingerprint binding_fingerprint = command_program_bindings_fingerprint(bindings);
    std::ostringstream                      out;
    out << "uid=" << graph_uid << "|target=" << (context.target != nullptr ? context.target : "") << '|'
        << command_shape << "|bindings=" << binding_fingerprint.value;
    return out.str();
}

bool PreparedCommandProgramCache::execute(const CommandProgramExecutionContext & context,
                                          uint64_t                               graph_uid,
                                          const std::string &                    command_shape,
                                          const CommandProgram &                 commands,
                                          const CommandProgramBindings &         bindings) {
    if (graph_uid == 0 || !commands.valid() || !bindings.valid()) {
        return execute_command_program(context, commands, bindings);
    }

    const std::string                     key = cache_key(graph_uid, context, command_shape, bindings);
    std::optional<PreparedCommandProgram> cached_prepared;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto                  found = programs_.find(key);
        if (found != programs_.end() && found->second.valid()) {
            cached_prepared = found->second;
            ++stats_.hits;
        }
    }
    if (cached_prepared.has_value()) {
        return execute_prepared_command_program(context, *cached_prepared);
    }

    PreparedCommandProgram prepared = prepare_command_program(context, commands, bindings);
    if (!prepared.valid()) {
        return execute_prepared_command_program(context, prepared);
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto                  inserted = programs_.emplace(key, prepared);
        if (inserted.second) {
            ++stats_.builds;
        }
    }
    return execute_prepared_command_program(context, prepared);
}

PreparedCommandProgramCacheStats PreparedCommandProgramCache::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void PreparedCommandProgramCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    programs_.clear();
}

}  // namespace ggml::hrx
