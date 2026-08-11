#include "prepared-command-program-cache.h"

#include <memory>
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
    return execute_with_result(context, graph_uid, command_shape, commands, bindings).success;
}

PreparedCommandProgramCacheExecutionResult PreparedCommandProgramCache::execute_with_result(
    const CommandProgramExecutionContext & context,
    uint64_t                               graph_uid,
    const std::string &                    command_shape,
    const CommandProgram &                 commands,
    const CommandProgramBindings &         bindings) {
    PreparedCommandProgramCacheExecutionResult result;
    if (graph_uid == 0 || !commands.valid() || !bindings.valid()) {
        PreparedCommandProgram prepared = prepare_command_program(context, commands, bindings);
        if (!prepared.valid()) {
            result.errors.append(prepared.errors);
            return result;
        }
        result.success = bind_and_execute_prepared_command_program(context, commands, prepared);
        if (!result.success) {
            result.errors.log("execute uncached HRX command program failed");
        }
        return result;
    }

    const std::string      key = cache_key(graph_uid, context, command_shape, bindings);
    std::shared_ptr<Entry> entry;
    bool                   created_entry = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        found = programs_.find(key);
        if (found == programs_.end()) {
            entry = std::make_shared<Entry>();
            programs_.emplace(key, entry);
            created_entry = true;
        } else {
            entry = found->second;
        }
    }

    std::lock_guard<std::mutex> entry_lock(entry->mutex);
    if (entry->has_program && entry->program.valid()) {
        record_hit();
        result.success = bind_and_execute_prepared_command_program(context, commands, entry->program);
        if (!result.success) {
            result.errors.log("execute cached HRX command program failed");
        }
        return result;
    }

    PreparedCommandProgram prepared = prepare_command_program(context, commands, bindings);
    if (!prepared.valid()) {
        if (created_entry) {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto                  found = programs_.find(key);
            if (found != programs_.end() && found->second == entry && !entry->has_program) {
                programs_.erase(found);
            }
        }
        result.errors.append(prepared.errors);
        return result;
    }
    entry->program     = std::move(prepared);
    entry->has_program = true;
    record_build();

    result.success = bind_and_execute_prepared_command_program(context, commands, entry->program);
    if (!result.success) {
        result.errors.log("execute prepared HRX command program failed");
    }
    return result;
}

PreparedCommandProgramCacheStats PreparedCommandProgramCache::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void PreparedCommandProgramCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    programs_.clear();
}

void PreparedCommandProgramCache::record_build() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++stats_.builds;
}

void PreparedCommandProgramCache::record_hit() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++stats_.hits;
}

}  // namespace ggml::hrx
