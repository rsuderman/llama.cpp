#pragma once

#include "error-log.h"
#include "graph/value-map.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ggml::hrx {

struct CommandProgramBinding {
    ValueId      value;
    hrx_buffer_t buffer     = nullptr;
    size_t       offset     = 0;
    size_t       length     = 0;
    uint64_t     identity   = 0;
    uint64_t     generation = 0;
    size_t       capacity   = 0;
};

struct CommandProgramBindingsFingerprint {
    std::string value;
};

class CommandProgramBindings {
  public:
    static CommandProgramBindings from_value_map(const ValueMap & values);
    static CommandProgramBindings from_bindings(std::vector<CommandProgramBinding> bindings,
                                                const ErrorLog &                   errors = {});

    const CommandProgramBinding * find(ValueId value) const;

    const std::vector<CommandProgramBinding> & bindings() const { return bindings_; }

    bool valid() const { return errors.success(); }

    ErrorLog errors;

  private:
    std::vector<CommandProgramBinding> bindings_;
};

CommandProgramBindingsFingerprint command_program_bindings_fingerprint(const CommandProgramBindings & bindings);

}  // namespace ggml::hrx
