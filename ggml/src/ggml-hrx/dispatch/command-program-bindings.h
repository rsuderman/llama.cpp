#pragma once

#include "error-log.h"
#include "graph/value-map.h"

#include <cstddef>
#include <vector>

namespace ggml::hrx {

struct CommandProgramBinding {
    ValueId      value;
    hrx_buffer_t buffer = nullptr;
    size_t       offset = 0;
    size_t       length = 0;
};

class CommandProgramBindings {
  public:
    static CommandProgramBindings from_value_map(const ValueMap & values);

    const CommandProgramBinding * find(ValueId value) const;

    const std::vector<CommandProgramBinding> & bindings() const { return bindings_; }

    bool valid() const { return errors.success(); }

    ErrorLog errors;

  private:
    std::vector<CommandProgramBinding> bindings_;
};

}  // namespace ggml::hrx
