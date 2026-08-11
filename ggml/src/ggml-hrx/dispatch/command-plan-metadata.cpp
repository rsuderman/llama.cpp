#include "command-plan-metadata.h"

#include <cstring>
#include <utility>

namespace ggml::hrx {
namespace {

static bool metadata_matches(const CommandPlanResourceMetadata & lhs, const CommandPlanResourceMetadata & rhs) {
    return lhs.kind == rhs.kind && lhs.size == rhs.size &&
           std::memcmp(lhs.bytes.data(), rhs.bytes.data(), lhs.size) == 0;
}

static bool generated_resource_matches(const CommandPlanGeneratedResource & lhs,
                                       const CommandPlanGeneratedResource & rhs) {
    return lhs.source_value == rhs.source_value && lhs.role == rhs.role && lhs.generated_value == rhs.generated_value &&
           lhs.byte_count == rhs.byte_count && metadata_matches(lhs.metadata, rhs.metadata);
}

static bool alternate_value_matches(const CommandPlanAlternateValue & lhs, const CommandPlanAlternateValue & rhs) {
    return lhs.graph_value == rhs.graph_value && lhs.alternate_value == rhs.alternate_value && lhs.type == rhs.type &&
           lhs.byte_count == rhs.byte_count && lhs.name == rhs.name;
}

}  // namespace

void CommandPlanMetadata::clear() {
    generated_resources_.clear();
    alternate_values_.clear();
}

bool CommandPlanMetadata::append(CommandPlanMetadata && other, Status & status) {
    for (CommandPlanGeneratedResource & resource : other.generated_resources_) {
        if (!append_generated_resource(std::move(resource), status)) {
            return false;
        }
    }
    for (CommandPlanAlternateValue & alternate : other.alternate_values_) {
        if (!append_alternate_value(std::move(alternate), status)) {
            return false;
        }
    }
    return true;
}

bool CommandPlanMetadata::append_generated_resource(CommandPlanGeneratedResource resource, Status & status) {
    for (const CommandPlanGeneratedResource & existing : generated_resources_) {
        if (existing.source_value == resource.source_value && existing.role == resource.role) {
            if (generated_resource_matches(existing, resource)) {
                return true;
            }
            status.log("conflicting generated resource for source value %d role %d", resource.source_value.value,
                       static_cast<int>(resource.role));
            return false;
        }
    }
    generated_resources_.push_back(std::move(resource));
    return true;
}

bool CommandPlanMetadata::append_alternate_value(CommandPlanAlternateValue alternate, Status & status) {
    for (const CommandPlanAlternateValue & existing : alternate_values_) {
        if (existing.graph_value == alternate.graph_value) {
            if (alternate_value_matches(existing, alternate)) {
                return true;
            }
            status.log("conflicting alternate value for graph value %d", alternate.graph_value.value);
            return false;
        }
    }
    alternate_values_.push_back(std::move(alternate));
    return true;
}

const CommandPlanGeneratedResource * CommandPlanMetadata::find_generated_resource(ValueId               source_value,
                                                                                  GeneratedResourceRole role) const {
    for (const CommandPlanGeneratedResource & resource : generated_resources_) {
        if (resource.source_value == source_value && resource.role == role) {
            return &resource;
        }
    }
    return nullptr;
}

const CommandPlanAlternateValue * CommandPlanMetadata::find_alternate_value(ValueId graph_value) const {
    for (const CommandPlanAlternateValue & alternate : alternate_values_) {
        if (alternate.graph_value == graph_value) {
            return &alternate;
        }
    }
    return nullptr;
}

}  // namespace ggml::hrx
