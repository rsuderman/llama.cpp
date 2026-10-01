#pragma once

#include "../dispatch-registry.h"

#include <cstddef>

namespace ggml::hrx {

enum class CommonActivationPublicationFormat {
    None,
    F16Row,
    F16K16Major,
    Q8_1X4,
    SymmetricI4K32,
};

enum class CommonActivationConsumerTraversal {
    Direct,
    FullRangeOrderedAliases,
};

using CommonActivationConsumerPredicate = bool (*)(const DispatchMatchContext & context,
                                                   const GraphNode &            consumer,
                                                   const Value &                input);

struct CommonActivationPublicationCandidate {
    CommonActivationPublicationFormat format    = CommonActivationPublicationFormat::None;
    CommonActivationConsumerPredicate accepts   = nullptr;
    CommonActivationConsumerTraversal traversal = CommonActivationConsumerTraversal::Direct;
    bool                              enabled   = true;
};

struct CommonActivationPublicationDemand {
    CommonActivationPublicationFormat format         = CommonActivationPublicationFormat::None;
    const Value *                     consumer_value = nullptr;
    const GraphNode *                 consumer       = nullptr;

    bool matched() const {
        return format != CommonActivationPublicationFormat::None && consumer_value != nullptr && consumer != nullptr;
    }
};

struct CommonActivationPublication {
    CommonActivationPublicationFormat format = CommonActivationPublicationFormat::None;
    ValueId                           source_value;
    ValueId                           alternate_value;
    size_t                            byte_count    = 0;
    size_t                            payload_bytes = 0;
    size_t                            scales_offset = 0;
    size_t                            scales_bytes  = 0;
    size_t                            sums_offset   = 0;
    size_t                            sums_bytes    = 0;

    bool matched() const {
        return format != CommonActivationPublicationFormat::None && source_value.value >= 0 &&
               alternate_value.value >= 0 && byte_count > 0;
    }

    DispatchBinding binding() const { return { alternate_value, 0, byte_count }; }

    DispatchBinding binding(size_t offset, size_t length) const { return { alternate_value, offset, length }; }
};

CommonActivationPublicationDemand common_select_activation_publication(
    const DispatchMatchContext &                 context,
    const Value &                                produced_value,
    const CommonActivationPublicationCandidate * candidates,
    size_t                                       candidate_count);

bool common_reserve_activation_publication(const DispatchMatchContext &              context,
                                           DispatchMatch &                           match,
                                           const Value &                             produced_value,
                                           const CommonActivationPublicationDemand & demand,
                                           const char *                              transient_name,
                                           CommonActivationPublication &             publication);

// A metadata conflict is appended to match.status and invalidates the entire match; callers must not fall back.
bool common_append_activation_publication(DispatchMatch &                     match,
                                          const CommonActivationPublication & publication,
                                          const char *                        metadata_name);

}  // namespace ggml::hrx
