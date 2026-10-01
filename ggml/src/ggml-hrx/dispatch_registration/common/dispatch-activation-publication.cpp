#include "dispatch-activation-publication.h"

#include "dispatch/command-plan.h"
#include "ggml.h"

#include <cstdint>
#include <limits>
#include <vector>

namespace ggml::hrx {
namespace {

static bool is_ordered_alias_node(const Graph & graph, const GraphNode & node, const Value & input) {
    if ((node.op != GGML_OP_VIEW && node.op != GGML_OP_RESHAPE) || node.inputs.size() != 1 ||
        node.inputs.front() != input.id || !is_layout_alias_node(graph, node)) {
        return false;
    }
    const Value * output = graph.values().find(node.output);
    return output != nullptr && same_full_ordered_value_range(input, *output);
}

static bool publication_values_compatible(CommonActivationPublicationFormat format,
                                          const Value &                     source,
                                          const Value &                     candidate) {
    if (format == CommonActivationPublicationFormat::F16Row) {
        return same_full_ordered_value_range(source, candidate);
    }
    if (format == CommonActivationPublicationFormat::Q8_1X4) {
        return same_q8_1_x4_geometry(source, candidate);
    }
    return same_packed_row_geometry(source, candidate);
}

static size_t align_up(size_t value, size_t alignment) {
    return value > std::numeric_limits<size_t>::max() - (alignment - 1) ?
               0 :
               (value + alignment - 1) / alignment * alignment;
}

static size_t add_and_align(size_t value, size_t increment, size_t alignment) {
    return value > std::numeric_limits<size_t>::max() - increment ? 0 : align_up(value + increment, alignment);
}

static bool checked_element_count(const Value & value, size_t & element_count) {
    if (value.type != GGML_TYPE_F32 || !value.contiguous || value.element_count <= 0 || value.ne[0] <= 0 ||
        value.element_count % value.ne[0] != 0) {
        return false;
    }
    element_count = static_cast<size_t>(value.element_count);
    return static_cast<int64_t>(element_count) == value.element_count;
}

static bool set_publication_layout(CommonActivationPublicationFormat format,
                                   const Value &                     value,
                                   CommonActivationPublication &     publication) {
    size_t element_count = 0;
    if (!checked_element_count(value, element_count)) {
        return false;
    }

    switch (format) {
        case CommonActivationPublicationFormat::F16Row:
        case CommonActivationPublicationFormat::F16K16Major:
            if (element_count > std::numeric_limits<size_t>::max() / sizeof(ggml_fp16_t)) {
                return false;
            }
            publication.byte_count    = element_count * sizeof(ggml_fp16_t);
            publication.payload_bytes = publication.byte_count;
            return true;
        case CommonActivationPublicationFormat::Q8_1X4:
            {
                if (value.ne[0] % 128 != 0) {
                    return false;
                }
                const size_t row_count = element_count / static_cast<size_t>(value.ne[0]);
                const size_t row_bytes = ggml_row_size(GGML_TYPE_Q8_1, value.ne[0]);
                if (row_count > std::numeric_limits<size_t>::max() / row_bytes) {
                    return false;
                }
                publication.byte_count    = row_count * row_bytes;
                publication.payload_bytes = publication.byte_count;
                return publication.byte_count > 0;
            }
        case CommonActivationPublicationFormat::SymmetricI4K32:
            {
                if (value.ne[0] % 64 != 0 || element_count % 8 != 0) {
                    return false;
                }
                const size_t payload_bytes  = element_count / 2;
                const size_t metadata_bytes = element_count / 8;
                const size_t scales_offset  = align_up(payload_bytes, 256);
                const size_t sums_offset = scales_offset == 0 ? 0 : add_and_align(scales_offset, metadata_bytes, 256);
                const size_t total_bytes = sums_offset == 0 ? 0 : add_and_align(sums_offset, metadata_bytes, 256);
                if (total_bytes == 0) {
                    return false;
                }
                publication.byte_count    = total_bytes;
                publication.payload_bytes = payload_bytes;
                publication.scales_offset = scales_offset;
                publication.scales_bytes  = metadata_bytes;
                publication.sums_offset   = sums_offset;
                publication.sums_bytes    = metadata_bytes;
                return true;
            }
        case CommonActivationPublicationFormat::None:
            return false;
    }
    return false;
}

}  // namespace

CommonActivationPublicationDemand common_select_activation_publication(
    const DispatchMatchContext &                 context,
    const Value &                                produced_value,
    const CommonActivationPublicationCandidate * candidates,
    size_t                                       candidate_count) {
    if (!context.graph.has_index() || candidates == nullptr) {
        return {};
    }

    for (size_t candidate_index = 0; candidate_index < candidate_count; ++candidate_index) {
        const CommonActivationPublicationCandidate & candidate = candidates[candidate_index];
        if (!candidate.enabled || candidate.format == CommonActivationPublicationFormat::None ||
            candidate.accepts == nullptr) {
            continue;
        }

        std::vector<const Value *> values = { &produced_value };
        for (size_t value_index = 0; value_index < values.size(); ++value_index) {
            const Value & value = *values[value_index];
            for (const GraphNode * consumer : context.graph.index().consumers(value.id)) {
                if (consumer == nullptr) {
                    continue;
                }
                if (candidate.accepts(context, *consumer, value)) {
                    return { candidate.format, &value, consumer };
                }
                if (candidate.traversal != CommonActivationConsumerTraversal::FullRangeOrderedAliases ||
                    !is_ordered_alias_node(context.graph, *consumer, value)) {
                    continue;
                }
                const Value * alias = context.graph.values().find(consumer->output);
                if (!publication_values_compatible(candidate.format, produced_value, *alias)) {
                    continue;
                }
                bool seen = false;
                for (const Value * existing : values) {
                    seen = seen || existing->id == alias->id;
                }
                if (!seen) {
                    values.push_back(alias);
                }
            }
        }
    }
    return {};
}

bool common_reserve_activation_publication(const DispatchMatchContext &              context,
                                           DispatchMatch &                           match,
                                           const Value &                             produced_value,
                                           const CommonActivationPublicationDemand & demand,
                                           const char *                              transient_name,
                                           CommonActivationPublication &             publication) {
    publication = {};
    if (!demand.matched() || transient_name == nullptr ||
        !publication_values_compatible(demand.format, produced_value, *demand.consumer_value)) {
        return false;
    }

    publication.format       = demand.format;
    publication.source_value = produced_value.id;
    if (!set_publication_layout(demand.format, *demand.consumer_value, publication)) {
        publication = {};
        return false;
    }

    const size_t value_offset = match.transients.size() + match.completion_counter_requests.size();
    if (context.next_plan_value.value < 0 || value_offset > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
        context.next_plan_value.value > std::numeric_limits<int32_t>::max() - static_cast<int32_t>(value_offset)) {
        publication = {};
        return false;
    }
    publication.alternate_value = ValueId(context.next_plan_value.value + static_cast<int32_t>(value_offset));
    match.transients.push_back({ publication.alternate_value, transient_name, publication.byte_count, 256 });
    return true;
}

bool common_append_activation_publication(DispatchMatch &                     match,
                                          const CommonActivationPublication & publication,
                                          const char *                        metadata_name) {
    if (!publication.matched() || metadata_name == nullptr) {
        return false;
    }

    Status status;
    bool   recorded = false;
    switch (publication.format) {
        case CommonActivationPublicationFormat::F16Row:
            recorded = match.metadata.append_alternate_value({ publication.source_value, publication.alternate_value,
                                                               GGML_TYPE_F16, publication.byte_count, metadata_name },
                                                             status);
            break;
        case CommonActivationPublicationFormat::F16K16Major:
            recorded = match.metadata.append_generated_resource({ publication.source_value,
                                                                  GeneratedResourceRole::F16K16Major,
                                                                  publication.alternate_value,
                                                                  publication.byte_count,
                                                                  {} },
                                                                status);
            break;
        case CommonActivationPublicationFormat::Q8_1X4:
            recorded = match.metadata.append_alternate_value({ publication.source_value, publication.alternate_value,
                                                               GGML_TYPE_Q8_1, publication.byte_count, metadata_name },
                                                             status);
            break;
        case CommonActivationPublicationFormat::SymmetricI4K32:
            recorded = match.metadata.append_alternate_value({ publication.source_value, publication.alternate_value,
                                                               GGML_TYPE_COUNT, publication.byte_count, metadata_name },
                                                             status);
            break;
        case CommonActivationPublicationFormat::None:
            break;
    }
    if (!recorded) {
        match.status.append(status);
    }
    return recorded;
}

}  // namespace ggml::hrx
