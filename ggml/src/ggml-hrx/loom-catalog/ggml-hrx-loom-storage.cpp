#include "ggml-hrx-loom-storage.h"

#include "ggml-hrx-runtime-util.h"
#include "ggml.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

namespace {

static bool type_matches(const char * type, ggml_type actual) {
    if (!type) {
        return false;
    }
    static const struct {
        const char * name;
        ggml_type type;
    } types[] = {
        {"BF16", GGML_TYPE_BF16},
        {"F16", GGML_TYPE_F16},
        {"F32", GGML_TYPE_F32},
        {"I32", GGML_TYPE_I32},
        {"I64", GGML_TYPE_I64},
        {"Q4_K", GGML_TYPE_Q4_K},
        {"Q5_K", GGML_TYPE_Q5_K},
        {"Q6_K", GGML_TYPE_Q6_K},
        {"Q8_0", GGML_TYPE_Q8_0},
    };
    for (const auto & candidate : types) {
        if (actual == candidate.type && std::strcmp(type, candidate.name) == 0) {
            return true;
        }
    }
    return false;
}

static bool name_matches(
    const ggml_backend_hrx_loom_storage_transform_entry & transform,
    const char *                                           name) {
    if (!name || !transform.name_prefix || !transform.name_suffix) {
        return false;
    }
    const size_t name_length = std::strlen(name);
    const size_t prefix_length = std::strlen(transform.name_prefix);
    const size_t suffix_length = std::strlen(transform.name_suffix);
    if (name_length <= prefix_length + suffix_length ||
        std::memcmp(name, transform.name_prefix, prefix_length) != 0 ||
        std::memcmp(
            name + name_length - suffix_length,
            transform.name_suffix,
            suffix_length) != 0) {
        return false;
    }
    if (!transform.decimal_middle) {
        return true;
    }
    const char * middle = name + prefix_length;
    const size_t middle_length = name_length - prefix_length - suffix_length;
    if (middle_length > 1 && middle[0] == '0') {
        return false;
    }
    for (size_t i = 0; i < middle_length; ++i) {
        if (middle[i] < '0' || middle[i] > '9') {
            return false;
        }
    }
    return true;
}

static bool checked_multiply(size_t lhs, size_t rhs, size_t * result) {
    if (!result || (rhs != 0 && lhs > std::numeric_limits<size_t>::max() / rhs)) {
        return false;
    }
    *result = lhs * rhs;
    return true;
}

static bool transform_size(
    const ggml_backend_hrx_loom_storage_transform_entry & transform,
    size_t *                                               result) {
    size_t value = transform.outer_count;
    return checked_multiply(value, transform.row_count, &value) &&
           checked_multiply(value, transform.block_count, &value) &&
           checked_multiply(value, transform.field_count, &value) &&
           checked_multiply(value, transform.unit_bytes, result);
}

static bool transform_data(
    const ggml_backend_hrx_loom_storage_transform_entry * transform,
    const void *                                          source,
    size_t                                                source_size,
    void *                                                destination,
    size_t                                                destination_size,
    bool                                                  pack) {
    if (!transform || !source || !destination || source == destination ||
        transform->kind != GGML_BACKEND_HRX_LOOM_STORAGE_ROW_GROUP_FIELD_INTERLEAVE ||
        transform->row_group == 0 || transform->row_count % transform->row_group != 0 ||
        !transform->field_order) {
        return false;
    }
    size_t expected_size = 0;
    if (!transform_size(*transform, &expected_size) ||
        source_size != expected_size || destination_size != expected_size) {
        return false;
    }

    const auto * source_bytes = static_cast<const uint8_t *>(source);
    auto * destination_bytes = static_cast<uint8_t *>(destination);
    size_t packed_offset = 0;
    for (size_t outer = 0; outer < transform->outer_count; ++outer) {
        for (size_t group = 0; group < transform->row_count / transform->row_group; ++group) {
            for (size_t block = 0; block < transform->block_count; ++block) {
                for (size_t component = 0; component < transform->field_count; ++component) {
                    const size_t field = transform->field_order[component];
                    if (field >= transform->field_count) {
                        return false;
                    }
                    for (size_t lane = 0; lane < transform->row_group; ++lane) {
                        const size_t row = group * transform->row_group + lane;
                        const size_t canonical_offset =
                            ((((outer * transform->row_count + row) *
                               transform->block_count + block) *
                              transform->field_count + field) *
                             transform->unit_bytes);
                        if (pack) {
                            std::memcpy(
                                destination_bytes + packed_offset,
                                source_bytes + canonical_offset,
                                transform->unit_bytes);
                        } else {
                            std::memcpy(
                                destination_bytes + canonical_offset,
                                source_bytes + packed_offset,
                                transform->unit_bytes);
                        }
                        packed_offset += transform->unit_bytes;
                    }
                }
            }
        }
    }
    return packed_offset == expected_size;
}

}  // namespace

const ggml_backend_hrx_loom_storage_transform_entry *
ggml_backend_hrx_loom_storage_transform_match(
    const char *        architecture,
    const ggml_tensor * tensor) {
    if (!architecture || !tensor) {
        return nullptr;
    }
    const std::string target = ggml_backend_hrx_architecture_base(architecture);
    size_t count = 0;
    const auto * entries = ggml_backend_hrx_loom_storage_transform_entries(&count);
    for (size_t i = 0; i < count; ++i) {
        const auto & entry = entries[i];
        if (!entry.target || target != entry.target ||
            !type_matches(entry.type, tensor->type) ||
            (entry.contiguous && !ggml_is_contiguous(tensor)) ||
            !name_matches(entry, ggml_get_name(tensor))) {
            continue;
        }
        bool shape_matches = true;
        for (size_t axis = 0; axis < 4; ++axis) {
            shape_matches = shape_matches && tensor->ne[axis] == entry.shape[axis];
        }
        size_t expected_size = 0;
        if (shape_matches && transform_size(entry, &expected_size) &&
            ggml_nbytes(tensor) == expected_size) {
            return &entry;
        }
    }
    return nullptr;
}

size_t ggml_backend_hrx_loom_storage_transform_size(
    const ggml_backend_hrx_loom_storage_transform_entry * transform) {
    size_t result = 0;
    return transform && transform_size(*transform, &result) ? result : 0;
}

bool ggml_backend_hrx_loom_storage_transform_pack(
    const ggml_backend_hrx_loom_storage_transform_entry * transform,
    const void *                                          canonical,
    size_t                                                canonical_size,
    void *                                                packed,
    size_t                                                packed_size) {
    return transform_data(
        transform, canonical, canonical_size, packed, packed_size, true);
}

bool ggml_backend_hrx_loom_storage_transform_unpack(
    const ggml_backend_hrx_loom_storage_transform_entry * transform,
    const void *                                          packed,
    size_t                                                packed_size,
    void *                                                canonical,
    size_t                                                canonical_size) {
    return transform_data(
        transform, packed, packed_size, canonical, canonical_size, false);
}
