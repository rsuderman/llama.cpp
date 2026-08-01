#pragma once

#include "ggml-hrx-loom-catalog-runtime.h"
#include "ggml-hrx-loom-catalog.h"
#include "ggml-hrx-runtime-util.h"
#include "ggml-impl.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

static constexpr size_t GGML_BACKEND_HRX_LOOM_MAX_BINDINGS        = 8;
static constexpr size_t GGML_BACKEND_HRX_LOOM_MAX_CONSTANTS_SIZE  = 256;
static constexpr size_t GGML_BACKEND_HRX_LOOM_MAX_CONFIG_BINDINGS = 64;
static constexpr size_t GGML_BACKEND_HRX_LOOM_CONFIG_NAME_BYTES   = 64;
static constexpr size_t GGML_BACKEND_HRX_LOOM_CONFIG_VALUE_BYTES  = 128;
static constexpr size_t GGML_BACKEND_HRX_LOOM_MAX_DISPATCHES      = 4;
static constexpr size_t GGML_BACKEND_HRX_LOOM_MAX_TRANSIENTS      = 4;
static constexpr size_t GGML_BACKEND_HRX_LOOM_TRANSIENT_ALIGNMENT = 256;

struct ggml_backend_hrx_loom_config_binding {
    char         name[GGML_BACKEND_HRX_LOOM_CONFIG_NAME_BYTES];
    char         value[GGML_BACKEND_HRX_LOOM_CONFIG_VALUE_BYTES];
    const char * type;
};

struct ggml_backend_hrx_loom_transient_buffer_plan {
    size_t offset = 0;
    size_t size   = 0;
};

struct ggml_backend_hrx_loom_dispatch_plan {
    const ggml_backend_hrx_loom_catalog_entry * entry                                                         = nullptr;
    hrx_dispatch_config_t                       dispatch                                                      = {};
    hrx_buffer_ref_t                            bindings[GGML_BACKEND_HRX_LOOM_MAX_BINDINGS]                  = {};
    int                                         transient_binding_indices[GGML_BACKEND_HRX_LOOM_MAX_BINDINGS] = {};
    size_t                                      binding_count                                                 = 0;
    uint8_t                                     constants[GGML_BACKEND_HRX_LOOM_MAX_CONSTANTS_SIZE]           = {};
    size_t                                      constants_size                                                = 0;
    ggml_backend_hrx_loom_config_binding        config_bindings[GGML_BACKEND_HRX_LOOM_MAX_CONFIG_BINDINGS]    = {};
    size_t                                      config_binding_count                                          = 0;
};

struct ggml_backend_hrx_loom_execution_plan {
    const char *                                route_id                                                   = nullptr;
    int                                         consumed_node_indices[GGML_BACKEND_HRX_LOOM_MAX_CONSUMED_NODES] = {};
    int                                         consumed_node_count                                             = 0;
    ggml_backend_hrx_loom_transient_buffer_plan transients[GGML_BACKEND_HRX_LOOM_MAX_TRANSIENTS]                = {};
    size_t                                      transient_count                                                 = 0;
    size_t                                      transient_byte_length                                           = 0;
    ggml_backend_hrx_loom_dispatch_plan         dispatches[GGML_BACKEND_HRX_LOOM_MAX_DISPATCHES]                = {};
    size_t                                      dispatch_count                                                  = 0;
};

static inline bool ggml_backend_hrx_loom_match_only(const ggml_backend_hrx_loom_execution_plan * plan) {
    return plan == nullptr;
}

static inline bool ggml_backend_hrx_loom_checked_add_size(size_t lhs, size_t rhs, size_t * out) {
    if (!out || lhs > std::numeric_limits<size_t>::max() - rhs) {
        return false;
    }
    *out = lhs + rhs;
    return true;
}

static inline bool ggml_backend_hrx_loom_align_size(size_t value, size_t alignment, size_t * out) {
    if (!out || alignment == 0) {
        return false;
    }
    const size_t remainder = value % alignment;
    if (remainder == 0) {
        *out = value;
        return true;
    }
    return ggml_backend_hrx_loom_checked_add_size(value, alignment - remainder, out);
}

struct ggml_backend_hrx_loom_compile_input {
    const void *                                 source_data;
    size_t                                       source_size;
    const char *                                 source_format;
    const char *                                 source_name;
    const char *                                 target;
    const char *                                 symbol;
    const ggml_backend_hrx_loom_source_entry *   dependencies;
    size_t                                       dependency_count;
    const ggml_backend_hrx_loom_config_binding * config_bindings;
    size_t                                       config_binding_count;
};

struct ggml_backend_hrx_loom_compile_output {
    void * executable_data;
    size_t executable_size;
    char * report_json;
    size_t report_json_size;
};

static inline uint64_t ggml_backend_hrx_loom_fnv1a64(const void * data, size_t size) {
    static constexpr uint64_t FNV_OFFSET_BASIS = UINT64_C(14695981039346656037);
    static constexpr uint64_t FNV_PRIME        = UINT64_C(1099511628211);

    uint64_t     result = FNV_OFFSET_BASIS;
    const auto * bytes  = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < size; ++i) {
        result ^= bytes[i];
        result *= FNV_PRIME;
    }
    return result;
}

static inline void ggml_backend_hrx_loom_append_key_field(std::string & key, const char * data, size_t size) {
    key += std::to_string(size);
    key += ':';
    if (data && size > 0) {
        key.append(data, size);
    }
}

static inline void ggml_backend_hrx_loom_append_key_field(std::string & key, const char * value) {
    if (!value) {
        ggml_backend_hrx_loom_append_key_field(key, "", 0);
        return;
    }
    ggml_backend_hrx_loom_append_key_field(key, value, std::strlen(value));
}

static inline void ggml_backend_hrx_loom_append_key_field(std::string & key, uint64_t value) {
    const std::string text = std::to_string(value);
    ggml_backend_hrx_loom_append_key_field(key, text.c_str(), text.size());
}

static inline std::string ggml_backend_hrx_loom_cache_key(const char *                                 target,
                                                          const ggml_backend_hrx_loom_execution_plan * plan) {
    std::string                                 key;
    const ggml_backend_hrx_loom_dispatch_plan * dispatch = plan && plan->dispatch_count > 0 ? &plan->dispatches[0] : nullptr;
    const ggml_backend_hrx_loom_catalog_entry * entry    = dispatch ? dispatch->entry : nullptr;
    if (!entry) {
        return key;
    }

    ggml_backend_hrx_loom_append_key_field(key, target);
    ggml_backend_hrx_loom_append_key_field(key, entry->id);
    ggml_backend_hrx_loom_append_key_field(key, entry->source_name);
    ggml_backend_hrx_loom_append_key_field(key, entry->source_format);
    ggml_backend_hrx_loom_append_key_field(key, static_cast<uint64_t>(entry->source_size));
    ggml_backend_hrx_loom_append_key_field(key, ggml_backend_hrx_loom_fnv1a64(entry->source_data, entry->source_size));
    ggml_backend_hrx_loom_append_key_field(key, entry->symbol);
    ggml_backend_hrx_loom_append_key_field(key, static_cast<uint64_t>(entry->dependency_count));
    for (size_t i = 0; i < entry->dependency_count; ++i) {
        const ggml_backend_hrx_loom_source_entry & dependency = entry->dependencies[i];
        ggml_backend_hrx_loom_append_key_field(key, dependency.name);
        ggml_backend_hrx_loom_append_key_field(key, dependency.format);
        ggml_backend_hrx_loom_append_key_field(key, static_cast<uint64_t>(dependency.size));
        ggml_backend_hrx_loom_append_key_field(key, ggml_backend_hrx_loom_fnv1a64(dependency.data, dependency.size));
    }

    for (size_t i = 0; i < dispatch->config_binding_count; ++i) {
        const ggml_backend_hrx_loom_config_binding & binding = dispatch->config_bindings[i];
        ggml_backend_hrx_loom_append_key_field(key, binding.name);
        ggml_backend_hrx_loom_append_key_field(key, binding.type);
        ggml_backend_hrx_loom_append_key_field(key, binding.value);
    }
    return key;
}

bool ggml_backend_hrx_loom_compile(const ggml_backend_hrx_loom_compile_input * input,
                                   ggml_backend_hrx_loom_compile_output *      output);

ggml_backend_hrx_loom_op_response ggml_backend_hrx_loom_unsupported(ggml_backend_hrx_loom_unsupported_reason reason);

ggml_backend_hrx_loom_op_response ggml_backend_hrx_loom_supported(const char * route_id);

ggml_backend_hrx_loom_op_response ggml_backend_hrx_loom_failed(const char * route_id);

const ggml_backend_hrx_loom_catalog_entry * ggml_backend_hrx_loom_find_entry(
    const ggml_backend_hrx_loom_catalog * catalog,
    const char *                          route_id);

int64_t ggml_backend_hrx_loom_next_power_of_2(int64_t value);

bool ggml_backend_hrx_loom_bind_tensor(const ggml_backend_hrx_loom_op_request * request,
                                       const ggml_tensor *                      tensor,
                                       hrx_buffer_ref_t *                       out_ref);

ggml_backend_hrx_loom_op_response ggml_backend_hrx_loom_match_request(ggml_backend_hrx_loom_catalog *          catalog,
                                                                      const ggml_backend_hrx_loom_op_request * request);

ggml_backend_hrx_loom_op_response ggml_backend_hrx_loom_prepare_plan(ggml_backend_hrx_loom_catalog *          catalog,
                                                                     const ggml_backend_hrx_loom_op_request * request,
                                                                     ggml_backend_hrx_loom_execution_plan *   plan);

bool ggml_backend_hrx_loom_dispatch_prepared(ggml_backend_hrx_loom_catalog *        catalog,
                                             hrx_stream_t                           stream,
                                             ggml_backend_hrx_loom_execution_plan * plan,
                                             hrx_buffer_t                           transient_buffer,
                                             size_t                                 transient_buffer_size);

ggml_backend_hrx_loom_op_response ggml_backend_hrx_loom_match_or_prepare_request(
    ggml_backend_hrx_loom_catalog *          catalog,
    const ggml_backend_hrx_loom_op_request * request,
    ggml_backend_hrx_loom_execution_plan *   plan);
