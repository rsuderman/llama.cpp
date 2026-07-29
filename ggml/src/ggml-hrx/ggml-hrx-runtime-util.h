#pragma once

#include "ggml-impl.h"
#include "hrx_runtime.h"

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

static inline bool ggml_backend_hrx_log_hrx_status(hrx_status_t status,
                                                   const char * expr,
                                                   const char * file,
                                                   int          line) {
    if (hrx_status_is_ok(status)) {
        return true;
    }

    char * message = nullptr;
    size_t length  = 0;
    hrx_status_to_string(status, &message, &length);
    GGML_LOG_ERROR("%s:%d: %s failed: %s\n", file, line, expr, message ? message : "unknown HRX error");
    hrx_status_free_message(message);
    hrx_status_ignore(status);
    return false;
}

static inline std::string ggml_backend_hrx_architecture_base(const char * architecture) {
    if (!architecture) {
        return std::string();
    }

    const std::string architecture_string(architecture);
    const size_t      feature_pos = architecture_string.find(':');
    return feature_pos == std::string::npos ? architecture_string : architecture_string.substr(0, feature_pos);
}

static inline bool ggml_backend_hrx_export_abi_matches(const char *                         function,
                                                       const char *                         route_id,
                                                       const hrx_executable_export_info_t & export_info,
                                                       uint32_t                             expected_binding_count,
                                                       uint32_t                             expected_parameter_count,
                                                       uint32_t expected_constant_byte_length) {
    if (export_info.binding_count == expected_binding_count &&
        export_info.parameter_count == expected_parameter_count &&
        export_info.constant_byte_length == expected_constant_byte_length) {
        return true;
    }

    GGML_LOG_ERROR(
        "%s: route %s export ABI mismatch bindings=%u expected=%u parameters=%u expected=%u constants=%u "
        "expected=%u\n",
        function, route_id, export_info.binding_count, expected_binding_count, export_info.parameter_count,
        expected_parameter_count, export_info.constant_byte_length, expected_constant_byte_length);
    return false;
}

static inline bool ggml_backend_hrx_dispatch_abi_matches(const char * function,
                                                         const char * route_id,
                                                         size_t       binding_count,
                                                         uint32_t     expected_binding_count,
                                                         size_t       constants_size,
                                                         uint32_t     expected_constant_byte_length) {
    if (binding_count == expected_binding_count && constants_size == expected_constant_byte_length) {
        return true;
    }

    GGML_LOG_ERROR("%s: route %s dispatch ABI mismatch bindings=%zu expected=%u constants=%zu expected=%u\n", function,
                   route_id, binding_count, expected_binding_count, constants_size, expected_constant_byte_length);
    return false;
}

static inline bool ggml_backend_hrx_make_work_items_dispatch_config(const uint32_t          workgroup_size[3],
                                                                    const int64_t           work_items[3],
                                                                    hrx_dispatch_config_t * out_config) {
    if (!workgroup_size || !work_items || !out_config) {
        return false;
    }

    uint32_t workgroup_count[3] = {};
    for (size_t axis = 0; axis < 3; ++axis) {
        if (workgroup_size[axis] == 0 || work_items[axis] < 0) {
            return false;
        }
        const uint64_t work_items_axis      = static_cast<uint64_t>(work_items[axis]);
        const uint64_t workgroup_size_axis  = workgroup_size[axis];
        const uint64_t workgroup_count_axis = (work_items_axis + workgroup_size_axis - 1) / workgroup_size_axis;
        if (workgroup_count_axis > std::numeric_limits<uint32_t>::max()) {
            GGML_LOG_ERROR("%s: workgroup count is too large: %" PRIu64 "\n", __func__, workgroup_count_axis);
            return false;
        }
        workgroup_count[axis] = static_cast<uint32_t>(workgroup_count_axis);
    }

    *out_config = {
        /* .workgroup_count = */ { workgroup_count[0], workgroup_count[1], workgroup_count[2] },
        /* .workgroup_size  = */
        { workgroup_size[0],  workgroup_size[1],  workgroup_size[2]  },
        /* .subgroup_size   = */
        0,
    };
    return true;
}

static inline bool ggml_backend_hrx_make_workgroup_dispatch_config(const uint32_t          workgroup_size[3],
                                                                   const int64_t           workgroups[3],
                                                                   hrx_dispatch_config_t * out_config) {
    if (!workgroup_size || !workgroups || !out_config) {
        return false;
    }

    uint32_t workgroup_count[3] = {};
    for (size_t axis = 0; axis < 3; ++axis) {
        if (workgroup_size[axis] == 0 || workgroups[axis] < 0) {
            return false;
        }
        if (static_cast<uint64_t>(workgroups[axis]) > std::numeric_limits<uint32_t>::max()) {
            GGML_LOG_ERROR("%s: workgroup count is too large: %" PRIu64 "\n", __func__,
                           static_cast<uint64_t>(workgroups[axis]));
            return false;
        }
        workgroup_count[axis] = static_cast<uint32_t>(workgroups[axis]);
    }

    *out_config = {
        /* .workgroup_count = */ { workgroup_count[0], workgroup_count[1], workgroup_count[2] },
        /* .workgroup_size  = */
        { workgroup_size[0],  workgroup_size[1],  workgroup_size[2]  },
        /* .subgroup_size   = */
        0,
    };
    return true;
}
