#pragma once

#include "ggml-impl.h"
#include "hrx_runtime.h"

#include <cstddef>
#include <cstdint>
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
