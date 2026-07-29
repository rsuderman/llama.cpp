#include "ggml-hrx-hsaco-catalog-runtime-internal.h"
#include "ggml-hrx-runtime-util.h"

#include <cinttypes>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <vector>

namespace {

struct ggml_backend_hrx_loaded_hsaco_route {
    const ggml_backend_hrx_hsaco_catalog_entry * entry          = nullptr;
    hrx_executable_t                             executable     = nullptr;
    uint32_t                                     export_ordinal = 0;
    hrx_executable_export_info_t                 export_info    = {};

    ~ggml_backend_hrx_loaded_hsaco_route() {
        if (executable) {
            hrx_executable_release(executable);
        }
    }
};

#define GGML_HRX_HSACO_CHECK(expr) ggml_backend_hrx_log_hrx_status((expr), #expr, __FILE__, __LINE__)

static ggml_backend_hrx_hsaco_op_response ggml_backend_hrx_hsaco_response(
    ggml_backend_hrx_hsaco_result             result,
    ggml_backend_hrx_hsaco_unsupported_reason unsupported_reason,
    const char *                              route_id) {
    return {
        /* .result             = */ result,
        /* .unsupported_reason = */ unsupported_reason,
        /* .route_id           = */ route_id,
    };
}

}  // namespace

ggml_backend_hrx_hsaco_op_response ggml_backend_hrx_hsaco_unsupported(
    ggml_backend_hrx_hsaco_unsupported_reason reason) {
    return ggml_backend_hrx_hsaco_response(GGML_BACKEND_HRX_HSACO_UNSUPPORTED, reason, nullptr);
}

ggml_backend_hrx_hsaco_op_response ggml_backend_hrx_hsaco_supported(const char * route_id) {
    return ggml_backend_hrx_hsaco_response(GGML_BACKEND_HRX_HSACO_INVOKED, GGML_BACKEND_HRX_HSACO_UNSUPPORTED_NONE,
                                           route_id);
}

ggml_backend_hrx_hsaco_op_response ggml_backend_hrx_hsaco_failed(const char * route_id) {
    return ggml_backend_hrx_hsaco_response(GGML_BACKEND_HRX_HSACO_FAILED, GGML_BACKEND_HRX_HSACO_UNSUPPORTED_NONE,
                                           route_id);
}

bool ggml_backend_hrx_hsaco_make_1d_dispatch_config(const ggml_backend_hrx_hsaco_catalog_entry * entry,
                                                    int64_t                                      nelements,
                                                    hrx_dispatch_config_t *                      out_config) {
    if (!entry || !out_config || entry->threads_per_block == 0 || nelements < 0) {
        return false;
    }

    const uint64_t threads_per_block = entry->threads_per_block;
    const uint64_t workgroup_count   = (static_cast<uint64_t>(nelements) + threads_per_block - 1) / threads_per_block;
    if (workgroup_count > std::numeric_limits<uint32_t>::max()) {
        GGML_LOG_ERROR("%s: workgroup count is too large: %" PRIu64 "\n", __func__, workgroup_count);
        return false;
    }

    *out_config = {
        /* .workgroup_count = */ { static_cast<uint32_t>(workgroup_count), 1, 1 },
        /* .workgroup_size = */
        {
                                  entry->workgroup_size[0],
                                  entry->workgroup_size[1],
                                  entry->workgroup_size[2],
                                  },
        /* .subgroup_size = */
        0,
    };
    return true;
}

bool ggml_backend_hrx_hsaco_make_row_dispatch_config(const ggml_backend_hrx_hsaco_catalog_entry * entry,
                                                     int64_t                                      nrows,
                                                     hrx_dispatch_config_t *                      out_config) {
    if (!entry || !out_config || nrows < 0) {
        return false;
    }
    if (static_cast<uint64_t>(nrows) > std::numeric_limits<uint32_t>::max()) {
        GGML_LOG_ERROR("%s: row count is too large: %" PRId64 "\n", __func__, nrows);
        return false;
    }

    *out_config = {
        /* .workgroup_count = */ { static_cast<uint32_t>(nrows), 1, 1 },
        /* .workgroup_size = */
        {
                                  entry->workgroup_size[0],
                                  entry->workgroup_size[1],
                                  entry->workgroup_size[2],
                                  },
        /* .subgroup_size = */
        0,
    };
    return true;
}

int64_t ggml_backend_hrx_hsaco_next_power_of_2(int64_t value) {
    int64_t result = 1;
    while (result < value) {
        result <<= 1;
    }
    return result;
}

struct ggml_backend_hrx_hsaco_catalog {
    hrx_device_t                                                      device = nullptr;
    std::string                                                       architecture;
    std::string                                                       target;
    std::mutex                                                        routes_mutex;
    std::vector<std::unique_ptr<ggml_backend_hrx_loaded_hsaco_route>> routes;

    ~ggml_backend_hrx_hsaco_catalog() {
        routes.clear();
        if (device) {
            hrx_device_release(device);
        }
    }
};

const ggml_backend_hrx_hsaco_catalog_entry * ggml_backend_hrx_hsaco_find_entry(
    const ggml_backend_hrx_hsaco_catalog * catalog,
    const char *                           route_id) {
    if (!catalog || !route_id) {
        return nullptr;
    }

    size_t                                       count   = 0;
    const ggml_backend_hrx_hsaco_catalog_entry * entries = ggml_backend_hrx_hsaco_catalog_entries(&count);
    for (size_t i = 0; i < count; ++i) {
        if (std::strcmp(entries[i].id, route_id) == 0 && catalog->target == entries[i].target) {
            return &entries[i];
        }
    }
    return nullptr;
}

namespace {

static ggml_backend_hrx_loaded_hsaco_route * ggml_backend_hrx_hsaco_get_loaded_route(
    ggml_backend_hrx_hsaco_catalog *             catalog,
    const ggml_backend_hrx_hsaco_catalog_entry * entry) {
    if (!catalog || !entry) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(catalog->routes_mutex);
    for (const auto & route : catalog->routes) {
        if (route->entry == entry) {
            return route.get();
        }
    }

    hrx_executable_t executable = nullptr;
    if (!GGML_HRX_HSACO_CHECK(hrx_executable_load_data(catalog->device, entry->data, entry->data_size,
                                                       "amdgpu", catalog->target.c_str(), &executable))) {
        return nullptr;
    }

    uint32_t export_ordinal = 0;
    if (!GGML_HRX_HSACO_CHECK(hrx_executable_lookup_export_by_name(executable, entry->symbol, &export_ordinal))) {
        hrx_executable_release(executable);
        return nullptr;
    }

    hrx_executable_export_info_t export_info = {};
    if (!GGML_HRX_HSACO_CHECK(hrx_executable_export_info(executable, export_ordinal, &export_info))) {
        hrx_executable_release(executable);
        return nullptr;
    }

    if (!ggml_backend_hrx_export_abi_matches(__func__, entry->id, export_info, entry->binding_count,
                                             entry->parameter_count, entry->constant_byte_length)) {
        hrx_executable_release(executable);
        return nullptr;
    }

    auto route            = std::make_unique<ggml_backend_hrx_loaded_hsaco_route>();
    route->entry          = entry;
    route->executable     = executable;
    route->export_ordinal = export_ordinal;
    route->export_info    = export_info;
    catalog->routes.push_back(std::move(route));
    return catalog->routes.back().get();
}

static bool ggml_backend_hrx_hsaco_dispatch_plan(ggml_backend_hrx_hsaco_catalog *              catalog,
                                                 hrx_stream_t                                  stream,
                                                 const ggml_backend_hrx_hsaco_execution_plan * plan) {
    if (!plan || !plan->entry) {
        return false;
    }

    auto * route = ggml_backend_hrx_hsaco_get_loaded_route(catalog, plan->entry);
    if (!route || !stream) {
        return false;
    }

    if (!ggml_backend_hrx_dispatch_abi_matches(__func__, plan->entry->id, plan->binding_count,
                                               plan->entry->binding_count, plan->constants_size,
                                               plan->entry->constant_byte_length)) {
        return false;
    }

    return GGML_HRX_HSACO_CHECK(hrx_stream_dispatch(stream, route->executable, route->export_ordinal, &plan->dispatch,
                                                    plan->constants, plan->constants_size, plan->bindings,
                                                    plan->binding_count, HRX_DISPATCH_FLAG_NONE));
}

}  // namespace

bool ggml_backend_hrx_hsaco_bind_tensor(const ggml_backend_hrx_hsaco_op_request * request,
                                        const ggml_tensor *                       tensor,
                                        hrx_buffer_ref_t *                        out_ref) {
    return request && request->bind_tensor && request->bind_tensor(request->bind_tensor_user_data, tensor, out_ref);
}

ggml_backend_hrx_hsaco_op_response ggml_backend_hrx_hsaco_match_request(
    ggml_backend_hrx_hsaco_catalog * catalog, const ggml_backend_hrx_hsaco_op_request * request) {
    return ggml_backend_hrx_hsaco_match_or_prepare_request(catalog, request, nullptr);
}

ggml_backend_hrx_hsaco_op_response ggml_backend_hrx_hsaco_prepare_plan(
    ggml_backend_hrx_hsaco_catalog *          catalog,
    const ggml_backend_hrx_hsaco_op_request * request,
    ggml_backend_hrx_hsaco_execution_plan *   plan) {
    return ggml_backend_hrx_hsaco_match_or_prepare_request(catalog, request, plan);
}

ggml_backend_hrx_hsaco_catalog * ggml_backend_hrx_hsaco_catalog_new(hrx_device_t device, const char * architecture) {
    if (!device || !architecture || architecture[0] == '\0') {
        return nullptr;
    }

    auto * catalog = new (std::nothrow) ggml_backend_hrx_hsaco_catalog();
    if (!catalog) {
        return nullptr;
    }

    hrx_device_retain(device);
    catalog->device       = device;
    catalog->architecture = architecture;
    catalog->target       = ggml_backend_hrx_architecture_base(architecture);
    return catalog;
}

void ggml_backend_hrx_hsaco_catalog_free(ggml_backend_hrx_hsaco_catalog * catalog) {
    delete catalog;
}

ggml_backend_hrx_hsaco_op_response ggml_backend_hrx_hsaco_supports_op(ggml_backend_hrx_hsaco_catalog * catalog,
                                                                      const ggml_tensor *              op) {
    if (!catalog || !op) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_NO_ROUTE);
    }

    const ggml_backend_hrx_hsaco_op_request request = {
        /* .op                    = */ op,
        /* .stream                = */ nullptr,
        /* .bind_tensor           = */ nullptr,
        /* .bind_tensor_user_data = */ nullptr,
    };
    return ggml_backend_hrx_hsaco_match_request(catalog, &request);
}

ggml_backend_hrx_hsaco_op_response ggml_backend_hrx_hsaco_invoke(ggml_backend_hrx_hsaco_catalog *          catalog,
                                                                 const ggml_backend_hrx_hsaco_op_request * request) {
    if (!catalog || !request || !request->op) {
        return ggml_backend_hrx_hsaco_failed(nullptr);
    }

    ggml_backend_hrx_hsaco_execution_plan plan     = {};
    ggml_backend_hrx_hsaco_op_response    response = ggml_backend_hrx_hsaco_prepare_plan(catalog, request, &plan);
    if (response.result != GGML_BACKEND_HRX_HSACO_INVOKED) {
        return response;
    }
    if (!ggml_backend_hrx_hsaco_dispatch_plan(catalog, request->stream, &plan)) {
        return ggml_backend_hrx_hsaco_failed(response.route_id);
    }
    return response;
}
