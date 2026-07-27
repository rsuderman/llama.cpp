#include "ggml-hrx-hsaco-catalog-runtime.h"

#include "ggml-hrx-hsaco-catalog.h"
#include "ggml-impl.h"

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
    const ggml_backend_hrx_hsaco_catalog_entry * entry = nullptr;
    hrx_executable_t executable = nullptr;
    uint32_t export_ordinal = 0;
    hrx_executable_export_info_t export_info = {};

    ~ggml_backend_hrx_loaded_hsaco_route() {
        if (executable) {
            hrx_executable_release(executable);
        }
    }
};

static bool ggml_backend_hrx_hsaco_log_status(hrx_status_t status, const char * expr, const char * file, int line) {
    if (hrx_status_is_ok(status)) {
        return true;
    }

    char * message = nullptr;
    size_t length = 0;
    hrx_status_to_string(status, &message, &length);
    GGML_LOG_ERROR("%s:%d: %s failed: %s\n", file, line, expr, message ? message : "unknown HRX error");
    hrx_status_free_message(message);
    hrx_status_ignore(status);
    return false;
}

#define GGML_HRX_HSACO_CHECK(expr) ggml_backend_hrx_hsaco_log_status((expr), #expr, __FILE__, __LINE__)

static std::string ggml_backend_hrx_hsaco_architecture_base(const char * architecture) {
    if (!architecture) {
        return std::string();
    }

    const std::string architecture_string(architecture);
    const size_t feature_pos = architecture_string.find(':');
    return feature_pos == std::string::npos ? architecture_string : architecture_string.substr(0, feature_pos);
}

static const ggml_backend_hrx_hsaco_catalog_entry * ggml_backend_hrx_hsaco_find_entry(
        const ggml_backend_hrx_hsaco_route_cache * cache,
        const char * route_id);

static bool ggml_backend_hrx_hsaco_make_1d_dispatch_config(
        const ggml_backend_hrx_hsaco_catalog_entry * entry,
        int64_t nelements,
        hrx_dispatch_config_t * out_config) {
    if (!entry || !out_config || entry->threads_per_block == 0 || nelements < 0) {
        return false;
    }

    const uint64_t threads_per_block = entry->threads_per_block;
    const uint64_t workgroup_count = (static_cast<uint64_t>(nelements) + threads_per_block - 1) / threads_per_block;
    if (workgroup_count > std::numeric_limits<uint32_t>::max()) {
        GGML_LOG_ERROR("%s: workgroup count is too large: %" PRIu64 "\n", __func__, workgroup_count);
        return false;
    }

    *out_config = {
        /* .workgroup_count = */ {static_cast<uint32_t>(workgroup_count), 1, 1},
        /* .workgroup_size = */ {
            entry->workgroup_size[0],
            entry->workgroup_size[1],
            entry->workgroup_size[2],
        },
        /* .subgroup_size = */ 0,
    };
    return true;
}

} // namespace

struct ggml_backend_hrx_hsaco_route_cache {
    hrx_device_t device = nullptr;
    std::string architecture;
    std::string target;
    std::mutex routes_mutex;
    std::vector<std::unique_ptr<ggml_backend_hrx_loaded_hsaco_route>> routes;

    ~ggml_backend_hrx_hsaco_route_cache() {
        routes.clear();
        if (device) {
            hrx_device_release(device);
        }
    }
};

namespace {

static const ggml_backend_hrx_hsaco_catalog_entry * ggml_backend_hrx_hsaco_find_entry(
        const ggml_backend_hrx_hsaco_route_cache * cache,
        const char * route_id) {
    if (!cache || !route_id) {
        return nullptr;
    }

    size_t count = 0;
    const ggml_backend_hrx_hsaco_catalog_entry * entries = ggml_backend_hrx_hsaco_catalog_entries(&count);
    for (size_t i = 0; i < count; ++i) {
        if (std::strcmp(entries[i].id, route_id) == 0 && cache->target == entries[i].target) {
            return &entries[i];
        }
    }
    return nullptr;
}

static ggml_backend_hrx_loaded_hsaco_route * ggml_backend_hrx_hsaco_get_loaded_route(
        ggml_backend_hrx_hsaco_route_cache * cache,
        const ggml_backend_hrx_hsaco_catalog_entry * entry) {
    if (!cache || !entry) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(cache->routes_mutex);
    for (const auto & route : cache->routes) {
        if (route->entry == entry) {
            return route.get();
        }
    }

    hrx_executable_t executable = nullptr;
    if (!GGML_HRX_HSACO_CHECK(hrx_executable_load_data(
            cache->device,
            entry->data,
            entry->data_size,
            cache->architecture.c_str(),
            &executable))) {
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

    if (export_info.binding_count != entry->binding_count ||
        export_info.parameter_count != entry->parameter_count ||
        export_info.constant_byte_length != entry->constant_byte_length) {
        GGML_LOG_ERROR(
            "%s: route %s export ABI mismatch bindings=%u expected=%u parameters=%u expected=%u constants=%u expected=%u\n",
            __func__,
            entry->id,
            export_info.binding_count,
            entry->binding_count,
            export_info.parameter_count,
            entry->parameter_count,
            export_info.constant_byte_length,
            entry->constant_byte_length);
        hrx_executable_release(executable);
        return nullptr;
    }

    auto route = std::make_unique<ggml_backend_hrx_loaded_hsaco_route>();
    route->entry = entry;
    route->executable = executable;
    route->export_ordinal = export_ordinal;
    route->export_info = export_info;
    cache->routes.push_back(std::move(route));
    return cache->routes.back().get();
}

} // namespace

ggml_backend_hrx_hsaco_route_cache * ggml_backend_hrx_hsaco_route_cache_new(
        hrx_device_t device,
        const char * architecture) {
    if (!device || !architecture || architecture[0] == '\0') {
        return nullptr;
    }

    auto * cache = new (std::nothrow) ggml_backend_hrx_hsaco_route_cache();
    if (!cache) {
        return nullptr;
    }

    hrx_device_retain(device);
    cache->device = device;
    cache->architecture = architecture;
    cache->target = ggml_backend_hrx_hsaco_architecture_base(architecture);
    return cache;
}

void ggml_backend_hrx_hsaco_route_cache_free(ggml_backend_hrx_hsaco_route_cache * cache) {
    delete cache;
}

bool ggml_backend_hrx_hsaco_route_available(
        ggml_backend_hrx_hsaco_route_cache * cache,
        const char * route_id) {
    return ggml_backend_hrx_hsaco_find_entry(cache, route_id) != nullptr;
}

bool ggml_backend_hrx_hsaco_dispatch_1d(
        ggml_backend_hrx_hsaco_route_cache * cache,
        hrx_stream_t stream,
        const char * route_id,
        const void * constants,
        size_t constants_size,
        const hrx_buffer_ref_t * bindings,
        size_t binding_count,
        int64_t nelements) {
    const ggml_backend_hrx_hsaco_catalog_entry * entry = ggml_backend_hrx_hsaco_find_entry(cache, route_id);
    auto * route = ggml_backend_hrx_hsaco_get_loaded_route(cache, entry);
    if (!route || !stream) {
        return false;
    }

    if (constants_size != entry->constant_byte_length || binding_count != entry->binding_count) {
        GGML_LOG_ERROR(
            "%s: route %s dispatch ABI mismatch bindings=%zu expected=%u constants=%zu expected=%u\n",
            __func__,
            entry->id,
            binding_count,
            entry->binding_count,
            constants_size,
            entry->constant_byte_length);
        return false;
    }

    hrx_dispatch_config_t dispatch_config = {};
    if (!ggml_backend_hrx_hsaco_make_1d_dispatch_config(entry, nelements, &dispatch_config)) {
        return false;
    }

    return GGML_HRX_HSACO_CHECK(hrx_stream_dispatch(
        stream,
        route->executable,
        route->export_ordinal,
        &dispatch_config,
        constants,
        constants_size,
        bindings,
        binding_count,
        HRX_DISPATCH_FLAG_NONE));
}
