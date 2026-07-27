#include "ggml-hrx-hsaco-catalog-runtime.h"

#include "ggml-hrx-hsaco-catalog.h"
#include "ggml.h"
#include "ggml-impl.h"

#include <cinttypes>
#include <cstdint>
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

static constexpr const char * GGML_HRX_ROUTE_SCALE_F32_CONTIGUOUS = "scale_f32_contiguous";
static constexpr const char * GGML_HRX_ROUTE_CLAMP_F32_CONTIGUOUS = "clamp_f32_contiguous";
static constexpr const char * GGML_HRX_ROUTE_ADD_F32_CONTIGUOUS = "add_f32_contiguous";
static constexpr const char * GGML_HRX_ROUTE_MUL_F32_CONTIGUOUS_ROW_BROADCAST_128X32 =
    "mul_f32_contiguous_row_broadcast_128x32";
static constexpr const char * GGML_HRX_ROUTE_DIV_F32_CONTIGUOUS_SCALAR_8 = "div_f32_contiguous_scalar_8";
static constexpr const char * GGML_HRX_ROUTE_SUM_ROWS_F32_CONTIGUOUS_NCOLS_8 = "sum_rows_f32_contiguous_ncols_8";
static constexpr const char * GGML_HRX_ROUTE_SOFT_MAX_F32_CONTIGUOUS_NOMASK_NCOLS_128 =
    "soft_max_f32_contiguous_nomask_ncols_128";
static constexpr const char * GGML_HRX_ROUTE_ARGSORT_F32_I32_CONTIGUOUS_NCOLS_128 =
    "argsort_f32_i32_contiguous_ncols_128";

struct ggml_backend_hrx_scale_f32_constants {
    float scale;
    float bias;
    int64_t nelements;
};

static_assert(sizeof(ggml_backend_hrx_scale_f32_constants) == 16, "unexpected scale constant packing");

struct ggml_backend_hrx_clamp_f32_constants {
    float minimum;
    float maximum;
    int32_t nelements;
};

static_assert(sizeof(ggml_backend_hrx_clamp_f32_constants) == 12, "unexpected clamp constant packing");

struct ggml_backend_hrx_add_f32_constants {
    int64_t nelements;
};

static_assert(sizeof(ggml_backend_hrx_add_f32_constants) == 8, "unexpected add constant packing");

struct ggml_backend_hrx_binary_broadcast_f32_constants {
    int64_t nelements;
    int64_t ncols;
    int64_t src1_nelements;
};

static_assert(sizeof(ggml_backend_hrx_binary_broadcast_f32_constants) == 24, "unexpected binary constant packing");

struct ggml_backend_hrx_rows_f32_constants {
    int64_t ncols;
    int64_t nrows;
};

static_assert(sizeof(ggml_backend_hrx_rows_f32_constants) == 16, "unexpected row constant packing");

struct ggml_backend_hrx_argsort_f32_i32_constants {
    int64_t ncols;
    int64_t nrows;
    int64_t ncols_pad;
    int32_t order;
} __attribute__((packed));

static_assert(sizeof(ggml_backend_hrx_argsort_f32_i32_constants) == 28, "unexpected argsort constant packing");

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
        const ggml_backend_hrx_hsaco_catalog * catalog,
        const char * route_id);

static ggml_backend_hrx_hsaco_op_response ggml_backend_hrx_hsaco_response(
        ggml_backend_hrx_hsaco_result result,
        ggml_backend_hrx_hsaco_unsupported_reason unsupported_reason,
        const char * route_id) {
    return {
        /* .result             = */ result,
        /* .unsupported_reason = */ unsupported_reason,
        /* .route_id           = */ route_id,
    };
}

static ggml_backend_hrx_hsaco_op_response ggml_backend_hrx_hsaco_unsupported(
        ggml_backend_hrx_hsaco_unsupported_reason reason) {
    return ggml_backend_hrx_hsaco_response(GGML_BACKEND_HRX_HSACO_UNSUPPORTED, reason, nullptr);
}

static ggml_backend_hrx_hsaco_op_response ggml_backend_hrx_hsaco_supported(const char * route_id) {
    return ggml_backend_hrx_hsaco_response(
        GGML_BACKEND_HRX_HSACO_INVOKED,
        GGML_BACKEND_HRX_HSACO_UNSUPPORTED_NONE,
        route_id);
}

static ggml_backend_hrx_hsaco_op_response ggml_backend_hrx_hsaco_failed(const char * route_id) {
    return ggml_backend_hrx_hsaco_response(
        GGML_BACKEND_HRX_HSACO_FAILED,
        GGML_BACKEND_HRX_HSACO_UNSUPPORTED_NONE,
        route_id);
}

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

static bool ggml_backend_hrx_hsaco_make_row_dispatch_config(
        const ggml_backend_hrx_hsaco_catalog_entry * entry,
        int64_t nrows,
        hrx_dispatch_config_t * out_config) {
    if (!entry || !out_config || nrows < 0) {
        return false;
    }
    if (static_cast<uint64_t>(nrows) > std::numeric_limits<uint32_t>::max()) {
        GGML_LOG_ERROR("%s: row count is too large: %" PRId64 "\n", __func__, nrows);
        return false;
    }

    *out_config = {
        /* .workgroup_count = */ {static_cast<uint32_t>(nrows), 1, 1},
        /* .workgroup_size = */ {
            entry->workgroup_size[0],
            entry->workgroup_size[1],
            entry->workgroup_size[2],
        },
        /* .subgroup_size = */ 0,
    };
    return true;
}

static bool ggml_backend_hrx_hsaco_src1_supports_scalar_or_row_broadcast(
        const ggml_tensor * src1,
        const ggml_tensor * dst) {
    if (ggml_nelements(src1) == 1) {
        return true;
    }
    return src1->ne[0] == dst->ne[0] && src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1;
}

static int64_t ggml_backend_hrx_hsaco_next_power_of_2(int64_t value) {
    int64_t result = 1;
    while (result < value) {
        result <<= 1;
    }
    return result;
}

} // namespace

struct ggml_backend_hrx_hsaco_catalog {
    hrx_device_t device = nullptr;
    std::string architecture;
    std::string target;
    std::mutex routes_mutex;
    std::vector<std::unique_ptr<ggml_backend_hrx_loaded_hsaco_route>> routes;

    ~ggml_backend_hrx_hsaco_catalog() {
        routes.clear();
        if (device) {
            hrx_device_release(device);
        }
    }
};

namespace {

static const ggml_backend_hrx_hsaco_catalog_entry * ggml_backend_hrx_hsaco_find_entry(
        const ggml_backend_hrx_hsaco_catalog * catalog,
        const char * route_id) {
    if (!catalog || !route_id) {
        return nullptr;
    }

    size_t count = 0;
    const ggml_backend_hrx_hsaco_catalog_entry * entries = ggml_backend_hrx_hsaco_catalog_entries(&count);
    for (size_t i = 0; i < count; ++i) {
        if (std::strcmp(entries[i].id, route_id) == 0 && catalog->target == entries[i].target) {
            return &entries[i];
        }
    }
    return nullptr;
}

static bool ggml_backend_hrx_hsaco_route_available(
        const ggml_backend_hrx_hsaco_catalog * catalog,
        const char * route_id) {
    return ggml_backend_hrx_hsaco_find_entry(catalog, route_id) != nullptr;
}

static ggml_backend_hrx_loaded_hsaco_route * ggml_backend_hrx_hsaco_get_loaded_route(
        ggml_backend_hrx_hsaco_catalog * catalog,
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
    if (!GGML_HRX_HSACO_CHECK(hrx_executable_load_data(
            catalog->device,
            entry->data,
            entry->data_size,
            catalog->architecture.c_str(),
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
    catalog->routes.push_back(std::move(route));
    return catalog->routes.back().get();
}

} // namespace

ggml_backend_hrx_hsaco_catalog * ggml_backend_hrx_hsaco_catalog_new(
        hrx_device_t device,
        const char * architecture) {
    if (!device || !architecture || architecture[0] == '\0') {
        return nullptr;
    }

    auto * catalog = new (std::nothrow) ggml_backend_hrx_hsaco_catalog();
    if (!catalog) {
        return nullptr;
    }

    hrx_device_retain(device);
    catalog->device = device;
    catalog->architecture = architecture;
    catalog->target = ggml_backend_hrx_hsaco_architecture_base(architecture);
    return catalog;
}

void ggml_backend_hrx_hsaco_catalog_free(ggml_backend_hrx_hsaco_catalog * catalog) {
    delete catalog;
}

static bool ggml_backend_hrx_hsaco_dispatch_1d(
        ggml_backend_hrx_hsaco_catalog * catalog,
        hrx_stream_t stream,
        const char * route_id,
        const void * constants,
        size_t constants_size,
        const hrx_buffer_ref_t * bindings,
        size_t binding_count,
        int64_t nelements) {
    const ggml_backend_hrx_hsaco_catalog_entry * entry = ggml_backend_hrx_hsaco_find_entry(catalog, route_id);
    auto * route = ggml_backend_hrx_hsaco_get_loaded_route(catalog, entry);
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

static bool ggml_backend_hrx_hsaco_dispatch_rows(
        ggml_backend_hrx_hsaco_catalog * catalog,
        hrx_stream_t stream,
        const char * route_id,
        const void * constants,
        size_t constants_size,
        const hrx_buffer_ref_t * bindings,
        size_t binding_count,
        int64_t nrows) {
    const ggml_backend_hrx_hsaco_catalog_entry * entry = ggml_backend_hrx_hsaco_find_entry(catalog, route_id);
    auto * route = ggml_backend_hrx_hsaco_get_loaded_route(catalog, entry);
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
    if (!ggml_backend_hrx_hsaco_make_row_dispatch_config(entry, nrows, &dispatch_config)) {
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

static ggml_backend_hrx_hsaco_op_response ggml_backend_hrx_hsaco_supports_scale_f32(
        ggml_backend_hrx_hsaco_catalog * catalog,
        const ggml_tensor * op) {
    if (!op || op->op != GGML_OP_SCALE) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_NO_ROUTE);
    }
    if (!op->src[0]) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_SHAPE);
    }
    if (!ggml_backend_hrx_hsaco_route_available(catalog, GGML_HRX_ROUTE_SCALE_F32_CONTIGUOUS)) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_TARGET);
    }
    if (op->src[0]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_DTYPE);
    }
    if (!ggml_is_contiguous(op->src[0]) || !ggml_is_contiguous(op)) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_LAYOUT);
    }
    if (!ggml_are_same_shape(op->src[0], op)) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_SHAPE);
    }
    return ggml_backend_hrx_hsaco_supported(GGML_HRX_ROUTE_SCALE_F32_CONTIGUOUS);
}

static ggml_backend_hrx_hsaco_op_response ggml_backend_hrx_hsaco_supports_clamp_f32(
        ggml_backend_hrx_hsaco_catalog * catalog,
        const ggml_tensor * op) {
    if (!op || op->op != GGML_OP_CLAMP) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_NO_ROUTE);
    }
    if (!op->src[0]) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_SHAPE);
    }
    if (!ggml_backend_hrx_hsaco_route_available(catalog, GGML_HRX_ROUTE_CLAMP_F32_CONTIGUOUS)) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_TARGET);
    }
    if (op->src[0]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_DTYPE);
    }
    if (!ggml_is_contiguous(op->src[0]) || !ggml_is_contiguous(op)) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_LAYOUT);
    }
    if (!ggml_are_same_shape(op->src[0], op) || ggml_nelements(op) > std::numeric_limits<int32_t>::max()) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_SHAPE);
    }
    return ggml_backend_hrx_hsaco_supported(GGML_HRX_ROUTE_CLAMP_F32_CONTIGUOUS);
}

static ggml_backend_hrx_hsaco_op_response ggml_backend_hrx_hsaco_supports_add_f32(
        ggml_backend_hrx_hsaco_catalog * catalog,
        const ggml_tensor * op) {
    if (!op || op->op != GGML_OP_ADD) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_NO_ROUTE);
    }
    if (!op->src[0] || !op->src[1]) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_SHAPE);
    }
    if (!ggml_backend_hrx_hsaco_route_available(catalog, GGML_HRX_ROUTE_ADD_F32_CONTIGUOUS)) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_TARGET);
    }
    if (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_DTYPE);
    }
    if (!ggml_is_contiguous(op->src[0]) || !ggml_is_contiguous(op->src[1]) || !ggml_is_contiguous(op)) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_LAYOUT);
    }
    if (!ggml_are_same_shape(op->src[0], op->src[1]) || !ggml_are_same_shape(op->src[0], op)) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_SHAPE);
    }
    return ggml_backend_hrx_hsaco_supported(GGML_HRX_ROUTE_ADD_F32_CONTIGUOUS);
}

static ggml_backend_hrx_hsaco_op_response ggml_backend_hrx_hsaco_supports_binary_broadcast_f32(
        ggml_backend_hrx_hsaco_catalog * catalog,
        const ggml_tensor * op,
        ggml_op expected_op,
        const char * route_id) {
    if (!op || op->op != expected_op) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_NO_ROUTE);
    }
    if (!op->src[0] || !op->src[1]) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_SHAPE);
    }
    if (!ggml_backend_hrx_hsaco_route_available(catalog, route_id)) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_TARGET);
    }
    if (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_DTYPE);
    }
    if (!ggml_is_contiguous(op->src[0]) || !ggml_is_contiguous(op->src[1]) || !ggml_is_contiguous(op)) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_LAYOUT);
    }
    if (!ggml_are_same_shape(op->src[0], op) ||
        !ggml_backend_hrx_hsaco_src1_supports_scalar_or_row_broadcast(op->src[1], op)) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_SHAPE);
    }
    if (expected_op == GGML_OP_MUL &&
        (op->ne[0] != 128 || op->ne[1] != 32 || op->ne[2] != 1 || op->ne[3] != 1 ||
         op->src[1]->ne[0] != 128 || op->src[1]->ne[1] != 1 ||
         op->src[1]->ne[2] != 1 || op->src[1]->ne[3] != 1)) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_SHAPE);
    }
    if (expected_op == GGML_OP_DIV &&
        (op->ne[0] != 8 || op->ne[1] != 1 || op->ne[2] != 1 || op->ne[3] != 1 ||
         ggml_nelements(op->src[1]) != 1)) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_SHAPE);
    }
    return ggml_backend_hrx_hsaco_supported(route_id);
}

static ggml_backend_hrx_hsaco_op_response ggml_backend_hrx_hsaco_supports_sum_rows_f32(
        ggml_backend_hrx_hsaco_catalog * catalog,
        const ggml_tensor * op) {
    if (!op || op->op != GGML_OP_SUM_ROWS) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_NO_ROUTE);
    }
    if (!op->src[0]) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_SHAPE);
    }
    if (!ggml_backend_hrx_hsaco_route_available(catalog, GGML_HRX_ROUTE_SUM_ROWS_F32_CONTIGUOUS_NCOLS_8)) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_TARGET);
    }
    if (op->src[0]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_DTYPE);
    }
    if (!ggml_is_contiguous(op->src[0]) || !ggml_is_contiguous(op)) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_LAYOUT);
    }
    if (op->src[0]->ne[0] != 8 ||
        op->ne[0] != 1 ||
        op->ne[1] != op->src[0]->ne[1] ||
        op->ne[2] != op->src[0]->ne[2] ||
        op->ne[3] != op->src[0]->ne[3]) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_SHAPE);
    }
    return ggml_backend_hrx_hsaco_supported(GGML_HRX_ROUTE_SUM_ROWS_F32_CONTIGUOUS_NCOLS_8);
}

static ggml_backend_hrx_hsaco_op_response ggml_backend_hrx_hsaco_supports_soft_max_f32(
        ggml_backend_hrx_hsaco_catalog * catalog,
        const ggml_tensor * op) {
    if (!op || op->op != GGML_OP_SOFT_MAX) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_NO_ROUTE);
    }
    if (!op->src[0]) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_SHAPE);
    }
    if (!ggml_backend_hrx_hsaco_route_available(catalog, GGML_HRX_ROUTE_SOFT_MAX_F32_CONTIGUOUS_NOMASK_NCOLS_128)) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_TARGET);
    }
    if (op->src[0]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_DTYPE);
    }
    if (!ggml_is_contiguous(op->src[0]) || !ggml_is_contiguous(op)) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_LAYOUT);
    }
    if (!ggml_are_same_shape(op->src[0], op) || op->ne[0] != 128 || op->src[1] || op->src[2]) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_SHAPE);
    }
    if (ggml_get_op_params_f32(op, 0) != 1.0f || ggml_get_op_params_f32(op, 1) != 0.0f) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_SHAPE);
    }
    return ggml_backend_hrx_hsaco_supported(GGML_HRX_ROUTE_SOFT_MAX_F32_CONTIGUOUS_NOMASK_NCOLS_128);
}

static ggml_backend_hrx_hsaco_op_response ggml_backend_hrx_hsaco_supports_argsort_f32_i32(
        ggml_backend_hrx_hsaco_catalog * catalog,
        const ggml_tensor * op) {
    if (!op || op->op != GGML_OP_ARGSORT) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_NO_ROUTE);
    }
    if (!op->src[0]) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_SHAPE);
    }
    if (!ggml_backend_hrx_hsaco_route_available(catalog, GGML_HRX_ROUTE_ARGSORT_F32_I32_CONTIGUOUS_NCOLS_128)) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_TARGET);
    }
    if (op->src[0]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_I32) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_DTYPE);
    }
    if (!ggml_is_contiguous(op->src[0]) || !ggml_is_contiguous(op)) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_LAYOUT);
    }
    const int32_t order = ggml_get_op_params_i32(op, 0);
    if (!ggml_are_same_shape(op->src[0], op) ||
        op->ne[0] != 128 ||
        (order != GGML_SORT_ORDER_ASC && order != GGML_SORT_ORDER_DESC)) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_SHAPE);
    }
    return ggml_backend_hrx_hsaco_supported(GGML_HRX_ROUTE_ARGSORT_F32_I32_CONTIGUOUS_NCOLS_128);
}

ggml_backend_hrx_hsaco_op_response ggml_backend_hrx_hsaco_supports_op(
        ggml_backend_hrx_hsaco_catalog * catalog,
        const ggml_tensor * op) {
    if (!catalog || !op) {
        return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_NO_ROUTE);
    }

    switch (op->op) {
        case GGML_OP_SCALE:
            return ggml_backend_hrx_hsaco_supports_scale_f32(catalog, op);
        case GGML_OP_CLAMP:
            return ggml_backend_hrx_hsaco_supports_clamp_f32(catalog, op);
        case GGML_OP_ADD:
            return ggml_backend_hrx_hsaco_supports_add_f32(catalog, op);
        case GGML_OP_MUL:
            return ggml_backend_hrx_hsaco_supports_binary_broadcast_f32(
                catalog, op, GGML_OP_MUL, GGML_HRX_ROUTE_MUL_F32_CONTIGUOUS_ROW_BROADCAST_128X32);
        case GGML_OP_DIV:
            return ggml_backend_hrx_hsaco_supports_binary_broadcast_f32(
                catalog, op, GGML_OP_DIV, GGML_HRX_ROUTE_DIV_F32_CONTIGUOUS_SCALAR_8);
        case GGML_OP_SUM_ROWS:
            return ggml_backend_hrx_hsaco_supports_sum_rows_f32(catalog, op);
        case GGML_OP_SOFT_MAX:
            return ggml_backend_hrx_hsaco_supports_soft_max_f32(catalog, op);
        case GGML_OP_ARGSORT:
            return ggml_backend_hrx_hsaco_supports_argsort_f32_i32(catalog, op);
        default:
            return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_NO_ROUTE);
    }
}

static bool ggml_backend_hrx_hsaco_bind_tensor(
        const ggml_backend_hrx_hsaco_op_request * request,
        const ggml_tensor * tensor,
        hrx_buffer_ref_t * out_ref) {
    return request && request->bind_tensor && request->bind_tensor(request->bind_tensor_user_data, tensor, out_ref);
}

static ggml_backend_hrx_hsaco_op_response ggml_backend_hrx_hsaco_invoke_scale_f32(
        ggml_backend_hrx_hsaco_catalog * catalog,
        const ggml_backend_hrx_hsaco_op_request * request,
        const char * route_id) {
    const ggml_tensor * node = request->op;
    hrx_buffer_ref_t bindings[2] = {};
    if (!ggml_backend_hrx_hsaco_bind_tensor(request, node->src[0], &bindings[0]) ||
        !ggml_backend_hrx_hsaco_bind_tensor(request, node, &bindings[1])) {
        GGML_LOG_ERROR("%s: failed to bind tensors for route %s\n", __func__, route_id);
        return ggml_backend_hrx_hsaco_failed(route_id);
    }

    const int64_t nelements = ggml_nelements(node);
    const ggml_backend_hrx_scale_f32_constants constants = {
        /* .scale     = */ ggml_get_op_params_f32(node, 0),
        /* .bias      = */ ggml_get_op_params_f32(node, 1),
        /* .nelements = */ nelements,
    };

    if (!ggml_backend_hrx_hsaco_dispatch_1d(
            catalog, request->stream, route_id, &constants, sizeof(constants), bindings, 2, nelements)) {
        return ggml_backend_hrx_hsaco_failed(route_id);
    }
    return ggml_backend_hrx_hsaco_supported(route_id);
}

static ggml_backend_hrx_hsaco_op_response ggml_backend_hrx_hsaco_invoke_clamp_f32(
        ggml_backend_hrx_hsaco_catalog * catalog,
        const ggml_backend_hrx_hsaco_op_request * request,
        const char * route_id) {
    const ggml_tensor * node = request->op;
    hrx_buffer_ref_t bindings[2] = {};
    if (!ggml_backend_hrx_hsaco_bind_tensor(request, node->src[0], &bindings[0]) ||
        !ggml_backend_hrx_hsaco_bind_tensor(request, node, &bindings[1])) {
        GGML_LOG_ERROR("%s: failed to bind tensors for route %s\n", __func__, route_id);
        return ggml_backend_hrx_hsaco_failed(route_id);
    }

    const int64_t nelements = ggml_nelements(node);
    const ggml_backend_hrx_clamp_f32_constants constants = {
        /* .minimum   = */ ggml_get_op_params_f32(node, 0),
        /* .maximum   = */ ggml_get_op_params_f32(node, 1),
        /* .nelements = */ static_cast<int32_t>(nelements),
    };

    if (!ggml_backend_hrx_hsaco_dispatch_1d(
            catalog, request->stream, route_id, &constants, sizeof(constants), bindings, 2, nelements)) {
        return ggml_backend_hrx_hsaco_failed(route_id);
    }
    return ggml_backend_hrx_hsaco_supported(route_id);
}

static ggml_backend_hrx_hsaco_op_response ggml_backend_hrx_hsaco_invoke_add_f32(
        ggml_backend_hrx_hsaco_catalog * catalog,
        const ggml_backend_hrx_hsaco_op_request * request,
        const char * route_id) {
    const ggml_tensor * node = request->op;
    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx_hsaco_bind_tensor(request, node->src[0], &bindings[0]) ||
        !ggml_backend_hrx_hsaco_bind_tensor(request, node->src[1], &bindings[1]) ||
        !ggml_backend_hrx_hsaco_bind_tensor(request, node, &bindings[2])) {
        GGML_LOG_ERROR("%s: failed to bind tensors for route %s\n", __func__, route_id);
        return ggml_backend_hrx_hsaco_failed(route_id);
    }

    const int64_t nelements = ggml_nelements(node);
    const ggml_backend_hrx_add_f32_constants constants = {
        /* .nelements = */ nelements,
    };

    if (!ggml_backend_hrx_hsaco_dispatch_1d(
            catalog, request->stream, route_id, &constants, sizeof(constants), bindings, 3, nelements)) {
        return ggml_backend_hrx_hsaco_failed(route_id);
    }
    return ggml_backend_hrx_hsaco_supported(route_id);
}

static ggml_backend_hrx_hsaco_op_response ggml_backend_hrx_hsaco_invoke_binary_broadcast_f32(
        ggml_backend_hrx_hsaco_catalog * catalog,
        const ggml_backend_hrx_hsaco_op_request * request,
        const char * route_id) {
    const ggml_tensor * node = request->op;
    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx_hsaco_bind_tensor(request, node->src[0], &bindings[0]) ||
        !ggml_backend_hrx_hsaco_bind_tensor(request, node->src[1], &bindings[1]) ||
        !ggml_backend_hrx_hsaco_bind_tensor(request, node, &bindings[2])) {
        GGML_LOG_ERROR("%s: failed to bind tensors for route %s\n", __func__, route_id);
        return ggml_backend_hrx_hsaco_failed(route_id);
    }

    const int64_t nelements = ggml_nelements(node);
    const ggml_backend_hrx_binary_broadcast_f32_constants constants = {
        /* .nelements      = */ nelements,
        /* .ncols          = */ node->ne[0],
        /* .src1_nelements = */ ggml_nelements(node->src[1]),
    };

    if (!ggml_backend_hrx_hsaco_dispatch_1d(
            catalog, request->stream, route_id, &constants, sizeof(constants), bindings, 3, nelements)) {
        return ggml_backend_hrx_hsaco_failed(route_id);
    }
    return ggml_backend_hrx_hsaco_supported(route_id);
}

static ggml_backend_hrx_hsaco_op_response ggml_backend_hrx_hsaco_invoke_sum_rows_f32(
        ggml_backend_hrx_hsaco_catalog * catalog,
        const ggml_backend_hrx_hsaco_op_request * request,
        const char * route_id) {
    const ggml_tensor * node = request->op;
    hrx_buffer_ref_t bindings[2] = {};
    if (!ggml_backend_hrx_hsaco_bind_tensor(request, node->src[0], &bindings[0]) ||
        !ggml_backend_hrx_hsaco_bind_tensor(request, node, &bindings[1])) {
        GGML_LOG_ERROR("%s: failed to bind tensors for route %s\n", __func__, route_id);
        return ggml_backend_hrx_hsaco_failed(route_id);
    }

    const int64_t nrows = ggml_nrows(node->src[0]);
    const ggml_backend_hrx_rows_f32_constants constants = {
        /* .ncols = */ node->src[0]->ne[0],
        /* .nrows = */ nrows,
    };

    if (!ggml_backend_hrx_hsaco_dispatch_rows(
            catalog, request->stream, route_id, &constants, sizeof(constants), bindings, 2, nrows)) {
        return ggml_backend_hrx_hsaco_failed(route_id);
    }
    return ggml_backend_hrx_hsaco_supported(route_id);
}

static ggml_backend_hrx_hsaco_op_response ggml_backend_hrx_hsaco_invoke_rows_f32(
        ggml_backend_hrx_hsaco_catalog * catalog,
        const ggml_backend_hrx_hsaco_op_request * request,
        const char * route_id) {
    const ggml_tensor * node = request->op;
    hrx_buffer_ref_t bindings[2] = {};
    if (!ggml_backend_hrx_hsaco_bind_tensor(request, node->src[0], &bindings[0]) ||
        !ggml_backend_hrx_hsaco_bind_tensor(request, node, &bindings[1])) {
        GGML_LOG_ERROR("%s: failed to bind tensors for route %s\n", __func__, route_id);
        return ggml_backend_hrx_hsaco_failed(route_id);
    }

    const int64_t nrows = ggml_nrows(node->src[0]);
    const ggml_backend_hrx_rows_f32_constants constants = {
        /* .ncols = */ node->src[0]->ne[0],
        /* .nrows = */ nrows,
    };

    if (!ggml_backend_hrx_hsaco_dispatch_rows(
            catalog, request->stream, route_id, &constants, sizeof(constants), bindings, 2, nrows)) {
        return ggml_backend_hrx_hsaco_failed(route_id);
    }
    return ggml_backend_hrx_hsaco_supported(route_id);
}

static ggml_backend_hrx_hsaco_op_response ggml_backend_hrx_hsaco_invoke_argsort_f32_i32(
        ggml_backend_hrx_hsaco_catalog * catalog,
        const ggml_backend_hrx_hsaco_op_request * request,
        const char * route_id) {
    const ggml_tensor * node = request->op;
    hrx_buffer_ref_t bindings[2] = {};
    if (!ggml_backend_hrx_hsaco_bind_tensor(request, node->src[0], &bindings[0]) ||
        !ggml_backend_hrx_hsaco_bind_tensor(request, node, &bindings[1])) {
        GGML_LOG_ERROR("%s: failed to bind tensors for route %s\n", __func__, route_id);
        return ggml_backend_hrx_hsaco_failed(route_id);
    }

    const int64_t nrows = ggml_nrows(node->src[0]);
    const ggml_backend_hrx_argsort_f32_i32_constants constants = {
        /* .ncols     = */ node->src[0]->ne[0],
        /* .nrows     = */ nrows,
        /* .ncols_pad = */ ggml_backend_hrx_hsaco_next_power_of_2(node->src[0]->ne[0]),
        /* .order     = */ ggml_get_op_params_i32(node, 0),
    };

    if (!ggml_backend_hrx_hsaco_dispatch_rows(
            catalog, request->stream, route_id, &constants, sizeof(constants), bindings, 2, nrows)) {
        return ggml_backend_hrx_hsaco_failed(route_id);
    }
    return ggml_backend_hrx_hsaco_supported(route_id);
}

ggml_backend_hrx_hsaco_op_response ggml_backend_hrx_hsaco_invoke(
        ggml_backend_hrx_hsaco_catalog * catalog,
        const ggml_backend_hrx_hsaco_op_request * request) {
    if (!catalog || !request || !request->op) {
        return ggml_backend_hrx_hsaco_failed(nullptr);
    }

    ggml_backend_hrx_hsaco_op_response support = ggml_backend_hrx_hsaco_supports_op(catalog, request->op);
    if (support.result != GGML_BACKEND_HRX_HSACO_INVOKED) {
        return support;
    }
    if (!request->stream || !request->bind_tensor) {
        return ggml_backend_hrx_hsaco_failed(support.route_id);
    }

    switch (request->op->op) {
        case GGML_OP_SCALE:
            return ggml_backend_hrx_hsaco_invoke_scale_f32(catalog, request, support.route_id);
        case GGML_OP_CLAMP:
            return ggml_backend_hrx_hsaco_invoke_clamp_f32(catalog, request, support.route_id);
        case GGML_OP_ADD:
            return ggml_backend_hrx_hsaco_invoke_add_f32(catalog, request, support.route_id);
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            return ggml_backend_hrx_hsaco_invoke_binary_broadcast_f32(catalog, request, support.route_id);
        case GGML_OP_SUM_ROWS:
            return ggml_backend_hrx_hsaco_invoke_sum_rows_f32(catalog, request, support.route_id);
        case GGML_OP_SOFT_MAX:
            return ggml_backend_hrx_hsaco_invoke_rows_f32(catalog, request, support.route_id);
        case GGML_OP_ARGSORT:
            return ggml_backend_hrx_hsaco_invoke_argsort_f32_i32(catalog, request, support.route_id);
        default:
            return ggml_backend_hrx_hsaco_unsupported(GGML_BACKEND_HRX_HSACO_UNSUPPORTED_NO_ROUTE);
    }
}
