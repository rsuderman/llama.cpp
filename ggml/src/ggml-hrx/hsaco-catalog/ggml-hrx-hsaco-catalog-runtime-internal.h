#pragma once

#include "ggml-hrx-hsaco-catalog-runtime.h"
#include "ggml-hrx-hsaco-catalog.h"
#include "ggml-impl.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

static constexpr size_t GGML_BACKEND_HRX_HSACO_MAX_BINDINGS       = 8;
static constexpr size_t GGML_BACKEND_HRX_HSACO_MAX_CONSTANTS_SIZE = 256;

struct ggml_backend_hrx_hsaco_execution_plan {
    const ggml_backend_hrx_hsaco_catalog_entry * entry                                                = nullptr;
    hrx_dispatch_config_t                        dispatch                                             = {};
    hrx_buffer_ref_t                             bindings[GGML_BACKEND_HRX_HSACO_MAX_BINDINGS]        = {};
    size_t                                       binding_count                                        = 0;
    uint8_t                                      constants[GGML_BACKEND_HRX_HSACO_MAX_CONSTANTS_SIZE] = {};
    size_t                                       constants_size                                       = 0;
};

static inline bool ggml_backend_hrx_hsaco_match_only(const ggml_backend_hrx_hsaco_execution_plan * plan) {
    return plan == nullptr;
}

ggml_backend_hrx_hsaco_op_response ggml_backend_hrx_hsaco_unsupported(ggml_backend_hrx_hsaco_unsupported_reason reason);

ggml_backend_hrx_hsaco_op_response ggml_backend_hrx_hsaco_supported(const char * route_id);

ggml_backend_hrx_hsaco_op_response ggml_backend_hrx_hsaco_failed(const char * route_id);

const ggml_backend_hrx_hsaco_catalog_entry * ggml_backend_hrx_hsaco_find_entry(
    const ggml_backend_hrx_hsaco_catalog * catalog,
    const char *                           route_id);

bool ggml_backend_hrx_hsaco_make_1d_dispatch_config(const ggml_backend_hrx_hsaco_catalog_entry * entry,
                                                    int64_t                                      nelements,
                                                    hrx_dispatch_config_t *                      out_config);

bool ggml_backend_hrx_hsaco_make_row_dispatch_config(const ggml_backend_hrx_hsaco_catalog_entry * entry,
                                                     int64_t                                      nrows,
                                                     hrx_dispatch_config_t *                      out_config);

int64_t ggml_backend_hrx_hsaco_next_power_of_2(int64_t value);

bool ggml_backend_hrx_hsaco_bind_tensor(const ggml_backend_hrx_hsaco_op_request * request,
                                        const ggml_tensor *                       tensor,
                                        hrx_buffer_ref_t *                        out_ref);

ggml_backend_hrx_hsaco_op_response ggml_backend_hrx_hsaco_match_request(
    ggml_backend_hrx_hsaco_catalog *          catalog,
    const ggml_backend_hrx_hsaco_op_request * request);

ggml_backend_hrx_hsaco_op_response ggml_backend_hrx_hsaco_prepare_plan(
    ggml_backend_hrx_hsaco_catalog *          catalog,
    const ggml_backend_hrx_hsaco_op_request * request,
    ggml_backend_hrx_hsaco_execution_plan *   plan);

ggml_backend_hrx_hsaco_op_response ggml_backend_hrx_hsaco_match_or_prepare_request(
    ggml_backend_hrx_hsaco_catalog *          catalog,
    const ggml_backend_hrx_hsaco_op_request * request,
    ggml_backend_hrx_hsaco_execution_plan *   plan);
