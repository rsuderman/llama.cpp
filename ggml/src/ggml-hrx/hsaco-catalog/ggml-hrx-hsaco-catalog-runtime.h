#pragma once

#include "hrx_runtime.h"

#include <cstddef>

struct ggml_tensor;

struct ggml_backend_hrx_hsaco_catalog;

enum ggml_backend_hrx_hsaco_result {
    GGML_BACKEND_HRX_HSACO_UNSUPPORTED,
    GGML_BACKEND_HRX_HSACO_INVOKED,
    GGML_BACKEND_HRX_HSACO_FAILED,
};

enum ggml_backend_hrx_hsaco_unsupported_reason {
    GGML_BACKEND_HRX_HSACO_UNSUPPORTED_NONE,
    GGML_BACKEND_HRX_HSACO_UNSUPPORTED_NO_ROUTE,
    GGML_BACKEND_HRX_HSACO_UNSUPPORTED_DTYPE,
    GGML_BACKEND_HRX_HSACO_UNSUPPORTED_LAYOUT,
    GGML_BACKEND_HRX_HSACO_UNSUPPORTED_SHAPE,
    GGML_BACKEND_HRX_HSACO_UNSUPPORTED_TARGET,
};

typedef bool (*ggml_backend_hrx_hsaco_bind_tensor_fn)(
        void * user_data,
        const ggml_tensor * tensor,
        hrx_buffer_ref_t * out_ref);

struct ggml_backend_hrx_hsaco_op_request {
    const ggml_tensor * op;
    hrx_stream_t stream;
    ggml_backend_hrx_hsaco_bind_tensor_fn bind_tensor;
    void * bind_tensor_user_data;
};

struct ggml_backend_hrx_hsaco_op_response {
    ggml_backend_hrx_hsaco_result result;
    ggml_backend_hrx_hsaco_unsupported_reason unsupported_reason;
    const char * route_id;
};

ggml_backend_hrx_hsaco_catalog * ggml_backend_hrx_hsaco_catalog_new(
        hrx_device_t device,
        const char * architecture);

void ggml_backend_hrx_hsaco_catalog_free(ggml_backend_hrx_hsaco_catalog * catalog);

ggml_backend_hrx_hsaco_op_response ggml_backend_hrx_hsaco_supports_op(
        ggml_backend_hrx_hsaco_catalog * catalog,
        const ggml_tensor * op);

ggml_backend_hrx_hsaco_op_response ggml_backend_hrx_hsaco_invoke(
        ggml_backend_hrx_hsaco_catalog * catalog,
        const ggml_backend_hrx_hsaco_op_request * request);
