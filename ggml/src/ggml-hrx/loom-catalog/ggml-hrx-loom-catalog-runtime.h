#pragma once

#include "hrx_runtime.h"

#include <cstddef>

struct ggml_tensor;
struct ggml_cgraph;

struct ggml_backend_hrx_loom_catalog;

static constexpr int GGML_BACKEND_HRX_LOOM_MAX_CONSUMED_NODES = 8;

enum ggml_backend_hrx_loom_result {
    GGML_BACKEND_HRX_LOOM_UNSUPPORTED,
    GGML_BACKEND_HRX_LOOM_INVOKED,
    GGML_BACKEND_HRX_LOOM_FAILED,
};

enum ggml_backend_hrx_loom_unsupported_reason {
    GGML_BACKEND_HRX_LOOM_UNSUPPORTED_NONE,
    GGML_BACKEND_HRX_LOOM_UNSUPPORTED_NO_ROUTE,
    GGML_BACKEND_HRX_LOOM_UNSUPPORTED_DTYPE,
    GGML_BACKEND_HRX_LOOM_UNSUPPORTED_LAYOUT,
    GGML_BACKEND_HRX_LOOM_UNSUPPORTED_SHAPE,
    GGML_BACKEND_HRX_LOOM_UNSUPPORTED_TARGET,
    GGML_BACKEND_HRX_LOOM_UNSUPPORTED_CONFIG,
};

typedef bool (*ggml_backend_hrx_loom_bind_tensor_fn)(void *              user_data,
                                                     const ggml_tensor * tensor,
                                                     hrx_buffer_ref_t *  out_ref);

struct ggml_backend_hrx_loom_op_request {
    const ggml_tensor *                  op;
    const ggml_cgraph *                  cgraph;
    int                                  node_index;
    hrx_stream_t                         stream;
    ggml_backend_hrx_loom_bind_tensor_fn bind_tensor;
    void *                               bind_tensor_user_data;
};

struct ggml_backend_hrx_loom_op_response {
    ggml_backend_hrx_loom_result             result;
    ggml_backend_hrx_loom_unsupported_reason unsupported_reason;
    const char *                             route_id;
};

struct ggml_backend_hrx_loom_consumed_nodes {
    int count;
    int indices[GGML_BACKEND_HRX_LOOM_MAX_CONSUMED_NODES];
};

ggml_backend_hrx_loom_catalog * ggml_backend_hrx_loom_catalog_new(hrx_device_t device, const char * architecture);

void ggml_backend_hrx_loom_catalog_free(ggml_backend_hrx_loom_catalog * catalog);

ggml_backend_hrx_loom_op_response ggml_backend_hrx_loom_supports_op(ggml_backend_hrx_loom_catalog * catalog,
                                                                    const ggml_tensor *             op);

ggml_backend_hrx_loom_op_response ggml_backend_hrx_loom_invoke(
    ggml_backend_hrx_loom_catalog *          catalog,
    const ggml_backend_hrx_loom_op_request * request,
    ggml_backend_hrx_loom_consumed_nodes *   consumed_nodes);
