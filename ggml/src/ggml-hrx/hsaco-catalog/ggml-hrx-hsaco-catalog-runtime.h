#pragma once

#include "hrx_runtime.h"

#include <cstddef>
#include <cstdint>

struct ggml_backend_hrx_hsaco_route_cache;

ggml_backend_hrx_hsaco_route_cache * ggml_backend_hrx_hsaco_route_cache_new(
        hrx_device_t device,
        const char * architecture);

void ggml_backend_hrx_hsaco_route_cache_free(ggml_backend_hrx_hsaco_route_cache * cache);

bool ggml_backend_hrx_hsaco_route_available(
        ggml_backend_hrx_hsaco_route_cache * cache,
        const char * route_id);

bool ggml_backend_hrx_hsaco_dispatch_1d(
        ggml_backend_hrx_hsaco_route_cache * cache,
        hrx_stream_t stream,
        const char * route_id,
        const void * constants,
        size_t constants_size,
        const hrx_buffer_ref_t * bindings,
        size_t binding_count,
        int64_t nelements);
