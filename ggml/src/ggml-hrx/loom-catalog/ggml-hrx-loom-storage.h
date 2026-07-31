#pragma once

#include "ggml-hrx-loom-catalog.h"

#include <cstddef>

struct ggml_tensor;

const ggml_backend_hrx_loom_storage_transform_entry *
ggml_backend_hrx_loom_storage_transform_match(
    const char *        architecture,
    const ggml_tensor * tensor);

size_t ggml_backend_hrx_loom_storage_transform_size(
    const ggml_backend_hrx_loom_storage_transform_entry * transform);

bool ggml_backend_hrx_loom_storage_transform_pack(
    const ggml_backend_hrx_loom_storage_transform_entry * transform,
    const void *                                          canonical,
    size_t                                                canonical_size,
    void *                                                packed,
    size_t                                                packed_size);

bool ggml_backend_hrx_loom_storage_transform_unpack(
    const ggml_backend_hrx_loom_storage_transform_entry * transform,
    const void *                                          packed,
    size_t                                                packed_size,
    void *                                                canonical,
    size_t                                                canonical_size);
