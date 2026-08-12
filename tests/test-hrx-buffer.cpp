#include "backend-context.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-hrx.h"
#include "ggml.h"
#include "hrx-interop-utils.h"
#include "runtime/host-memory.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define REQUIRE(condition)                                                                           \
    do {                                                                                             \
        if (!(condition)) {                                                                          \
            std::fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, #condition); \
            std::abort();                                                                            \
        }                                                                                            \
    } while (false)

static void require_hrx_status(hrx_status_t status) {
    if (ggml::hrx::ErrorResult error = ggml::hrx::take_status(status)) {
        std::fprintf(stderr, "HRX status failed: %s\n", error->c_str());
        std::abort();
    }
}

static ggml_backend_hrx_context * backend_context(ggml_backend_t backend) {
    auto * context = static_cast<ggml_backend_hrx_context *>(backend->context);
    REQUIRE(context != nullptr);
    REQUIRE(context->device != nullptr);
    REQUIRE(context->device->device != nullptr);
    REQUIRE(context->stream != nullptr);
    return context;
}

static void run_backend_buffer_checks(ggml_backend_t backend) {
    ggml_backend_hrx_context * hrx = backend_context(backend);

    ggml_init_params params = {};
    params.mem_size         = 16 * 1024;
    params.no_alloc         = true;
    ggml_context * context  = ggml_init(params);
    REQUIRE(context != nullptr);
    ggml_tensor *         tensor      = ggml_new_tensor_1d(context, GGML_TYPE_I32, 64);
    ggml_tensor *         copy        = ggml_new_tensor_1d(context, GGML_TYPE_I32, 64);
    ggml_backend_buffer_t buffer      = ggml_backend_alloc_buffer(backend, 4096);
    ggml_backend_buffer_t copy_buffer = ggml_backend_alloc_buffer(backend, 4096);
    REQUIRE(buffer != nullptr);
    REQUIRE(copy_buffer != nullptr);
    tensor->buffer = buffer;
    tensor->data   = ggml_backend_buffer_get_base(buffer);
    copy->buffer   = copy_buffer;
    copy->data     = ggml_backend_buffer_get_base(copy_buffer);
    REQUIRE(ggml_backend_buffer_init_tensor(buffer, tensor) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_buffer_init_tensor(copy_buffer, copy) == GGML_STATUS_SUCCESS);

    std::array<uint32_t, 64> input = {};
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<uint32_t>(i * 17 + 3);
    }
    ggml_backend_tensor_set(tensor, input.data(), 0, sizeof(input));
    std::array<uint32_t, 64> output = {};
    ggml_backend_tensor_get(tensor, output.data(), 0, sizeof(output));
    REQUIRE(output == input);
    ggml_backend_tensor_copy(tensor, copy);
    output.fill(0);
    ggml_backend_tensor_get(copy, output.data(), 0, sizeof(output));
    REQUIRE(output == input);

    input[0] = 0x12345678;
    ggml_backend_tensor_set_async(backend, tensor, input.data(), 0, sizeof(input));
    ggml_backend_synchronize(backend);
    output.fill(0);
    ggml_backend_tensor_get(tensor, output.data(), 0, sizeof(output));
    REQUIRE(output == input);

    ggml_backend_tensor_memset(tensor, 0x5a, 16, 32);
    ggml_backend_tensor_get(tensor, output.data(), 0, sizeof(output));
    const uint8_t * bytes = reinterpret_cast<const uint8_t *>(output.data());
    for (size_t i = 16; i < 48; ++i) {
        REQUIRE(bytes[i] == 0x5a);
    }

    ggml_backend_buffer_clear(buffer, 0);
    ggml_backend_tensor_get(tensor, output.data(), 0, sizeof(output));
    for (uint32_t value : output) {
        REQUIRE(value == 0);
    }

    ggml_backend_buffer_free(buffer);
    ggml_backend_buffer_free(copy_buffer);
    ggml_free(context);
    ggml_backend_synchronize(backend);
    REQUIRE(hrx->device->device != nullptr);
}

static void run_host_transfer_checks(ggml_backend_hrx_context * context) {
    ggml::hrx::HostTransferManager transfers;
    ggml::hrx::HostStagingBuffer   staging;
    REQUIRE(ggml::hrx::allocate_host_staging_buffer(context->device->device, 64, staging).success());

    std::array<uint8_t, 64> host = {};
    for (size_t i = 0; i < host.size(); ++i) {
        host[i] = static_cast<uint8_t>(i + 1);
    }

    REQUIRE(transfers.upload(context->stream, host.data(), staging.buffer, 0, 0).success());
    REQUIRE(transfers.upload(context->stream, host.data() + 8, staging.buffer, 16, 24).success());
    ggml::hrx::HostTransferStats stats = transfers.stats();
    REQUIRE(stats.uploads == 1);
    REQUIRE(stats.upload_bytes == 24);

    std::array<uint8_t, 64> upload_result = {};
    require_hrx_status(
        hrx_synchronous_d2h(context->device->device, staging.buffer, 0, upload_result.data(), upload_result.size()));
    for (size_t i = 0; i < upload_result.size(); ++i) {
        const uint8_t expected = i >= 16 && i < 40 ? host[i - 8] : 0;
        REQUIRE(upload_result[i] == expected);
    }

    std::array<uint8_t, 64> device_values = {};
    for (size_t i = 0; i < device_values.size(); ++i) {
        device_values[i] = static_cast<uint8_t>(0xa0 + i);
    }
    require_hrx_status(
        hrx_synchronous_h2d(context->device->device, device_values.data(), staging.buffer, 0, device_values.size()));

    std::array<uint8_t, 48> download_result = {};
    REQUIRE(transfers.download(context->stream, staging.buffer, 12, download_result.data() + 4, 20).success());
    stats = transfers.stats();
    REQUIRE(stats.downloads == 1);
    REQUIRE(stats.download_bytes == 20);
    for (size_t i = 0; i < download_result.size(); ++i) {
        const uint8_t expected = i >= 4 && i < 24 ? device_values[i + 8] : 0;
        REQUIRE(download_result[i] == expected);
    }

    REQUIRE(!transfers.upload(nullptr, host.data(), staging.buffer, 0, 4).success());
    REQUIRE(!transfers.upload(context->stream, nullptr, staging.buffer, 0, 4).success());
    REQUIRE(!transfers.upload(context->stream, host.data(), nullptr, 0, 4).success());
    REQUIRE(!transfers.download(nullptr, staging.buffer, 0, download_result.data(), 4).success());
    REQUIRE(!transfers.download(context->stream, nullptr, 0, download_result.data(), 4).success());
    REQUIRE(!transfers.download(context->stream, staging.buffer, 0, nullptr, 4).success());

    transfers.clear();
    stats = transfers.stats();
    REQUIRE(stats.uploads == 0);
    REQUIRE(stats.downloads == 0);
    REQUIRE(stats.upload_bytes == 0);
    REQUIRE(stats.download_bytes == 0);
}

static void run_host_staging_checks(ggml_backend_hrx_context * context) {
    ggml::hrx::HostStagingBuffer staging;
    REQUIRE(ggml::hrx::allocate_host_staging_buffer(context->device->device, 32, staging).success());
    REQUIRE(staging.buffer != nullptr);
    REQUIRE(staging.length == 32);

    hrx_buffer_t                 original = staging.buffer;
    ggml::hrx::HostStagingBuffer moved(std::move(staging));
    REQUIRE(moved.buffer == original);
    REQUIRE(moved.length == 32);
    REQUIRE(staging.buffer == nullptr);
    REQUIRE(staging.length == 0);

    ggml::hrx::HostStagingBuffer assigned;
    assigned = std::move(moved);
    REQUIRE(assigned.buffer == original);
    REQUIRE(assigned.length == 32);
    REQUIRE(moved.buffer == nullptr);
    REQUIRE(moved.length == 0);

    assigned.clear();
    REQUIRE(assigned.buffer == nullptr);
    REQUIRE(assigned.length == 0);
    assigned.clear();
    REQUIRE(assigned.buffer == nullptr);
}

static void run_host_weight_cache_checks(ggml_backend_hrx_context * context) {
    ggml::hrx::HostTransferManager transfers;
    ggml::hrx::HostWeightCache     weights;

    std::array<uint8_t, 128> host = {};
    for (size_t i = 0; i < host.size(); ++i) {
        host[i] = static_cast<uint8_t>(i);
    }

    ggml::hrx::HostWeightSource source;
    source.host_data  = host.data();
    source.identity   = 0x1234;
    source.generation = 1;
    source.capacity   = host.size();
    source.offset     = 16;
    source.length     = 32;
    source.layout     = "ggml-native";

    ggml::hrx::HostWeightAcquireResult first =
        weights.acquire(context->device->device, context->stream, transfers, source);
    REQUIRE(first.valid());

    std::array<uint8_t, 32> first_bytes = {};
    require_hrx_status(
        hrx_synchronous_d2h(context->device->device, first.lease.buffer(), 0, first_bytes.data(), first_bytes.size()));
    for (size_t i = 0; i < first_bytes.size(); ++i) {
        REQUIRE(first_bytes[i] == host[source.offset + i]);
    }

    ggml::hrx::HostWeightAcquireResult second =
        weights.acquire(context->device->device, context->stream, transfers, source);
    REQUIRE(second.valid());
    REQUIRE(second.lease.buffer() == first.lease.buffer());

    source.offset = 32;
    ggml::hrx::HostWeightAcquireResult slice =
        weights.acquire(context->device->device, context->stream, transfers, source);
    REQUIRE(slice.valid());
    REQUIRE(slice.lease.buffer() != first.lease.buffer());

    source.offset     = 16;
    source.generation = 2;
    ggml::hrx::HostWeightAcquireResult next_generation =
        weights.acquire(context->device->device, context->stream, transfers, source);
    REQUIRE(next_generation.valid());
    REQUIRE(next_generation.lease.buffer() != first.lease.buffer());

    source.generation = 1;
    source.layout     = "alternate-layout";
    ggml::hrx::HostWeightAcquireResult conflict =
        weights.acquire(context->device->device, context->stream, transfers, source);
    REQUIRE(!conflict.valid());

    ggml::hrx::HostWeightCacheStats weight_stats = weights.stats();
    REQUIRE(weight_stats.hits == 1);
    REQUIRE(weight_stats.misses == 3);
    REQUIRE(weight_stats.layout_conflicts == 1);
    REQUIRE(weight_stats.allocation_count == 3);
    REQUIRE(weight_stats.resident_bytes == 96);

    const ggml::hrx::HostTransferStats transfer_stats = transfers.stats();
    REQUIRE(transfer_stats.uploads == 3);
    REQUIRE(transfer_stats.upload_bytes == 96);

    weights.clear();
    weight_stats = weights.stats();
    REQUIRE(weight_stats.hits == 0);
    REQUIRE(weight_stats.misses == 0);
    REQUIRE(weight_stats.layout_conflicts == 0);
    REQUIRE(weight_stats.allocation_count == 0);
    REQUIRE(weight_stats.resident_bytes == 0);
}

int main() {
    if (ggml_backend_hrx_get_device_count() == 0) {
        std::fprintf(stderr, "test skipped: no HRX devices available\n");
        return 0;
    }

    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);
    ggml_backend_hrx_context * context = backend_context(backend);

    run_backend_buffer_checks(backend);
    run_host_transfer_checks(context);
    run_host_staging_checks(context);
    run_host_weight_cache_checks(context);

    ggml_backend_free(backend);
    return 0;
}
