#include "transfer-manager.h"
#include "weight-residency.h"

#include "hrx_runtime.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define REQUIRE(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, #condition); \
        std::abort(); \
    } \
} while (false)

static void check(hrx_status_t status, const char * operation) {
    if (hrx_status_is_ok(status)) return;
    char * message = nullptr;
    size_t length = 0;
    hrx_status_to_string(status, &message, &length);
    std::fprintf(stderr, "%s failed: %.*s\n", operation, static_cast<int>(length),
                 message != nullptr ? message : "unknown HRX error");
    hrx_status_free_message(message);
    hrx_status_ignore(status);
    std::abort();
}

int main() {
    hrx_status_t initialize_status = hrx_gpu_initialize(0);
    if (!hrx_status_is_ok(initialize_status)) {
        REQUIRE(hrx_status_code(initialize_status) == HRX_STATUS_ALREADY_EXISTS);
        hrx_status_ignore(initialize_status);
    }

    hrx_device_t device = nullptr;
    check(hrx_gpu_device_get(0, &device), "get device");
    REQUIRE(device != nullptr);
    hrx_device_retain(device);

    hrx_stream_t consumer = nullptr;
    check(hrx_stream_create(device, 0, &consumer), "create consumer stream");
    hrx_buffer_t first = nullptr;
    hrx_buffer_t second = nullptr;
    constexpr size_t byte_count = 5 * 1024 + 37;
    check(hrx_buffer_allocate(consumer, byte_count, HRX_MEMORY_TYPE_DEVICE_LOCAL,
                              HRX_BUFFER_USAGE_DEFAULT, &first), "allocate first buffer");
    check(hrx_buffer_allocate(consumer, byte_count, HRX_MEMORY_TYPE_DEVICE_LOCAL,
                              HRX_BUFFER_USAGE_DEFAULT, &second), "allocate second buffer");

    {
        ggml::hrx::TransferManagerOptions options;
        options.staging_page_size = 1024;
        options.maximum_staging_bytes = 2 * 1024;
        ggml::hrx::TransferManager transfers(device, options);
        REQUIRE(transfers.valid());

        std::vector<uint8_t> expected(byte_count);
        for (size_t i = 0; i < expected.size(); ++i) {
            expected[i] = static_cast<uint8_t>((i * 29 + 7) & 0xff);
        }
        REQUIRE(transfers.upload(expected.data(), first, 0, expected.size()).empty());
        REQUIRE(transfers.join(consumer).empty());

        // A device copy is ordered after the consumer that consumed the upload,
        // then joined back into that consumer without a host synchronization.
        REQUIRE(transfers.copy(consumer, first, 0, second, 0, expected.size()).empty());
        REQUIRE(transfers.join(consumer).empty());

        std::vector<uint8_t> actual(byte_count, 0);
        REQUIRE(transfers.download(consumer, second, 0, actual.data(), actual.size()).empty());
        REQUIRE(actual == expected);

        // Reusing a binding on the transfer stream must be ordered after its
        // previous consumer. This is the repeated executable launch shape.
        const uint8_t old_pattern = 0x3c;
        check(hrx_stream_fill_buffer(consumer, first, 0, byte_count,
                                     &old_pattern, sizeof(old_pattern)), "record prior consumer write");
        REQUIRE(transfers.wait_for_producer(consumer).empty());
        for (uint8_t & value : expected) value ^= 0x5a;
        REQUIRE(transfers.upload(expected.data(), first, 0, expected.size()).empty());
        REQUIRE(transfers.join(consumer).empty());
        REQUIRE(transfers.download(consumer, first, 0, actual.data(), actual.size()).empty());
        REQUIRE(actual == expected);

        const uint8_t pattern = 0xa5;
        REQUIRE(transfers.fill(second, 0, byte_count,
                               &pattern, sizeof(pattern)).empty());
        REQUIRE(transfers.join(consumer).empty());
        REQUIRE(transfers.download(consumer, second, 0, actual.data(), actual.size()).empty());
        REQUIRE(std::all_of(actual.begin(), actual.end(), [](uint8_t value) { return value == 0xa5; }));

        const ggml::hrx::TransferManagerStats stats = transfers.stats();
        REQUIRE(stats.uploads == 2);
        REQUIRE(stats.downloads == 3);
        REQUIRE(stats.uploaded_bytes == 2 * byte_count);
        REQUIRE(stats.downloaded_bytes == 3 * byte_count);
        REQUIRE(stats.page_allocations == 2);
        REQUIRE(stats.page_reuses > 0);
        REQUIRE(stats.backpressure_waits > 0);
        REQUIRE(stats.consumer_waits >= 3);
        REQUIRE(stats.producer_waits >= 2);
        REQUIRE(stats.staging_bytes == 2 * 1024);

        ggml::hrx::WeightResidencyCache weights(device);
        REQUIRE(weights.valid());
        ggml::hrx::WeightSource source;
        source.host_data = expected.data();
        source.buffer_identity = 17;
        source.generation = 3;
        source.capacity = expected.size();
        source.length = expected.size();
        auto first_weight = weights.acquire(consumer, transfers, source);
        REQUIRE(first_weight.valid());
        auto second_weight = weights.acquire(consumer, transfers, source);
        REQUIRE(second_weight.valid());
        REQUIRE(first_weight.lease.buffer() == second_weight.lease.buffer());
        ggml::hrx::WeightResidencyStats weight_stats = weights.stats();
        REQUIRE(weight_stats.misses == 1);
        REQUIRE(weight_stats.hits == 1);
        REQUIRE(weight_stats.allocation_count == 1);
        REQUIRE(weight_stats.resident_bytes == expected.size());
        source.layout = "incompatible-layout";
        auto conflict = weights.acquire(consumer, transfers, source);
        REQUIRE(!conflict.valid());
        REQUIRE(weights.stats().layout_conflicts == 1);
        REQUIRE(transfers.synchronize().empty());
    }

    hrx_buffer_release(second);
    hrx_buffer_release(first);
    hrx_stream_release(consumer);
    hrx_device_release(device);
    hrx_status_t shutdown_status = hrx_gpu_shutdown();
    if (!hrx_status_is_ok(shutdown_status)) hrx_status_ignore(shutdown_status);
    return 0;
}
