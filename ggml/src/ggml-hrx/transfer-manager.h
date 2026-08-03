#pragma once

#include "hrx_runtime.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace ggml::hrx {

struct TransferManagerOptions {
    size_t staging_page_size = 8ull * 1024ull * 1024ull;
    size_t maximum_staging_bytes = 32ull * 1024ull * 1024ull;
};

struct TransferManagerStats {
    uint64_t uploads = 0;
    uint64_t downloads = 0;
    uint64_t uploaded_bytes = 0;
    uint64_t downloaded_bytes = 0;
    uint64_t submissions = 0;
    uint64_t consumer_waits = 0;
    uint64_t producer_waits = 0;
    uint64_t page_allocations = 0;
    uint64_t page_reuses = 0;
    uint64_t backpressure_waits = 0;
    size_t staging_bytes = 0;
};

// Owns the host-visible staging pool and its transfer timeline. Host data is
// copied into mapped staging memory before this API returns, so callers do not
// need to preserve host pointer lifetime. Device consumers are ordered with a
// timeline dependency and do not require a host-side synchronization.
class TransferManager {
public:
    TransferManager(hrx_device_t device, const TransferManagerOptions & options = {});
    ~TransferManager();
    TransferManager(TransferManager &&) noexcept;
    TransferManager & operator=(TransferManager &&) noexcept;
    TransferManager(const TransferManager &) = delete;
    TransferManager & operator=(const TransferManager &) = delete;

    bool valid() const;
    const std::string & initialization_error() const;

    // Stages an upload on the manager's transfer stream. Several uploads may
    // be batched until join() or flush() publishes them.
    std::string upload(const void * host_source, hrx_buffer_t destination,
                       size_t destination_offset, size_t size);

    // Device-side operations use the same transfer timeline and are published
    // by join()/flush(). producer may identify compute work that writes source.
    std::string fill(hrx_buffer_t destination, size_t destination_offset,
                     size_t size, const void * pattern, size_t pattern_size);
    std::string copy(hrx_stream_t producer, hrx_buffer_t source, size_t source_offset,
                     hrx_buffer_t destination, size_t destination_offset, size_t size);

    // Publishes pending transfers and inserts a device-side wait into consumer.
    // This does not wait on the host.
    std::string join(hrx_stream_t consumer);

    // Publishes producer work and makes the transfer stream wait for it. Use
    // this before overwriting a device allocation consumed by earlier work.
    std::string wait_for_producer(hrx_stream_t producer);

    // Publishes pending transfers without attaching a consumer dependency.
    std::string flush(hrx_timeline_point_t * position = nullptr);

    // Readback has an immediate host-visibility contract and therefore waits
    // for the requested bytes. If producer is non-null, its current timeline
    // is joined into the transfer stream before the copy.
    std::string download(hrx_stream_t producer, hrx_buffer_t source,
                         size_t source_offset, void * host_destination, size_t size);

    // Waits for transfer completion and recycles all staging pages. Intended
    // for explicit synchronization and teardown, not dispatch paths.
    std::string synchronize();

    // Recycles pages whose transfer timeline points have completed.
    std::string collect();

    TransferManagerStats stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::string format_transfer_manager_stats(const TransferManagerStats & stats);

} // namespace ggml::hrx
