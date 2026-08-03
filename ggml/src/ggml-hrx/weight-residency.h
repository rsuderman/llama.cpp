#pragma once

#include "hrx_runtime.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace ggml::hrx {

class TransferManager;

struct WeightSource {
    // Base of the backing host allocation; offset selects the resident range.
    const void * host_data = nullptr;
    uint64_t buffer_identity = 0;
    uint64_t generation = 0;
    size_t capacity = 0;
    size_t offset = 0;
    size_t length = 0;
    std::string layout = "ggml-native";
};

struct WeightResidencyStats {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t layout_conflicts = 0;
    size_t allocation_count = 0;
    size_t resident_bytes = 0;
};

class WeightResidencyLease {
public:
    WeightResidencyLease();
    ~WeightResidencyLease();
    WeightResidencyLease(const WeightResidencyLease &) = default;
    WeightResidencyLease & operator=(const WeightResidencyLease &) = default;
    WeightResidencyLease(WeightResidencyLease &&) noexcept = default;
    WeightResidencyLease & operator=(WeightResidencyLease &&) noexcept = default;

    bool valid() const;
    hrx_buffer_t buffer() const;
    size_t length() const;
    const std::string & layout() const;

private:
    struct Entry;
    std::shared_ptr<Entry> entry_;
    explicit WeightResidencyLease(std::shared_ptr<Entry> entry);
    friend class WeightResidencyCache;
};

struct WeightResidencyResult {
    WeightResidencyLease lease;
    std::string error;

    bool valid() const { return error.empty() && lease.valid(); }
};

// Owns the exceptional host-backed weights used by prepared executables. The
// normal path binds GGML's existing DEVICE_LOCAL HRX allocations directly.
// Entries intentionally live until backend teardown; eviction and relayout are
// outside the initial execution milestone.
class WeightResidencyCache {
public:
    explicit WeightResidencyCache(hrx_device_t device);
    ~WeightResidencyCache();
    WeightResidencyCache(const WeightResidencyCache &) = delete;
    WeightResidencyCache & operator=(const WeightResidencyCache &) = delete;

    bool valid() const;
    const std::string & initialization_error() const;
    WeightResidencyResult acquire(hrx_stream_t stream, TransferManager & transfers,
                                  const WeightSource & source);
    WeightResidencyStats stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::string format_weight_residency_stats(const WeightResidencyStats & stats);

} // namespace ggml::hrx
