#include "weight-residency.h"
#include "transfer-manager.h"

#include <mutex>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace ggml::hrx {
namespace {

struct SourceKey {
    uint64_t buffer_identity = 0;
    uint64_t generation = 0;
    size_t capacity = 0;
    size_t offset = 0;
    size_t length = 0;

    bool operator==(const SourceKey & other) const {
        return buffer_identity == other.buffer_identity && generation == other.generation &&
               capacity == other.capacity && offset == other.offset && length == other.length;
    }
};

struct SourceKeyHash {
    size_t operator()(const SourceKey & key) const {
        size_t result = static_cast<size_t>(key.buffer_identity);
        auto mix = [&](uint64_t value) {
            result ^= static_cast<size_t>(value) + 0x9e3779b97f4a7c15ull + (result << 6) + (result >> 2);
        };
        mix(key.generation);
        mix(key.capacity);
        mix(key.offset);
        mix(key.length);
        return result;
    }
};

} // namespace

struct WeightResidencyLease::Entry {
    ~Entry() { if (buffer != nullptr) hrx_buffer_release(buffer); }
    hrx_buffer_t buffer = nullptr;
    size_t length = 0;
    std::string layout;
};

WeightResidencyLease::WeightResidencyLease() = default;
WeightResidencyLease::~WeightResidencyLease() = default;
WeightResidencyLease::WeightResidencyLease(std::shared_ptr<Entry> entry) : entry_(std::move(entry)) {}
bool WeightResidencyLease::valid() const { return entry_ != nullptr && entry_->buffer != nullptr; }
hrx_buffer_t WeightResidencyLease::buffer() const { return valid() ? entry_->buffer : nullptr; }
size_t WeightResidencyLease::length() const { return valid() ? entry_->length : 0; }
const std::string & WeightResidencyLease::layout() const {
    static const std::string empty;
    return valid() ? entry_->layout : empty;
}

struct WeightResidencyCache::Impl {
    ~Impl() { if (device != nullptr) hrx_device_release(device); }
    hrx_device_t device = nullptr;
    std::string initialization_error;
    mutable std::mutex mutex;
    std::unordered_map<SourceKey, std::shared_ptr<WeightResidencyLease::Entry>, SourceKeyHash> entries;
    WeightResidencyStats stats;
};

WeightResidencyCache::WeightResidencyCache(hrx_device_t device) : impl_(new Impl()) {
    if (device == nullptr) {
        impl_->initialization_error = "weight residency cache requires a device";
        return;
    }
    impl_->device = device;
    hrx_device_retain(device);
}

WeightResidencyCache::~WeightResidencyCache() = default;
bool WeightResidencyCache::valid() const {
    return impl_ != nullptr && impl_->device != nullptr && impl_->initialization_error.empty();
}
const std::string & WeightResidencyCache::initialization_error() const { return impl_->initialization_error; }

WeightResidencyResult WeightResidencyCache::acquire(hrx_stream_t stream, TransferManager & transfers,
                                                     const WeightSource & source) {
    WeightResidencyResult result;
    if (!valid()) { result.error = initialization_error(); return result; }
    if (stream == nullptr || !transfers.valid()) { result.error = "weight residency requires a stream and transfer manager"; return result; }
    if (source.host_data == nullptr || source.buffer_identity == 0 || source.generation == 0 ||
        source.length == 0 || source.offset > source.capacity || source.length > source.capacity - source.offset) {
        result.error = "invalid host weight source";
        return result;
    }
    if (source.layout.empty()) { result.error = "host weight source has no layout"; return result; }

    const SourceKey key { source.buffer_identity, source.generation, source.capacity, source.offset, source.length };
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto found = impl_->entries.find(key);
    if (found != impl_->entries.end()) {
        if (found->second->layout != source.layout) {
            ++impl_->stats.layout_conflicts;
            result.error = "weight source already has resident layout " + found->second->layout +
                           ", cannot also materialize " + source.layout;
            return result;
        }
        ++impl_->stats.hits;
        result.lease = WeightResidencyLease(found->second);
        return result;
    }

    auto entry = std::make_shared<WeightResidencyLease::Entry>();
    entry->length = source.length;
    entry->layout = source.layout;
    ErrorResult error = take_status(hrx_buffer_allocate(stream, source.length, HRX_MEMORY_TYPE_DEVICE_LOCAL,
                                                        HRX_BUFFER_USAGE_DEFAULT, &entry->buffer));
    if (error) { result.error = "allocate resident weight: " + *error; return result; }
    if (std::string transfer_error = transfers.upload(
            static_cast<const uint8_t *>(source.host_data) + source.offset,
            entry->buffer, 0, source.length); !transfer_error.empty()) {
        result.error = "initialize resident weight: " + transfer_error;
        return result;
    }
    if (std::string transfer_error = transfers.join(stream); !transfer_error.empty()) {
        result.error = "initialize resident weight: " + transfer_error;
        return result;
    }

    impl_->entries.emplace(key, entry);
    ++impl_->stats.misses;
    impl_->stats.allocation_count = impl_->entries.size();
    impl_->stats.resident_bytes += source.length;
    result.lease = WeightResidencyLease(std::move(entry));
    return result;
}

WeightResidencyStats WeightResidencyCache::stats() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->stats;
}

std::string format_weight_residency_stats(const WeightResidencyStats & stats) {
    std::ostringstream out;
    out << "weight residency\n"
        << "hits=" << stats.hits << '\n'
        << "misses=" << stats.misses << '\n'
        << "layout_conflicts=" << stats.layout_conflicts << '\n'
        << "allocation_count=" << stats.allocation_count << '\n'
        << "resident_bytes=" << stats.resident_bytes << '\n';
    return out.str();
}

} // namespace ggml::hrx
