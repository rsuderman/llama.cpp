#include "transfer-manager.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>
#include <sstream>
#include <utility>
#include <vector>

namespace ggml::hrx {
namespace {

constexpr size_t kTransferAlignment = 256;

size_t align_up(size_t value, size_t alignment) {
    const size_t remainder = value % alignment;
    return remainder == 0 ? value : value + alignment - remainder;
}

std::string take_status(hrx_status_t status) {
    if (hrx_status_is_ok(status)) return {};
    char * message = nullptr;
    size_t length = 0;
    hrx_status_to_string(status, &message, &length);
    std::string result = message != nullptr ? std::string(message, length) : "unknown HRX error";
    hrx_status_free_message(message);
    hrx_status_ignore(status);
    return result;
}

} // namespace

struct TransferManager::Impl {
    enum class PageState { Available, Recording, InFlight };

    struct Page {
        hrx_buffer_t buffer = nullptr;
        uint8_t * mapped = nullptr;
        size_t offset = 0;
        PageState state = PageState::Available;
        hrx_timeline_point_t retirement = {};
    };

    ~Impl() {
        if (stream != nullptr) {
            hrx_status_t status = hrx_stream_synchronize(stream);
            if (!hrx_status_is_ok(status)) hrx_status_ignore(status);
        }
        for (Page & page : pages) {
            if (page.buffer != nullptr) hrx_buffer_release(page.buffer);
        }
        if (stream != nullptr) hrx_stream_release(stream);
        if (device != nullptr) hrx_device_release(device);
    }

    hrx_device_t device = nullptr;
    hrx_stream_t stream = nullptr;
    size_t page_size = 0;
    size_t maximum_pages = 0;
    size_t current_page = std::numeric_limits<size_t>::max();
    bool has_pending_copies = false;
    std::string initialization_error;
    mutable std::mutex mutex;
    std::vector<Page> pages;
    TransferManagerStats stats;

    std::string collect_locked() {
        for (Page & page : pages) {
            if (page.state != PageState::InFlight) continue;
            uint64_t completed = 0;
            std::string error = take_status(hrx_semaphore_query(page.retirement.semaphore, &completed));
            if (!error.empty()) return "query staging page retirement: " + error;
            if (completed >= page.retirement.value) {
                page.state = PageState::Available;
                page.offset = 0;
                page.retirement = {};
            }
        }
        return {};
    }

    std::string join_producer_locked(hrx_stream_t producer) {
        if (producer == nullptr || producer == stream) return {};
        std::string error = submit_locked(nullptr);
        if (!error.empty()) return error;
        error = take_status(hrx_stream_flush(producer));
        if (!error.empty()) return "flush transfer producer: " + error;
        hrx_timeline_point_t position = {};
        error = take_status(hrx_stream_get_timeline_position(producer, &position));
        if (!error.empty()) return "query transfer producer timeline: " + error;
        if (position.semaphore == nullptr || position.value == 0) return {};
        error = take_status(hrx_stream_wait_on(stream, position));
        if (!error.empty()) return "join producer into transfer stream: " + error;
        ++stats.producer_waits;
        return {};
    }

    std::string submit_locked(hrx_timeline_point_t * out_position) {
        if (!has_pending_copies) {
            if (out_position != nullptr) *out_position = {};
            return {};
        }
        std::string error = take_status(hrx_stream_flush(stream));
        if (!error.empty()) return "flush transfer stream: " + error;
        hrx_timeline_point_t position = {};
        error = take_status(hrx_stream_get_timeline_position(stream, &position));
        if (!error.empty()) return "query transfer timeline: " + error;
        for (Page & page : pages) {
            if (page.state != PageState::Recording) continue;
            page.state = PageState::InFlight;
            page.retirement = position;
        }
        current_page = std::numeric_limits<size_t>::max();
        has_pending_copies = false;
        ++stats.submissions;
        if (out_position != nullptr) *out_position = position;
        return {};
    }

    std::string allocate_page_locked(size_t * out_index) {
        hrx_buffer_params_t params = {
            HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
            HRX_MEMORY_ACCESS_ALL,
            HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED | HRX_BUFFER_USAGE_MAPPING_PERSISTENT,
            0,
        };
        Page page;
        std::string error = take_status(hrx_allocator_allocate_buffer(
            hrx_device_allocator(device), params, page_size, &page.buffer));
        if (!error.empty()) return "allocate transfer staging page: " + error;
        void * mapped = nullptr;
        error = take_status(hrx_buffer_map(page.buffer, HRX_MAP_READ | HRX_MAP_WRITE,
                                           0, page_size, &mapped));
        if (!error.empty()) {
            hrx_buffer_release(page.buffer);
            return "map transfer staging page: " + error;
        }
        page.mapped = static_cast<uint8_t *>(mapped);
        pages.push_back(page);
        *out_index = pages.size() - 1;
        ++stats.page_allocations;
        stats.staging_bytes = pages.size() * page_size;
        return {};
    }

    std::string acquire_page_locked(size_t * out_index) {
        std::string error = collect_locked();
        if (!error.empty()) return error;
        for (size_t i = 0; i < pages.size(); ++i) {
            if (pages[i].state == PageState::Available) {
                pages[i].state = PageState::Recording;
                pages[i].offset = 0;
                *out_index = i;
                ++stats.page_reuses;
                return {};
            }
        }
        if (pages.size() < maximum_pages) {
            error = allocate_page_locked(out_index);
            if (error.empty()) pages[*out_index].state = PageState::Recording;
            return error;
        }

        // A bounded staging pool eventually needs backpressure. Wait on the
        // oldest page timepoint, never on unrelated compute work or the device.
        size_t oldest = std::numeric_limits<size_t>::max();
        uint64_t oldest_value = std::numeric_limits<uint64_t>::max();
        for (size_t i = 0; i < pages.size(); ++i) {
            if (pages[i].state == PageState::InFlight && pages[i].retirement.value < oldest_value) {
                oldest = i;
                oldest_value = pages[i].retirement.value;
            }
        }
        if (oldest == std::numeric_limits<size_t>::max()) {
            return "bounded transfer pool has no recyclable in-flight page";
        }
        error = take_status(hrx_semaphore_wait(pages[oldest].retirement.semaphore,
                                               pages[oldest].retirement.value, UINT64_MAX));
        if (!error.empty()) return "wait for staging page backpressure: " + error;
        ++stats.backpressure_waits;
        pages[oldest].state = PageState::Recording;
        pages[oldest].offset = 0;
        pages[oldest].retirement = {};
        *out_index = oldest;
        ++stats.page_reuses;
        return {};
    }
};

TransferManager::TransferManager(hrx_device_t device, const TransferManagerOptions & options)
    : impl_(new Impl()) {
    if (device == nullptr) {
        impl_->initialization_error = "transfer manager requires a device";
        return;
    }
    if (options.staging_page_size == 0 || options.maximum_staging_bytes == 0) {
        impl_->initialization_error = "transfer staging sizes must be positive";
        return;
    }
    impl_->page_size = align_up(options.staging_page_size, kTransferAlignment);
    impl_->maximum_pages = std::max<size_t>(1, options.maximum_staging_bytes / impl_->page_size);
    impl_->device = device;
    hrx_device_retain(device);
    impl_->initialization_error = take_status(hrx_stream_create(device, 0, &impl_->stream));
    if (!impl_->initialization_error.empty()) {
        impl_->initialization_error = "create transfer stream: " + impl_->initialization_error;
    }
}

TransferManager::~TransferManager() = default;
TransferManager::TransferManager(TransferManager &&) noexcept = default;
TransferManager & TransferManager::operator=(TransferManager &&) noexcept = default;

bool TransferManager::valid() const {
    return impl_ != nullptr && impl_->initialization_error.empty() && impl_->stream != nullptr;
}

const std::string & TransferManager::initialization_error() const {
    return impl_->initialization_error;
}

std::string TransferManager::upload(const void * host_source, hrx_buffer_t destination,
                                    size_t destination_offset, size_t size) {
    if (!valid()) return initialization_error();
    if (size == 0) return {};
    if (host_source == nullptr || destination == nullptr) return "upload requires host source and destination buffer";
    size_t destination_size = 0;
    std::string error = take_status(hrx_buffer_get_size(destination, &destination_size));
    if (!error.empty()) return "query upload destination size: " + error;
    if (destination_offset > destination_size || size > destination_size - destination_offset) {
        return "upload exceeds destination buffer";
    }

    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto * bytes = static_cast<const uint8_t *>(host_source);
    size_t copied = 0;
    while (copied < size) {
        if (impl_->current_page == std::numeric_limits<size_t>::max()) {
            error = impl_->acquire_page_locked(&impl_->current_page);
            if (!error.empty()) return error;
        }
        auto & page = impl_->pages[impl_->current_page];
        page.offset = align_up(page.offset, kTransferAlignment);
        if (page.offset == impl_->page_size) {
            error = impl_->submit_locked(nullptr);
            if (!error.empty()) return error;
            continue;
        }
        const size_t chunk = std::min(size - copied, impl_->page_size - page.offset);
        std::memcpy(page.mapped + page.offset, bytes + copied, chunk);
        error = take_status(hrx_stream_copy_buffer(impl_->stream, page.buffer, page.offset,
                                                   destination, destination_offset + copied, chunk));
        if (!error.empty()) return "record staged upload: " + error;
        impl_->has_pending_copies = true;
        page.offset += chunk;
        copied += chunk;
        if (page.offset == impl_->page_size && copied < size) {
            error = impl_->submit_locked(nullptr);
            if (!error.empty()) return error;
        }
    }
    ++impl_->stats.uploads;
    impl_->stats.uploaded_bytes += size;
    return {};
}

std::string TransferManager::join(hrx_stream_t consumer) {
    if (!valid()) return initialization_error();
    if (consumer == nullptr) return "transfer join requires a consumer stream";
    std::lock_guard<std::mutex> lock(impl_->mutex);
    hrx_timeline_point_t position = {};
    std::string error = impl_->submit_locked(&position);
    if (!error.empty() || position.semaphore == nullptr) return error;
    error = take_status(hrx_stream_wait_on(consumer, position));
    if (!error.empty()) return "join transfer timeline into consumer: " + error;
    ++impl_->stats.consumer_waits;
    return {};
}

std::string TransferManager::wait_for_producer(hrx_stream_t producer) {
    if (!valid()) return initialization_error();
    if (producer == nullptr) return "transfer producer wait requires a producer stream";
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->join_producer_locked(producer);
}

std::string TransferManager::fill(hrx_buffer_t destination, size_t destination_offset,
                                  size_t size, const void * pattern, size_t pattern_size) {
    if (!valid()) return initialization_error();
    if (size == 0) return {};
    if (destination == nullptr || pattern == nullptr || pattern_size == 0) {
        return "fill requires destination and pattern";
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    std::string error = take_status(hrx_stream_fill_buffer(
        impl_->stream, destination, destination_offset, size, pattern, pattern_size));
    if (!error.empty()) return "record transfer fill: " + error;
    impl_->has_pending_copies = true;
    return {};
}

std::string TransferManager::copy(hrx_stream_t producer, hrx_buffer_t source, size_t source_offset,
                                  hrx_buffer_t destination, size_t destination_offset, size_t size) {
    if (!valid()) return initialization_error();
    if (size == 0) return {};
    if (source == nullptr || destination == nullptr) return "copy requires source and destination";
    std::lock_guard<std::mutex> lock(impl_->mutex);
    std::string error = impl_->join_producer_locked(producer);
    if (!error.empty()) return error;
    error = take_status(hrx_stream_copy_buffer(impl_->stream, source, source_offset,
                                               destination, destination_offset, size));
    if (!error.empty()) return "record transfer copy: " + error;
    impl_->has_pending_copies = true;
    return {};
}

std::string TransferManager::flush(hrx_timeline_point_t * position) {
    if (!valid()) return initialization_error();
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->submit_locked(position);
}

std::string TransferManager::download(hrx_stream_t producer, hrx_buffer_t source,
                                      size_t source_offset, void * host_destination, size_t size) {
    if (!valid()) return initialization_error();
    if (size == 0) return {};
    if (source == nullptr || host_destination == nullptr) return "download requires source buffer and host destination";
    size_t source_size = 0;
    std::string error = take_status(hrx_buffer_get_size(source, &source_size));
    if (!error.empty()) return "query download source size: " + error;
    if (source_offset > source_size || size > source_size - source_offset) return "download exceeds source buffer";

    std::lock_guard<std::mutex> lock(impl_->mutex);
    error = impl_->submit_locked(nullptr);
    if (!error.empty()) return error;
    error = impl_->join_producer_locked(producer);
    if (!error.empty()) return error;

    auto * bytes = static_cast<uint8_t *>(host_destination);
    size_t copied = 0;
    while (copied < size) {
        size_t page_index = 0;
        error = impl_->acquire_page_locked(&page_index);
        if (!error.empty()) return error;
        auto & page = impl_->pages[page_index];
        const size_t chunk = std::min(size - copied, impl_->page_size);
        error = take_status(hrx_stream_copy_buffer(impl_->stream, source, source_offset + copied,
                                                   page.buffer, 0, chunk));
        if (!error.empty()) return "record staged download: " + error;
        page.offset = chunk;
        impl_->has_pending_copies = true;
        hrx_timeline_point_t position = {};
        error = impl_->submit_locked(&position);
        if (!error.empty()) return error;
        error = take_status(hrx_semaphore_wait(position.semaphore, position.value, UINT64_MAX));
        if (!error.empty()) return "wait for staged download: " + error;
        std::memcpy(bytes + copied, page.mapped, chunk);
        page.state = Impl::PageState::Available;
        page.offset = 0;
        page.retirement = {};
        copied += chunk;
    }
    ++impl_->stats.downloads;
    impl_->stats.downloaded_bytes += size;
    return {};
}

std::string TransferManager::synchronize() {
    if (!valid()) return initialization_error();
    std::lock_guard<std::mutex> lock(impl_->mutex);
    std::string error = impl_->submit_locked(nullptr);
    if (!error.empty()) return error;
    error = take_status(hrx_stream_synchronize(impl_->stream));
    if (!error.empty()) return "synchronize transfer stream: " + error;
    for (auto & page : impl_->pages) {
        page.state = Impl::PageState::Available;
        page.offset = 0;
        page.retirement = {};
    }
    return {};
}

std::string TransferManager::collect() {
    if (!valid()) return initialization_error();
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->collect_locked();
}

TransferManagerStats TransferManager::stats() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->stats;
}

std::string format_transfer_manager_stats(const TransferManagerStats & stats) {
    std::ostringstream out;
    out << "transfer manager\n"
        << "uploads=" << stats.uploads << '\n'
        << "downloads=" << stats.downloads << '\n'
        << "uploaded_bytes=" << stats.uploaded_bytes << '\n'
        << "downloaded_bytes=" << stats.downloaded_bytes << '\n'
        << "submissions=" << stats.submissions << '\n'
        << "consumer_waits=" << stats.consumer_waits << '\n'
        << "producer_waits=" << stats.producer_waits << '\n'
        << "page_allocations=" << stats.page_allocations << '\n'
        << "page_reuses=" << stats.page_reuses << '\n'
        << "backpressure_waits=" << stats.backpressure_waits << '\n'
        << "staging_bytes=" << stats.staging_bytes << '\n';
    return out.str();
}

} // namespace ggml::hrx
