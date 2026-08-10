#include "transient-arena.h"

#include "hrx-interop-utils.h"
#include "hrx_runtime.h"

#include <utility>

namespace ggml::hrx {

TransientArena::AllocationLease::AllocationLease(TransientArena & arena, std::unique_lock<std::mutex> lock) :
    arena_(&arena),
    lock_(std::move(lock)) {}

bool TransientArena::AllocationLease::ensure_capacity(hrx_device_t device,
                                                      hrx_stream_t stream,
                                                      size_t       required_size,
                                                      ErrorLog &   errors) {
    if (arena_ == nullptr || !lock_.owns_lock()) {
        errors.log("missing transient arena allocation lease");
        return false;
    }
    return arena_->ensure_capacity_locked(device, stream, required_size, errors);
}

TransientArenaAllocationRef TransientArena::AllocationLease::current_allocation() const {
    if (arena_ == nullptr || !lock_.owns_lock()) {
        return {};
    }
    return arena_->current_allocation_locked();
}

TransientArena::~TransientArena() {
    clear();
}

void TransientArena::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (buffer_ != nullptr) {
        hrx_buffer_release(buffer_);
        buffer_ = nullptr;
    }
    allocation_capacity_ = 0;
    allocation_id_       = kInvalidTransientArenaAllocationId;
}

uint64_t TransientArena::next_allocation_id() {
    const uint64_t id = next_allocation_id_++;
    if (next_allocation_id_ == kInvalidTransientArenaAllocationId) {
        ++next_allocation_id_;
    }
    return id;
}

bool TransientArena::ensure_capacity(hrx_device_t device,
                                     hrx_stream_t stream,
                                     size_t       required_size,
                                     ErrorLog &   errors) {
    std::lock_guard<std::mutex> lock(mutex_);
    return ensure_capacity_locked(device, stream, required_size, errors);
}

TransientArena::AllocationLease TransientArena::acquire_allocation_lease() {
    return AllocationLease(*this, std::unique_lock<std::mutex>(mutex_));
}

bool TransientArena::ensure_capacity_locked(hrx_device_t device,
                                            hrx_stream_t stream,
                                            size_t       required_size,
                                            ErrorLog &   errors) {
    if (required_size == 0 || allocation_capacity_ >= required_size) {
        return true;
    }
    if (device == nullptr) {
        errors.log("missing HRX device for transient arena allocation");
        return false;
    }
    if (stream == nullptr) {
        errors.log("missing HRX stream for transient arena allocation");
        return false;
    }
    if (buffer_ != nullptr) {
        if (ErrorResult error = take_status(hrx_stream_synchronize(stream))) {
            errors.log("synchronize before growing transient arena: %s", error->c_str());
            return false;
        }
        hrx_buffer_release(buffer_);
        buffer_              = nullptr;
        allocation_capacity_ = 0;
        allocation_id_       = kInvalidTransientArenaAllocationId;
    }

    hrx_buffer_params_t params = {
        HRX_MEMORY_TYPE_DEVICE_LOCAL,
        HRX_MEMORY_ACCESS_ALL,
        HRX_BUFFER_USAGE_DEFAULT,
        0,
    };
    hrx_buffer_t allocation = nullptr;
    if (ErrorResult error = take_status(
            hrx_allocator_allocate_buffer(hrx_device_allocator(device), params, required_size, &allocation))) {
        errors.log("allocate transient arena: %s", error->c_str());
        return false;
    }

    buffer_              = allocation;
    allocation_capacity_ = required_size;
    allocation_id_       = next_allocation_id();
    return true;
}

TransientArenaAllocationRef TransientArena::current_allocation() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_allocation_locked();
}

size_t TransientArena::capacity() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocation_capacity_;
}

uint64_t TransientArena::allocation_id() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocation_id_;
}

TransientArenaAllocationRef TransientArena::current_allocation_locked() const {
    return { buffer_, allocation_capacity_, allocation_id_ };
}

}  // namespace ggml::hrx
