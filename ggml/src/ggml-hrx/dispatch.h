#pragma once

#include "hrx_runtime.h"
#include "kernel-corpus-catalog.h"
#include "kernel-types.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct ggml_tensor;

namespace ggml::hrx {

struct KernelSpecialization {
    uint64_t                           kernel_id = kUncatalogedKernelId;
    std::map<std::string, int64_t>     integer_parameters;
    std::map<std::string, std::string> compile_parameters;
};

inline KernelSpecialization make_kernel_specialization(KernelCatalogRef ref) {
    KernelSpecialization kernel;
    kernel.kernel_id = ref.id;
    return kernel;
}

struct DispatchBinding {
    const ggml_tensor * tensor = nullptr;
    size_t              offset = 0;
    size_t              length = 0;
    hrx_buffer_t        buffer = nullptr;
};

struct Dispatch {
    KernelSpecialization         kernel;
    std::vector<DispatchBinding> bindings;
};

using DispatchTensorBinder = bool (*)(const ggml_tensor * tensor, DispatchBinding & binding, void * user_data);

struct DispatchMatchContext {
    DispatchTensorBinder bind_tensor = nullptr;
    void *               user_data   = nullptr;
};

}  // namespace ggml::hrx
