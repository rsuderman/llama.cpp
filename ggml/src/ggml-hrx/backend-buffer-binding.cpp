#include "backend-buffer-binding.h"

#include "ggml-backend-impl.h"
#include "ggml.h"

ggml_backend_hrx_buffer_context * ggml_backend_hrx_buffer_context_from_buffer(ggml_backend_buffer_t buffer) {
    return static_cast<ggml_backend_hrx_buffer_context *>(buffer->context);
}

size_t ggml_backend_hrx_tensor_offset(const ggml_backend_hrx_buffer_context * context, const ggml_tensor * tensor) {
    return static_cast<size_t>(static_cast<const uint8_t *>(tensor->data) - context->base);
}

void * ggml_backend_hrx_buffer_base(ggml_backend_buffer_t buffer) {
    return ggml_backend_hrx_buffer_context_from_buffer(buffer)->base;
}

bool ggml_backend_hrx_tensor_binding(const ggml_tensor *                tensor,
                                     ggml_backend_hrx_buffer_context ** out_context,
                                     size_t *                           out_offset) {
    if (tensor == nullptr) {
        return false;
    }
    ggml_backend_buffer_t buffer = tensor->view_src != nullptr ? tensor->view_src->buffer : tensor->buffer;
    if (buffer == nullptr || buffer->iface.get_base != ggml_backend_hrx_buffer_base) {
        return false;
    }
    auto *       context = ggml_backend_hrx_buffer_context_from_buffer(buffer);
    const size_t offset  = ggml_backend_hrx_tensor_offset(context, tensor);
    if (context->buffer == nullptr || offset > buffer->size || ggml_nbytes(tensor) > buffer->size - offset) {
        return false;
    }
    *out_context = context;
    *out_offset  = offset;
    return true;
}

bool ggml_backend_hrx_resolve_value_buffer(const ggml_tensor * tensor, ggml::hrx::ValueBufferBinding & binding) {
    ggml_backend_hrx_buffer_context * context = nullptr;
    size_t                            offset  = 0;
    if (!ggml_backend_hrx_tensor_binding(tensor, &context, &offset)) {
        return false;
    }
    ggml_backend_buffer_t buffer = tensor->view_src != nullptr ? tensor->view_src->buffer : tensor->buffer;
    binding.buffer               = context->buffer;
    binding.offset               = offset;
    binding.length               = ggml_nbytes(tensor);
    binding.identity             = context->identity;
    binding.generation           = context->generation;
    binding.capacity             = buffer != nullptr ? buffer->size : 0;
    return true;
}
