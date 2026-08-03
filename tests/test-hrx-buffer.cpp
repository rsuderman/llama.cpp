#include "ggml-backend.h"
#include "ggml-hrx.h"
#include "ggml.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define REQUIRE(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, #condition); \
        std::abort(); \
    } \
} while (false)

int main() {
    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);

    ggml_init_params params = {};
    params.mem_size = 16 * 1024;
    params.no_alloc = true;
    ggml_context * context = ggml_init(params);
    REQUIRE(context != nullptr);
    ggml_tensor * tensor = ggml_new_tensor_1d(context, GGML_TYPE_I32, 64);
    ggml_tensor * copy = ggml_new_tensor_1d(context, GGML_TYPE_I32, 64);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_buffer(backend, 4096);
    ggml_backend_buffer_t copy_buffer = ggml_backend_alloc_buffer(backend, 4096);
    REQUIRE(buffer != nullptr);
    REQUIRE(copy_buffer != nullptr);
    tensor->buffer = buffer;
    tensor->data = ggml_backend_buffer_get_base(buffer);
    copy->buffer = copy_buffer;
    copy->data = ggml_backend_buffer_get_base(copy_buffer);
    REQUIRE(ggml_backend_buffer_init_tensor(buffer, tensor) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_buffer_init_tensor(copy_buffer, copy) == GGML_STATUS_SUCCESS);

    std::array<uint32_t, 64> input = {};
    for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<uint32_t>(i * 17 + 3);
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
    for (size_t i = 16; i < 48; ++i) REQUIRE(bytes[i] == 0x5a);

    ggml_backend_buffer_clear(buffer, 0);
    ggml_backend_tensor_get(tensor, output.data(), 0, sizeof(output));
    for (uint32_t value : output) REQUIRE(value == 0);

    ggml_backend_buffer_free(buffer);
    ggml_backend_buffer_free(copy_buffer);
    ggml_free(context);
    ggml_backend_free(backend);
    return 0;
}
