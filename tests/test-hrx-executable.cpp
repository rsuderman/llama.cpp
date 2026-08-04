#include "executable-program.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

#define REQUIRE(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, #condition); \
        std::abort(); \
    } \
} while (false)

namespace {

ggml::hrx::KernelDefinition definition() {
    ggml::hrx::KernelDefinition result;
    result.id = "test_kernel";
    result.symbol = "test_kernel";
    result.source = "test.loom";
    result.source_digest = "source-digest";
    result.target = "gfx1151";
    result.workload_parameters = { { "rows", "index" } };
    result.launch_parameters = { { "rows", "index" }, { "columns", "index" } };
    result.compile_recipe.mode = "direct";
    result.compile_recipe.primary_sources = { result.source };
    return result;
}

void test_constant_packing() {
    const ggml::hrx::KernelDefinition kernel = definition();
    ggml::hrx::Command command;
    command.scalar_parameters = {
        { "rows", 7 }, { "columns", std::numeric_limits<uint32_t>::max() },
    };
    const ggml::hrx::PackedKernelConstants packed =
        ggml::hrx::pack_kernel_constants(kernel, command);
    REQUIRE(packed.valid());
    REQUIRE(packed.bytes.size() == 8);
    uint32_t rows = 0;
    uint32_t columns = 0;
    std::memcpy(&rows, packed.bytes.data(), sizeof(rows));
    std::memcpy(&columns, packed.bytes.data() + sizeof(rows), sizeof(columns));
    REQUIRE(rows == 7);
    REQUIRE(columns == std::numeric_limits<uint32_t>::max());

    command.scalar_parameters.erase("columns");
    REQUIRE(!ggml::hrx::pack_kernel_constants(kernel, command).valid());
    command.scalar_parameters["columns"] = -1;
    REQUIRE(!ggml::hrx::pack_kernel_constants(kernel, command).valid());
    command.scalar_parameters["columns"] =
        static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) + 1;
    REQUIRE(!ggml::hrx::pack_kernel_constants(kernel, command).valid());

    ggml::hrx::KernelDefinition unsupported = kernel;
    unsupported.launch_parameters[0].type = "i64";
    command.scalar_parameters["columns"] = 1;
    REQUIRE(!ggml::hrx::pack_kernel_constants(unsupported, command).valid());
}

void test_artifact_key_uses_only_compilation_facts() {
    const ggml::hrx::KernelDefinition kernel = definition();
    ggml::hrx::Command command;
    command.scalar_parameters = { { "rows", 7 }, { "columns", 32 }, { "layer", 4 } };
    command.compile_parameters = { { "mode", "fast" } };
    const std::string first = ggml::hrx::kernel_artifact_key(kernel, command);
    command.scalar_parameters["layer"] = 47;
    REQUIRE(ggml::hrx::kernel_artifact_key(kernel, command) == first);
    command.scalar_parameters["rows"] = 8;
    REQUIRE(ggml::hrx::kernel_artifact_key(kernel, command) != first);
    command.scalar_parameters["rows"] = 7;
    command.compile_parameters["mode"] = "precise";
    REQUIRE(ggml::hrx::kernel_artifact_key(kernel, command) != first);
}

void test_binding_diagnostics_are_explicit() {
    ggml::hrx::ExecutableBindings bindings;
    ggml::hrx::ExecutableBufferBinding device_weight;
    device_weight.storage = 3;
    device_weight.buffer = reinterpret_cast<hrx_buffer_t>(uintptr_t { 1 });
    device_weight.capacity = 4096;
    device_weight.length = 2048;
    device_weight.weight = true;
    bindings.storages.push_back(device_weight);
    ggml::hrx::ExecutableBufferBinding host_weight;
    host_weight.storage = 4;
    host_weight.host_data = reinterpret_cast<void *>(uintptr_t { 1 });
    host_weight.capacity = 4096;
    host_weight.length = 4096;
    host_weight.weight = true;
    bindings.storages.push_back(host_weight);
    const std::string text = bindings.format();
    REQUIRE(text.find("borrowed_device_weight") != std::string::npos);
    REQUIRE(text.find("resident_host_weight") != std::string::npos);
    REQUIRE(text.find("layout=ggml-native") != std::string::npos);
    const std::string json = bindings.serialize_json();
    REQUIRE(json.find("borrowed_device_weight") != std::string::npos);
    REQUIRE(json.find("resident_host_weight") != std::string::npos);
}

} // namespace

int main() {
    test_constant_packing();
    test_artifact_key_uses_only_compilation_facts();
    test_binding_diagnostics_are_explicit();
    return 0;
}
