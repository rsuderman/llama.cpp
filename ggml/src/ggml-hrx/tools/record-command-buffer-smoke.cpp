#include "hrx_runtime.h"

#include <cstdint>
#include <iostream>
#include <string>

namespace {

static bool check(hrx_status_t status, const char * operation) {
    if (hrx_status_is_ok(status)) return true;
    char * message = nullptr;
    size_t length = 0;
    hrx_status_t format_status = hrx_status_to_string(status, &message, &length);
    if (!hrx_status_is_ok(format_status)) hrx_status_ignore(format_status);
    std::cerr << operation << ": " << (message ? message : "unknown HRX error") << '\n';
    hrx_status_free_message(message);
    hrx_status_ignore(status);
    return false;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::cerr << "usage: ggml-hrx-record-command-buffer-smoke gfx... kernel.hsaco\n";
        return 2;
    }
    const std::string target = argv[1];
    if (!check(hrx_gpu_initialize(0), "initialize GPU")) return 1;
    hrx_device_t device = nullptr;
    hrx_stream_t stream = nullptr;
    hrx_executable_t executable = nullptr;
    hrx_buffer_t input = nullptr;
    hrx_buffer_t weight = nullptr;
    hrx_buffer_t output = nullptr;
    hrx_graph_t graph = nullptr;
    hrx_graph_exec_t graph_exec = nullptr;
    bool ok = check(hrx_gpu_device_get(0, &device), "get GPU") &&
        check(hrx_stream_create(device, 0, &stream), "create stream") &&
        check(hrx_executable_load_file(device, argv[2], "amdgpu", target.c_str(), &executable), "load HSACO");
    uint32_t export_ordinal = 0;
    ok = ok && check(hrx_executable_lookup_export_by_name(
        executable, "qwen3_moe_router_projection_f32_one_row_wave64", &export_ordinal), "lookup export");
    if (ok) {
        ok = check(hrx_buffer_allocate(stream, 2048 * sizeof(float), HRX_MEMORY_TYPE_DEVICE_LOCAL,
                                      HRX_BUFFER_USAGE_DEFAULT, &input), "allocate input") &&
             check(hrx_buffer_allocate(stream, 128 * 2048 * sizeof(float), HRX_MEMORY_TYPE_DEVICE_LOCAL,
                                      HRX_BUFFER_USAGE_DEFAULT, &weight), "allocate weight") &&
             check(hrx_buffer_allocate(stream, 128 * sizeof(float), HRX_MEMORY_TYPE_DEVICE_LOCAL,
                                      HRX_BUFFER_USAGE_DEFAULT, &output), "allocate output");
    }
    if (ok) {
        const uint32_t token_count = 1;
        const hrx_buffer_ref_t bindings[] = {
            { input, 0, 2048 * sizeof(float) },
            { weight, 0, 128 * 2048 * sizeof(float) },
            { output, 0, 128 * sizeof(float) },
        };
        const hrx_graph_kernel_node_attrs_t attrs = {
            executable, export_ordinal,
            { { 128, 1, 1 }, { 64, 1, 1 }, 64 },
            &token_count, sizeof(token_count), bindings, 3, 0,
        };
        hrx_graph_node_t node = nullptr;
        size_t graph_size = 0;
        ok = check(hrx_graph_create(device, 0, &graph), "create graph") &&
             check(hrx_graph_add_kernel_node(graph, nullptr, 0, &attrs, &node), "record kernel node") &&
             check(hrx_graph_size(graph, &graph_size), "query graph size") && graph_size == 1 &&
             check(hrx_graph_instantiate(graph, 0, &graph_exec), "instantiate graph");
        if (ok) std::cout << "recorded target=" << target << " nodes=" << graph_size << " instantiated=true launched=false\n";
    }
    if (graph_exec) hrx_graph_exec_release(graph_exec);
    if (graph) hrx_graph_release(graph);
    if (output) hrx_buffer_release(output);
    if (weight) hrx_buffer_release(weight);
    if (input) hrx_buffer_release(input);
    if (executable) hrx_executable_release(executable);
    if (stream) hrx_stream_release(stream);
    if (device) hrx_device_release(device);
    if (!check(hrx_gpu_shutdown(), "shutdown GPU")) ok = false;
    return ok ? 0 : 1;
}
