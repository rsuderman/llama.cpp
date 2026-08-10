#include "dispatch/dispatch-scheduler.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-hrx.h"
#include "ggml.h"
#include "graph/graph.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define REQUIRE(condition)                                                                           \
    do {                                                                                             \
        if (!(condition)) {                                                                          \
            std::fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, #condition); \
            std::abort();                                                                            \
        }                                                                                            \
    } while (false)

static hrx_buffer_t dummy_hrx_buffer(uintptr_t value) {
    return reinterpret_cast<hrx_buffer_t>(value);
}

static bool contains_value_id(const std::vector<ggml::hrx::ValueId> & ids, ggml::hrx::ValueId id) {
    for (const ggml::hrx::ValueId candidate : ids) {
        if (candidate == id) {
            return true;
        }
    }
    return false;
}

static void run_graph_import_checks() {
    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * a   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * out = ggml_add(ctx, a, a);
    REQUIRE(a != nullptr);
    REQUIRE(out != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 1);
    const ggml::hrx::GraphNode * node = &imported.graph.nodes().front();
    REQUIRE(node->op == GGML_OP_ADD);
    REQUIRE(node->inputs.size() == 2);
    REQUIRE(node->inputs[0] == node->inputs[1]);
    REQUIRE(ggml::hrx::DispatchScheduler::supports_node(imported.graph, node));

    const ggml::hrx::Value * a_value   = imported.graph.values().find_tensor(a);
    const ggml::hrx::Value * out_value = imported.graph.values().find_tensor(out);
    REQUIRE(a_value != nullptr);
    REQUIRE(out_value != nullptr);
    REQUIRE(a_value->kind == ggml::hrx::ValueKind::External);
    REQUIRE(out_value->kind == ggml::hrx::ValueKind::External);
    REQUIRE(!a_value->buffer.has_value());
    REQUIRE(!out_value->buffer.has_value());

    const std::vector<ggml::hrx::ValueId> external_ids = imported.graph.values().external_value_ids();
    REQUIRE(external_ids.size() == 2);
    REQUIRE(contains_value_id(external_ids, a_value->id));
    REQUIRE(contains_value_id(external_ids, out_value->id));

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(!scheduler.schedule_graph(imported.graph));

    REQUIRE(imported.graph.values().bind_buffer(a_value->id, { dummy_hrx_buffer(0x1000), 0, a_value->byte_count }));
    REQUIRE(imported.graph.values().bind_buffer(out_value->id, { dummy_hrx_buffer(0x2000), 0, out_value->byte_count }));
    REQUIRE(scheduler.schedule_graph(imported.graph));
    REQUIRE(scheduler.dispatches().size() == 1);
    REQUIRE(scheduler.dispatches().front().bindings.size() == 3);

    ggml_free(ctx);
}

static void run_transient_import_checks() {
    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * a   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * b   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * sum = ggml_add(ctx, a, b);
    ggml_tensor * out = ggml_sqr(ctx, sum);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(sum != nullptr);
    REQUIRE(out != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 2);
    REQUIRE(imported.graph.nodes()[0].op == GGML_OP_ADD);
    REQUIRE(imported.graph.nodes()[1].op == GGML_OP_SQR);

    const ggml::hrx::Value * a_value   = imported.graph.values().find_tensor(a);
    const ggml::hrx::Value * sum_value = imported.graph.values().find_tensor(sum);
    const ggml::hrx::Value * out_value = imported.graph.values().find_tensor(out);
    REQUIRE(a_value != nullptr);
    REQUIRE(sum_value != nullptr);
    REQUIRE(out_value != nullptr);
    REQUIRE(a_value->kind == ggml::hrx::ValueKind::External);
    REQUIRE(sum_value->kind == ggml::hrx::ValueKind::Transient);
    REQUIRE(out_value->kind == ggml::hrx::ValueKind::External);
    REQUIRE(
        !imported.graph.values().bind_buffer(sum_value->id, { dummy_hrx_buffer(0x3000), 0, sum_value->byte_count }));

    ggml_free(ctx);
}

static void run_add_f32() {
    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    constexpr int64_t element_count = 1024;
    ggml_tensor *     a             = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, element_count);
    ggml_tensor *     b             = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, element_count);
    ggml_tensor *     out           = ggml_add(ctx, a, b);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(out != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);

    std::vector<float> a_data(element_count);
    std::vector<float> b_data(element_count);
    std::vector<float> expected(element_count);
    for (int64_t i = 0; i < element_count; ++i) {
        a_data[i]   = static_cast<float>(i % 17) * 0.25f - 2.0f;
        b_data[i]   = static_cast<float>(i % 13) * -0.5f + 3.0f;
        expected[i] = a_data[i] + b_data[i];
    }

    ggml_backend_tensor_set(a, a_data.data(), 0, a_data.size() * sizeof(float));
    ggml_backend_tensor_set(b, b_data.data(), 0, b_data.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    std::vector<float> actual(element_count);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    for (int64_t i = 0; i < element_count; ++i) {
        REQUIRE(actual[i] == expected[i]);
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static void run_unsupported_op_fails() {
    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * a   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * out = ggml_sqr(ctx, a);
    REQUIRE(a != nullptr);
    REQUIRE(out != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);

    std::vector<float> input(8, 2.0f);
    ggml_backend_tensor_set(a, input.data(), 0, input.size() * sizeof(float));
    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

int main() {
    run_graph_import_checks();
    run_transient_import_checks();

    if (ggml_backend_hrx_get_device_count() == 0) {
        std::fprintf(stderr, "test skipped: no HRX devices available\n");
        return 0;
    }

    run_add_f32();
    run_unsupported_op_fails();
    return 0;
}
