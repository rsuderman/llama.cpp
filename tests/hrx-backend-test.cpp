#include "dispatch/command-program-bindings.h"
#include "dispatch/command-program-diagnostics.h"
#include "dispatch/command-program-resolver.h"
#include "dispatch/command-program.h"
#include "dispatch/dispatch-scheduler.h"
#include "dispatch_registration/dispatch-registry.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-hrx.h"
#include "ggml-impl.h"
#include "ggml.h"
#include "graph/graph-traversal.h"
#include "graph/graph.h"
#include "kernel-corpus/kernel-corpus.h"
#include "runtime/command-program-executor.h"
#include "runtime/graph-executor.h"
#include "runtime/graph-program-cache.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>
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

static bool command_program_verifies(const ggml::hrx::CommandProgram & program) {
    return ggml::hrx::verify_command_program(program, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151").valid();
}

static ggml::hrx::CommandProgram copy_command_program_shape(const ggml::hrx::CommandProgram & program) {
    ggml::hrx::CommandProgram copy;
    copy.commands   = program.commands;
    copy.transients = program.transients;
    return copy;
}

static bool status_contains(const ggml::hrx::Status & status, const char * text) {
    for (const std::string & message : status.errors()) {
        if (message.find(text) != std::string::npos) {
            return true;
        }
    }
    return false;
}

static bool string_contains(const std::string & value, const char * text) {
    return value.find(text) != std::string::npos;
}

static ggml::hrx::DispatchTarget test_dispatch_target() {
    return { "gfx1151" };
}

static const ggml::hrx::DispatchRegistry & test_dispatch_registry() {
    const ggml::hrx::DispatchRegistry * registry = ggml::hrx::find_dispatch_registry(test_dispatch_target());
    REQUIRE(registry != nullptr);
    return *registry;
}

static std::string kernel_name_for_id(uint64_t kernel_id) {
    const ggml::hrx::KernelResolveResult resolved =
        ggml::hrx::resolve_kernel_definition(ggml::hrx::get_qwen_kernel_corpus(), "gfx1151", kernel_id);
    REQUIRE(resolved.found());
    return ggml::hrx::kernel_definition_name(*resolved.definition);
}

static void require_compile_parameter(const ggml::hrx::Dispatch & dispatch,
                                      const char *                name,
                                      const std::string &         value) {
    const auto found = dispatch.kernel.compile_parameters.find(name);
    REQUIRE(found != dispatch.kernel.compile_parameters.end());
    REQUIRE(found->second == value);
}

static constexpr int64_t kQwenFlashHeadSize       = 128;
static constexpr int64_t kQwenRouterExpertCount   = 128;
static constexpr int64_t kQwenRouterRouteCount    = 8;
static constexpr int64_t kQwenMoeHiddenSize       = 2048;
static constexpr int64_t kQwenMoeIntermediateSize = 768;

static size_t qwen_expert_table_size(int64_t token_count) {
    return static_cast<size_t>(kQwenRouterExpertCount + kQwenRouterExpertCount * token_count) * sizeof(int32_t);
}

static size_t qwen_partition_table_size(int64_t token_count) {
    const int64_t assignment_count           = token_count * kQwenRouterRouteCount;
    const int64_t assignment_partition_count = (assignment_count + 31) / 32;
    return static_cast<size_t>(1 + assignment_partition_count + kQwenRouterExpertCount) * sizeof(int32_t);
}

static size_t qwen_routed_gate_up_f16_output_size(int64_t token_count) {
    return static_cast<size_t>(token_count * kQwenRouterRouteCount * kQwenMoeIntermediateSize) * sizeof(ggml_fp16_t);
}

static size_t qwen_routed_down_f16_output_size(int64_t token_count) {
    return static_cast<size_t>(token_count * kQwenRouterRouteCount * kQwenMoeHiddenSize) * sizeof(ggml_fp16_t);
}

static void set_qwen_flash_query_layout(ggml_tensor * tensor, int64_t head_count) {
    REQUIRE(tensor != nullptr);
    tensor->nb[0] = sizeof(float);
    tensor->nb[1] = static_cast<size_t>(head_count * kQwenFlashHeadSize) * sizeof(float);
    tensor->nb[2] = static_cast<size_t>(kQwenFlashHeadSize) * sizeof(float);
}

static void set_qwen_flash_key_value_layout(ggml_tensor * tensor, int64_t head_count) {
    REQUIRE(tensor != nullptr);
    tensor->nb[0] = sizeof(ggml_fp16_t);
    tensor->nb[1] = static_cast<size_t>(head_count * kQwenFlashHeadSize) * sizeof(ggml_fp16_t);
    tensor->nb[2] = static_cast<size_t>(kQwenFlashHeadSize) * sizeof(ggml_fp16_t);
}

static ggml_tensor * build_qwen_flash_attention_graph(ggml_context * ctx,
                                                      int64_t        query_token_count,
                                                      int64_t        key_value_token_count,
                                                      int64_t        query_head_count,
                                                      int64_t        key_value_head_count,
                                                      ggml_type      query_type     = GGML_TYPE_F32,
                                                      ggml_type      key_value_type = GGML_TYPE_F16,
                                                      bool           include_mask   = true,
                                                      bool           include_sinks  = false,
                                                      int64_t        head_size      = kQwenFlashHeadSize,
                                                      float          scale          = 1.0f / std::sqrt(128.0f),
                                                      float          max_bias       = 0.0f,
                                                      float          logit_softcap  = 0.0f) {
    ggml_tensor * query = ggml_new_tensor_3d(ctx, query_type, head_size, query_token_count, query_head_count);
    ggml_tensor * key = ggml_new_tensor_3d(ctx, key_value_type, head_size, key_value_token_count, key_value_head_count);
    ggml_tensor * value =
        ggml_new_tensor_3d(ctx, key_value_type, head_size, key_value_token_count, key_value_head_count);
    REQUIRE(query != nullptr);
    REQUIRE(key != nullptr);
    REQUIRE(value != nullptr);
    if (query_type == GGML_TYPE_F32) {
        set_qwen_flash_query_layout(query, query_head_count);
    }
    if (key_value_type == GGML_TYPE_F16) {
        set_qwen_flash_key_value_layout(key, key_value_head_count);
        set_qwen_flash_key_value_layout(value, key_value_head_count);
    }
    ggml_tensor * mask = nullptr;
    if (include_mask) {
        mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, key_value_token_count, query_token_count);
        REQUIRE(mask != nullptr);
    }
    ggml_tensor * output = ggml_flash_attn_ext(ctx, query, key, value, mask, scale, max_bias, logit_softcap);
    REQUIRE(output != nullptr);
    if (include_sinks) {
        ggml_tensor * sinks = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, query_head_count);
        REQUIRE(sinks != nullptr);
        ggml_flash_attn_ext_add_sinks(output, sinks);
    }
    return output;
}

static ggml_tensor * build_qwen_router_top8_graph(ggml_context *  ctx,
                                                  ggml_tensor *   logits,
                                                  ggml_tensor **  route_ids   = nullptr,
                                                  ggml_sort_order order       = GGML_SORT_ORDER_DESC,
                                                  int64_t         route_count = kQwenRouterRouteCount,
                                                  float           clamp_min   = 1.0e-7f) {
    ggml_tensor * probs = ggml_soft_max(ctx, logits);
    REQUIRE(probs != nullptr);
    ggml_tensor * probs_reshaped = ggml_reshape_3d(ctx, probs, 1, logits->ne[0], logits->ne[1]);
    REQUIRE(probs_reshaped != nullptr);
    ggml_tensor * argsort = ggml_argsort(ctx, probs, order);
    REQUIRE(argsort != nullptr);
    ggml_tensor * topk = ggml_view_2d(ctx, argsort, route_count, logits->ne[1], argsort->nb[1], 0);
    REQUIRE(topk != nullptr);
    if (route_ids != nullptr) {
        *route_ids = topk;
    }
    ggml_tensor * selected = ggml_get_rows(ctx, probs_reshaped, topk);
    REQUIRE(selected != nullptr);
    ggml_tensor * selected_reshaped = ggml_reshape_2d(ctx, selected, route_count, logits->ne[1]);
    REQUIRE(selected_reshaped != nullptr);
    ggml_tensor * sum = ggml_sum_rows(ctx, selected_reshaped);
    REQUIRE(sum != nullptr);
    ggml_tensor * clamped_sum = ggml_clamp(ctx, sum, clamp_min, std::numeric_limits<float>::infinity());
    REQUIRE(clamped_sum != nullptr);
    ggml_tensor * normalized = ggml_div(ctx, selected_reshaped, clamped_sum);
    REQUIRE(normalized != nullptr);
    ggml_tensor * output = ggml_reshape_3d(ctx, normalized, 1, route_count, logits->ne[1]);
    REQUIRE(output != nullptr);
    return output;
}

static std::vector<size_t> traversal_indices(const ggml::hrx::Graph & graph) {
    const ggml::hrx::GraphTraversalOrder order = ggml::hrx::GraphTraversalOrder::build(graph);
    std::vector<size_t>                  indices;
    indices.reserve(order.nodes().size());
    for (const ggml::hrx::GraphNode * node : order.nodes()) {
        size_t index = 0;
        REQUIRE(node != nullptr);
        REQUIRE(graph.index().node_index(node, index));
        indices.push_back(index);
    }
    return indices;
}

static size_t find_position(const std::vector<size_t> & indices, size_t node_index) {
    const std::vector<size_t>::const_iterator it = std::find(indices.begin(), indices.end(), node_index);
    REQUIRE(it != indices.end());
    return static_cast<size_t>(it - indices.begin());
}

static size_t producer_index_for_tensor(const ggml::hrx::Graph & graph, const ggml_tensor * tensor) {
    const ggml::hrx::Value * value = graph.values().find_tensor(tensor);
    REQUIRE(value != nullptr);
    const ggml::hrx::GraphNode * producer = graph.index().producer(value->id);
    REQUIRE(producer != nullptr);
    size_t index = 0;
    REQUIRE(graph.index().node_index(producer, index));
    return index;
}

static void run_status_checks() {
    ggml::hrx::Status status;
    REQUIRE(status.success());
    REQUIRE(status.errors().empty());

    status.log("first");
    REQUIRE(!status.success());
    REQUIRE(!status.errors().empty());
    REQUIRE(status.errors().size() == 1);
    REQUIRE(status.errors()[0] == "first");

    status.log("value %d", 7);
    REQUIRE(status.errors().size() == 2);
    REQUIRE(status.errors()[1] == "value 7");

    ggml::hrx::Status other;
    other.log("third");
    status.append(other);
    REQUIRE(status.errors().size() == 3);
    REQUIRE(status.errors()[2] == "third");
}

static void run_command_plan_metadata_checks() {
    const ggml::hrx::QwenMoeRoutingResourceMetadata routing = {
        4,
        8,
        128,
        128,
    };
    const ggml::hrx::CommandPlanResourceMetadata metadata = ggml::hrx::make_command_plan_resource_metadata(routing);

    REQUIRE(metadata.kind == ggml::hrx::CommandPlanResourceMetadataKind::QwenMoeRoutingResource);
    ggml::hrx::QwenMoeRoutingResourceMetadata decoded;
    REQUIRE(metadata.read(decoded));
    REQUIRE(decoded.token_count == routing.token_count);
    REQUIRE(decoded.route_count == routing.route_count);
    REQUIRE(decoded.route_stride == routing.route_stride);
    REQUIRE(decoded.expert_count == routing.expert_count);

    const ggml::hrx::CommandPlanResourceMetadata empty;
    REQUIRE(!empty.read(decoded));

    ggml::hrx::CommandPlanMetadata metadata_plan;
    ggml::hrx::Status              status;
    REQUIRE(metadata_plan.append_alternate_value(
        { ggml::hrx::ValueId(1), ggml::hrx::ValueId(2), GGML_TYPE_F16, 16, "alternate" }, status));
    REQUIRE(metadata_plan.append_alternate_value(
        { ggml::hrx::ValueId(1), ggml::hrx::ValueId(2), GGML_TYPE_F16, 16, "alternate" }, status));
    REQUIRE(metadata_plan.alternate_values().size() == 1);
    REQUIRE(metadata_plan.find_alternate_value(ggml::hrx::ValueId(1), GGML_TYPE_F16, 16) != nullptr);
    REQUIRE(metadata_plan.find_alternate_value(ggml::hrx::ValueId(1), GGML_TYPE_F32, 16) == nullptr);
    REQUIRE(metadata_plan.find_alternate_value(ggml::hrx::ValueId(1), GGML_TYPE_F16, 32) == nullptr);
    REQUIRE(!metadata_plan.append_alternate_value(
        { ggml::hrx::ValueId(1), ggml::hrx::ValueId(3), GGML_TYPE_F16, 16, "alternate" }, status));
    REQUIRE(!status.success());

    ggml::hrx::CommandPlanMetadata                bundle_plan;
    ggml::hrx::Status                             bundle_status;
    const ggml::hrx::CommandPlanQwenRoutingBundle bundle = {
        ggml::hrx::ValueId(10),
        ggml::hrx::ValueId(11),
        ggml::hrx::ValueId(12),
        ggml::hrx::ValueId(13),
        128,
        64,
        4,
        8,
        128,
        128,
    };
    REQUIRE(bundle_plan.append_qwen_routing_bundle(bundle, bundle_status));
    REQUIRE(bundle_plan.append_qwen_routing_bundle(bundle, bundle_status));
    REQUIRE(bundle_plan.qwen_routing_bundles().size() == 1);
    const ggml::hrx::CommandPlanQwenRoutingBundle * found_bundle =
        bundle_plan.find_qwen_routing_bundle(ggml::hrx::ValueId(10));
    REQUIRE(found_bundle != nullptr);
    REQUIRE(found_bundle->route_weights == ggml::hrx::ValueId(11));
    REQUIRE(found_bundle->expert_table == ggml::hrx::ValueId(12));
    REQUIRE(found_bundle->partition_table == ggml::hrx::ValueId(13));
    ggml::hrx::CommandPlanQwenRoutingBundle conflicting_bundle = bundle;
    conflicting_bundle.route_weights                           = ggml::hrx::ValueId(14);
    REQUIRE(!bundle_plan.append_qwen_routing_bundle(conflicting_bundle, bundle_status));
    REQUIRE(!bundle_status.success());
}

static bool has_dispatch_registration(const std::vector<ggml::hrx::DispatchRegistration> & registrations,
                                      const char *                                         name) {
    for (const ggml::hrx::DispatchRegistration & registration : registrations) {
        if (std::string(registration.name) == name) {
            return true;
        }
    }
    return false;
}

static bool match_test_single_dispatch(const ggml::hrx::DispatchMatchContext & context,
                                       ggml::hrx::DispatchMatch &              match) {
    ggml::hrx::Dispatch dispatch;
    dispatch.kernel.integer_parameters.emplace("route", 1);
    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_test_fused_dispatch(const ggml::hrx::DispatchMatchContext & context,
                                      ggml::hrx::DispatchMatch &              match) {
    ggml::hrx::Dispatch dispatch;
    dispatch.kernel.integer_parameters.emplace("route", 2);
    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_test_wrong_root_dispatch(const ggml::hrx::DispatchMatchContext & context,
                                           ggml::hrx::DispatchMatch &              match) {
    ggml::hrx::Dispatch dispatch;
    dispatch.kernel.integer_parameters.emplace("route", 3);
    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static void run_dispatch_registry_checks() {
    const ggml::hrx::DispatchRegistry & registry = test_dispatch_registry();
    REQUIRE(ggml::hrx::find_dispatch_registry({ "gfx1100" }) != nullptr);
    REQUIRE(ggml::hrx::find_dispatch_registry({ "gfx1151" }) != nullptr);
    REQUIRE(ggml::hrx::find_dispatch_registry({ "gfx0000" }) == nullptr);

    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_ADD), "common.add_f32"));
    REQUIRE(
        has_dispatch_registration(registry.registrations_for_root(GGML_OP_MUL_MAT), "qwen.matmul.dense_q4k_f16_wmma"));
    REQUIRE(
        has_dispatch_registration(registry.registrations_for_root(GGML_OP_MUL_MAT), "qwen.matmul.dense_q6k_f16_wmma"));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_MUL_MAT),
                                      "qwen.matmul.router_projection_f32_four_row_wave32"));
    REQUIRE(
        has_dispatch_registration(registry.registrations_for_root(GGML_OP_RMS_NORM), "qwen.rmsnorm_f32.mul_weight"));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_FLASH_ATTN_EXT),
                                      "qwen.flash_attention_f32_f16_wmma"));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_SOFT_MAX), "qwen.router.top8_f32"));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_MUL_MAT_ID),
                                      "qwen.moe.routed_gate_up_swiglu_q4k_f16_wmma"));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_MUL_MAT_ID),
                                      "qwen.moe.routed_down_q4k_f16_wmma_grouped"));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_MUL_MAT_ID),
                                      "qwen.moe.routed_down_q6k_f16_wmma_grouped"));
    REQUIRE(registry.single_op_registrations().size() >= 5);

    ggml::hrx::DispatchRegistryBuilder builder;
    builder.add({
        "test.single_add",
        GGML_OP_ADD,
        ggml::hrx::DispatchMatchKind::SingleOp,
        1000,
        ggml::hrx::DispatchSource::Common,
        match_test_single_dispatch,
    });
    builder.add({
        "test.fused_add",
        GGML_OP_ADD,
        ggml::hrx::DispatchMatchKind::Fused,
        0,
        ggml::hrx::DispatchSource::Common,
        match_test_fused_dispatch,
    });
    builder.add({
        "test.wrong_root",
        GGML_OP_MUL_MAT,
        ggml::hrx::DispatchMatchKind::Fused,
        2000,
        ggml::hrx::DispatchSource::Common,
        match_test_wrong_root_dispatch,
    });
    const ggml::hrx::DispatchRegistry ordering_registry = builder.build();

    ggml::hrx::Graph graph;
    graph.add_node(GGML_OP_ADD, ggml::hrx::ValueId(0), {});
    REQUIRE(graph.build_index().success());

    const std::vector<bool>               covered_nodes(graph.nodes().size(), false);
    const ggml::hrx::CommandPlan          plan;
    const ggml::hrx::DispatchMatchContext context = {
        graph, &graph.nodes().front(),
        0,     covered_nodes,
        plan,  ggml::hrx::ValueId(static_cast<int32_t>(graph.values().size())),
    };
    ggml::hrx::DispatchMatch match;
    REQUIRE(ordering_registry.match(context, match));
    REQUIRE(match.dispatches.size() == 1);
    REQUIRE(match.dispatches.front().kernel.integer_parameters.at("route") == 2);
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
    REQUIRE(ggml::hrx::DispatchScheduler::supports_node(imported.graph, node, test_dispatch_target()));

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
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 1);
    REQUIRE(scheduler.plan().dispatches.front().bindings.size() == 3);
    REQUIRE(scheduler.plan().dispatches.front().bindings[0].length == a_value->byte_count);

    ggml::hrx::CommandProgramBindings missing_bindings =
        ggml::hrx::CommandProgramBindings::from_value_map(imported.graph.values());
    REQUIRE(!missing_bindings.valid());

    REQUIRE(imported.graph.values().bind_buffer(a_value->id, { dummy_hrx_buffer(0x1000), 0, a_value->byte_count }));
    ggml::hrx::CommandProgramBindings partial_bindings =
        ggml::hrx::CommandProgramBindings::from_value_map(imported.graph.values());
    REQUIRE(!partial_bindings.valid());

    REQUIRE(imported.graph.values().bind_buffer(out_value->id, { dummy_hrx_buffer(0x2000), 0, 0 }));
    ggml::hrx::CommandProgramBindings empty_runtime_binding =
        ggml::hrx::CommandProgramBindings::from_value_map(imported.graph.values());
    REQUIRE(!empty_runtime_binding.valid());

    REQUIRE(imported.graph.values().bind_buffer(out_value->id, { dummy_hrx_buffer(0x2000), 0, out_value->byte_count }));
    ggml::hrx::CommandProgramBindings runtime_bindings =
        ggml::hrx::CommandProgramBindings::from_value_map(imported.graph.values());
    REQUIRE(runtime_bindings.valid());
    REQUIRE(runtime_bindings.bindings().size() == 2);
    const ggml::hrx::CommandProgramBinding * a_binding = runtime_bindings.find(a_value->id);
    REQUIRE(a_binding != nullptr);
    REQUIRE(a_binding->buffer == dummy_hrx_buffer(0x1000));
    REQUIRE(a_binding->offset == 0);
    REQUIRE(a_binding->length == a_value->byte_count);
    REQUIRE(runtime_bindings.find(ggml::hrx::ValueId(123456)) == nullptr);

    const ggml::hrx::CommandProgramBindingsFingerprint runtime_fingerprint =
        ggml::hrx::command_program_bindings_fingerprint(runtime_bindings);
    REQUIRE(!runtime_fingerprint.value.empty());

    ggml::hrx::ValueMap changed_identity = imported.graph.values();
    REQUIRE(changed_identity.bind_buffer(
        a_value->id, { dummy_hrx_buffer(0x1000), 0, a_value->byte_count, 1, 0, a_value->byte_count }));
    const ggml::hrx::CommandProgramBindings changed_identity_bindings =
        ggml::hrx::CommandProgramBindings::from_value_map(changed_identity);
    REQUIRE(changed_identity_bindings.valid());
    REQUIRE(ggml::hrx::command_program_bindings_fingerprint(changed_identity_bindings).value !=
            runtime_fingerprint.value);

    ggml::hrx::ValueMap changed_generation = imported.graph.values();
    REQUIRE(changed_generation.bind_buffer(
        a_value->id, { dummy_hrx_buffer(0x1000), 0, a_value->byte_count, 0, 1, a_value->byte_count }));
    const ggml::hrx::CommandProgramBindings changed_generation_bindings =
        ggml::hrx::CommandProgramBindings::from_value_map(changed_generation);
    REQUIRE(changed_generation_bindings.valid());
    REQUIRE(ggml::hrx::command_program_bindings_fingerprint(changed_generation_bindings).value !=
            runtime_fingerprint.value);

    ggml::hrx::ValueMap changed_capacity = imported.graph.values();
    REQUIRE(changed_capacity.bind_buffer(
        a_value->id, { dummy_hrx_buffer(0x1000), 0, a_value->byte_count, 0, 0, a_value->byte_count + 256 }));
    const ggml::hrx::CommandProgramBindings changed_capacity_bindings =
        ggml::hrx::CommandProgramBindings::from_value_map(changed_capacity);
    REQUIRE(changed_capacity_bindings.valid());
    REQUIRE(ggml::hrx::command_program_bindings_fingerprint(changed_capacity_bindings).value !=
            runtime_fingerprint.value);

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 1);
    const ggml::hrx::Command & command = commands.commands.front();
    REQUIRE(command.ordinal == 0);
    REQUIRE(command.kind == ggml::hrx::CommandKind::Kernel);
    REQUIRE(command.kernel.kernel_id != ggml::hrx::kUncatalogedKernelId);
    REQUIRE(command.bindings.size() == 3);
    REQUIRE(command.bindings[0].name == "a");
    REQUIRE(command.bindings[0].origin == ggml::hrx::CommandBindingOrigin::GraphValue);
    REQUIRE(command.bindings[0].access == ggml::hrx::ResourceAccess::Read);
    REQUIRE(command.bindings[1].name == "b");
    REQUIRE(command.bindings[1].origin == ggml::hrx::CommandBindingOrigin::GraphValue);
    REQUIRE(command.bindings[1].access == ggml::hrx::ResourceAccess::Read);
    REQUIRE(command.bindings[2].name == "output");
    REQUIRE(command.bindings[2].origin == ggml::hrx::CommandBindingOrigin::GraphValue);
    REQUIRE(command.bindings[2].access == ggml::hrx::ResourceAccess::ReadWrite);
    REQUIRE(command_program_verifies(commands));

    const ggml::hrx::PreparedCommand default_prepared_command;
    REQUIRE(default_prepared_command.kind == ggml::hrx::CommandKind::Invalid);
    REQUIRE(ggml::hrx::command_kind_name(ggml::hrx::CommandKind::Invalid) == "Invalid");
    REQUIRE(ggml::hrx::command_kind_name(ggml::hrx::CommandKind::Kernel) == "Kernel");
    REQUIRE(ggml::hrx::command_kind_name(static_cast<ggml::hrx::CommandKind>(255)) == "Unknown(255)");
    REQUIRE(ggml::hrx::command_binding_origin_name(ggml::hrx::CommandBindingOrigin::GraphValue) == "GraphValue");
    REQUIRE(ggml::hrx::command_binding_origin_name(ggml::hrx::CommandBindingOrigin::Transient) == "Transient");
    REQUIRE(ggml::hrx::command_binding_origin_name(static_cast<ggml::hrx::CommandBindingOrigin>(255)) ==
            "Unknown(255)");
    REQUIRE(ggml::hrx::resource_access_name(ggml::hrx::ResourceAccess::Read) == "Read");
    REQUIRE(ggml::hrx::resource_access_name(ggml::hrx::ResourceAccess::Write) == "Write");
    REQUIRE(ggml::hrx::resource_access_name(ggml::hrx::ResourceAccess::ReadWrite) == "ReadWrite");
    REQUIRE(ggml::hrx::resource_access_name(static_cast<ggml::hrx::ResourceAccess>(255)) == "Unknown(255)");

    const std::string binding_text = ggml::hrx::format_command_binding(command.bindings[0]);
    REQUIRE(string_contains(binding_text, "binding a"));
    REQUIRE(string_contains(binding_text, "value="));
    REQUIRE(string_contains(binding_text, "origin=GraphValue"));
    REQUIRE(string_contains(binding_text, "access=Read"));
    REQUIRE(string_contains(binding_text, "range=[0, "));
    REQUIRE(string_contains(binding_text, std::to_string(a_value->byte_count).c_str()));

    const std::string command_text = ggml::hrx::format_command(command);
    REQUIRE(string_contains(command_text, "command 0"));
    REQUIRE(string_contains(command_text, "kind=Kernel"));
    REQUIRE(string_contains(command_text, "kernel_id="));
    REQUIRE(string_contains(command_text, "bindings=3"));
    REQUIRE(string_contains(command_text, "deps=0"));

    const std::string program_text = ggml::hrx::format_command_program(commands);
    REQUIRE(string_contains(program_text, "command_program commands=1"));
    REQUIRE(string_contains(program_text, "command 0"));
    REQUIRE(string_contains(program_text, "binding a"));
    REQUIRE(string_contains(program_text, "binding b"));
    REQUIRE(string_contains(program_text, "binding output"));

    ggml::hrx::ResolvedCommandProgram resolved =
        ggml::hrx::resolve_command_program_bindings(commands, runtime_bindings);
    REQUIRE(resolved.valid());
    REQUIRE(resolved.commands.size() == 1);
    REQUIRE(resolved.commands.front().ordinal == command.ordinal);
    REQUIRE(resolved.commands.front().kind == command.kind);
    REQUIRE(resolved.commands.front().kernel.kernel_id == command.kernel.kernel_id);
    REQUIRE(resolved.commands.front().bindings.size() == 3);
    REQUIRE(resolved.commands.front().bindings[0].binding.name == "a");
    REQUIRE(resolved.commands.front().bindings[0].ref.buffer == dummy_hrx_buffer(0x1000));
    REQUIRE(resolved.commands.front().bindings[0].ref.offset == 0);
    REQUIRE(resolved.commands.front().bindings[0].ref.length == a_value->byte_count);
    REQUIRE(resolved.commands.front().bindings[1].binding.name == "b");
    REQUIRE(resolved.commands.front().bindings[1].ref.buffer == dummy_hrx_buffer(0x1000));
    REQUIRE(resolved.commands.front().bindings[1].ref.offset == 0);
    REQUIRE(resolved.commands.front().bindings[1].ref.length == a_value->byte_count);
    REQUIRE(resolved.commands.front().bindings[2].binding.name == "output");
    REQUIRE(resolved.commands.front().bindings[2].ref.buffer == dummy_hrx_buffer(0x2000));
    REQUIRE(resolved.commands.front().bindings[2].ref.offset == 0);
    REQUIRE(resolved.commands.front().bindings[2].ref.length == out_value->byte_count);

    ggml::hrx::CommandProgram offset_command           = copy_command_program_shape(commands);
    offset_command.commands.front().bindings[0].offset = 4;
    offset_command.commands.front().bindings[0].length = 8;
    ggml::hrx::ValueMap offset_values                  = imported.graph.values();
    REQUIRE(offset_values.bind_buffer(a_value->id, { dummy_hrx_buffer(0x3000), 16, a_value->byte_count }));
    REQUIRE(offset_values.bind_buffer(out_value->id, { dummy_hrx_buffer(0x4000), 32, out_value->byte_count }));
    const ggml::hrx::CommandProgramBindings offset_bindings =
        ggml::hrx::CommandProgramBindings::from_value_map(offset_values);
    REQUIRE(offset_bindings.valid());
    resolved = ggml::hrx::resolve_command_program_bindings(offset_command, offset_bindings);
    REQUIRE(resolved.valid());
    REQUIRE(resolved.commands.front().bindings[0].ref.buffer == dummy_hrx_buffer(0x3000));
    REQUIRE(resolved.commands.front().bindings[0].ref.offset == 20);
    REQUIRE(resolved.commands.front().bindings[0].ref.length == 8);

    resolved = ggml::hrx::resolve_command_program_bindings(commands, missing_bindings);
    REQUIRE(!resolved.valid());
    REQUIRE(status_contains(resolved.status, "is not bound"));
    REQUIRE(status_contains(resolved.status, "binding output"));
    REQUIRE(status_contains(resolved.status, "value="));

    resolved = ggml::hrx::resolve_command_program_bindings(commands, partial_bindings);
    REQUIRE(!resolved.valid());
    REQUIRE(status_contains(resolved.status, "is not bound"));

    resolved = ggml::hrx::resolve_command_program_bindings(commands, empty_runtime_binding);
    REQUIRE(!resolved.valid());
    REQUIRE(status_contains(resolved.status, "empty binding"));

    ggml::hrx::ValueMap null_values = imported.graph.values();
    REQUIRE(null_values.bind_buffer(a_value->id, { nullptr, 0, a_value->byte_count }));
    REQUIRE(null_values.bind_buffer(out_value->id, { dummy_hrx_buffer(0x2000), 0, out_value->byte_count }));
    const ggml::hrx::CommandProgramBindings null_bindings =
        ggml::hrx::CommandProgramBindings::from_value_map(null_values);
    REQUIRE(!null_bindings.valid());
    resolved = ggml::hrx::resolve_command_program_bindings(commands, null_bindings);
    REQUIRE(!resolved.valid());
    REQUIRE(status_contains(resolved.status, "null buffer"));
    REQUIRE(status_contains(resolved.status, "binding a"));

    ggml::hrx::CommandProgram empty_resolve_binding           = copy_command_program_shape(commands);
    empty_resolve_binding.commands.front().bindings[0].length = 0;
    resolved = ggml::hrx::resolve_command_program_bindings(empty_resolve_binding, runtime_bindings);
    REQUIRE(!resolved.valid());
    REQUIRE(status_contains(resolved.status, "empty range"));
    REQUIRE(status_contains(resolved.status, "range=[0, 0)"));

    ggml::hrx::CommandProgram out_of_range_binding           = copy_command_program_shape(commands);
    out_of_range_binding.commands.front().bindings[0].offset = a_value->byte_count;
    out_of_range_binding.commands.front().bindings[0].length = 4;
    resolved = ggml::hrx::resolve_command_program_bindings(out_of_range_binding, runtime_bindings);
    REQUIRE(!resolved.valid());
    REQUIRE(status_contains(resolved.status, "outside runtime binding length"));
    REQUIRE(status_contains(resolved.status, "binding a"));

    ggml::hrx::CommandProgram unsupported_origin           = copy_command_program_shape(commands);
    unsupported_origin.commands.front().bindings[0].origin = static_cast<ggml::hrx::CommandBindingOrigin>(255);
    resolved = ggml::hrx::resolve_command_program_bindings(unsupported_origin, runtime_bindings);
    REQUIRE(!resolved.valid());
    REQUIRE(status_contains(resolved.status, "unsupported binding origin"));
    REQUIRE(status_contains(resolved.status, "origin=Unknown(255)"));
    ggml::hrx::VerificationResult verification =
        ggml::hrx::verify_command_program(unsupported_origin, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(status_contains(verification.status, "unsupported binding origin"));
    REQUIRE(status_contains(verification.status, "origin=Unknown(255)"));

    ggml::hrx::CommandProgram invalid_kernel         = copy_command_program_shape(commands);
    invalid_kernel.commands.front().kernel.kernel_id = ggml::hrx::kUncatalogedKernelId;
    verification = ggml::hrx::verify_command_program(invalid_kernel, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(status_contains(verification.status, "command 0"));
    REQUIRE(status_contains(verification.status, "kernel_id="));

    ggml::hrx::CommandProgram empty_bindings = copy_command_program_shape(commands);
    empty_bindings.commands.front().bindings.clear();
    verification = ggml::hrx::verify_command_program(empty_bindings, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(status_contains(verification.status, "bindings=0"));

    ggml::hrx::CommandProgram empty_binding           = copy_command_program_shape(commands);
    empty_binding.commands.front().bindings[0].length = 0;
    verification = ggml::hrx::verify_command_program(empty_binding, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(status_contains(verification.status, "binding a"));
    REQUIRE(status_contains(verification.status, "range=[0, 0)"));

    ggml::hrx::CommandProgram invalid_value          = copy_command_program_shape(commands);
    invalid_value.commands.front().bindings[0].value = ggml::hrx::ValueId(-1);
    verification = ggml::hrx::verify_command_program(invalid_value, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(status_contains(verification.status, "value=-1"));

    ggml::hrx::CommandProgram forward_dependency = copy_command_program_shape(commands);
    forward_dependency.commands.front().dependencies.push_back(0);
    verification =
        ggml::hrx::verify_command_program(forward_dependency, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(status_contains(verification.status, "forward dependency 0"));

    ggml::hrx::CommandProgram wrong_binding_name         = copy_command_program_shape(commands);
    wrong_binding_name.commands.front().bindings[0].name = "wrong";
    verification =
        ggml::hrx::verify_command_program(wrong_binding_name, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(status_contains(verification.status, "binding wrong"));

    ggml::hrx::CommandProgram wrong_binding_access           = copy_command_program_shape(commands);
    wrong_binding_access.commands.front().bindings[0].access = ggml::hrx::ResourceAccess::Write;
    verification =
        ggml::hrx::verify_command_program(wrong_binding_access, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(status_contains(verification.status, "access=Write"));

    ggml_hrx_loom_jit_amdgpu *                      jit             = nullptr;
    const ggml::hrx::KernelCorpus &                 corpus          = ggml::hrx::get_qwen_kernel_corpus();
    const ggml::hrx::CommandProgramExecutionContext prepare_context = {
        nullptr, nullptr, "gfx1151", &corpus, &jit, nullptr, nullptr,
    };

    ggml::hrx::PreparedCommandProgram prepared =
        ggml::hrx::prepare_command_program(prepare_context, invalid_kernel, runtime_bindings);
    REQUIRE(!prepared.valid());
    REQUIRE(status_contains(prepared.status, "kernel_id="));

    prepared = ggml::hrx::prepare_command_program(prepare_context, commands, missing_bindings);
    REQUIRE(!prepared.valid());
    REQUIRE(status_contains(prepared.status, "is not bound"));

    prepared = ggml::hrx::prepare_command_program(prepare_context, commands, runtime_bindings);
    REQUIRE(!prepared.valid());
    REQUIRE(status_contains(prepared.status, "missing HRX device"));

    ggml_free(ctx);
}

static void run_graph_index_checks() {
    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 1);
    ggml_tensor * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
    REQUIRE(input != nullptr);
    REQUIRE(weight != nullptr);
    ggml_tensor * rms = ggml_rms_norm(ctx, input, 0.000001f);
    REQUIRE(rms != nullptr);
    ggml_tensor * out = ggml_mul(ctx, rms, weight);
    REQUIRE(out != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.has_index());
    REQUIRE(imported.graph.nodes().size() == 2);
    const ggml::hrx::GraphNode * rms_node = &imported.graph.nodes()[0];
    const ggml::hrx::GraphNode * mul_node = &imported.graph.nodes()[1];
    REQUIRE(rms_node->op == GGML_OP_RMS_NORM);
    REQUIRE(mul_node->op == GGML_OP_MUL);
    const ggml::hrx::RmsNormParams * rms_params = ggml::hrx::op_params_as<ggml::hrx::RmsNormParams>(rms_node->params);
    REQUIRE(rms_params != nullptr);
    REQUIRE(rms_params->eps == 0.000001f);
    REQUIRE(imported.graph.index().producer(rms_node->output) == rms_node);
    REQUIRE(imported.graph.index().producer(mul_node->output) == mul_node);
    REQUIRE(imported.graph.index().has_single_consumer(rms_node->output));
    const std::vector<const ggml::hrx::GraphNode *> & consumers = imported.graph.index().consumers(rms_node->output);
    REQUIRE(consumers.size() == 1);
    REQUIRE(consumers.front() == mul_node);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 1);

    ggml_free(ctx);
}

static void run_graph_traversal_checks() {
    {
        ggml_init_params params = {};
        params.mem_size         = 256 * 1024;
        params.no_alloc         = true;
        ggml_context * ctx      = ggml_init(params);
        REQUIRE(ctx != nullptr);

        ggml_tensor * a    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_tensor * b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_tensor * c    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_tensor * d    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_tensor * out0 = ggml_add(ctx, a, b);
        ggml_tensor * out1 = ggml_add(ctx, c, d);
        REQUIRE(a != nullptr);
        REQUIRE(b != nullptr);
        REQUIRE(c != nullptr);
        REQUIRE(d != nullptr);
        REQUIRE(out0 != nullptr);
        REQUIRE(out1 != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, out0);
        ggml_build_forward_expand(graph, out1);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        REQUIRE(imported.graph.nodes().size() == 2);
        REQUIRE(imported.graph.nodes()[0].output == imported.graph.values().find_tensor(out0)->id);
        REQUIRE(imported.graph.nodes()[1].output == imported.graph.values().find_tensor(out1)->id);

        const std::vector<size_t> order = traversal_indices(imported.graph);
        REQUIRE(order.size() == 2);
        REQUIRE(order[0] == 0);
        REQUIRE(order[1] == 1);

        ggml_free(ctx);
    }

    {
        ggml_init_params params = {};
        params.mem_size         = 256 * 1024;
        params.no_alloc         = true;
        ggml_context * ctx      = ggml_init(params);
        REQUIRE(ctx != nullptr);

        ggml_tensor * a       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_tensor * b       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_tensor * add_out = ggml_add(ctx, a, b);
        ggml_tensor * weight  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
        ggml_tensor * input   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
        ggml_tensor * matmul  = ggml_mul_mat(ctx, weight, input);
        REQUIRE(a != nullptr);
        REQUIRE(b != nullptr);
        REQUIRE(add_out != nullptr);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        REQUIRE(matmul != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, add_out);
        ggml_build_forward_expand(graph, matmul);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        REQUIRE(imported.graph.nodes().size() == 2);
        REQUIRE(imported.graph.nodes()[0].op == GGML_OP_ADD);
        REQUIRE(imported.graph.nodes()[1].op == GGML_OP_MUL_MAT);

        const std::vector<size_t> order = traversal_indices(imported.graph);
        REQUIRE(order.size() == 2);
        REQUIRE(order[0] == 1);
        REQUIRE(order[1] == 0);

        ggml_free(ctx);
    }

    {
        ggml_init_params params = {};
        params.mem_size         = 256 * 1024;
        params.no_alloc         = true;
        ggml_context * ctx      = ggml_init(params);
        REQUIRE(ctx != nullptr);

        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 1);
        ggml_tensor * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
        ggml_tensor * rms    = ggml_rms_norm(ctx, input, 0.000001f);
        ggml_tensor * out    = ggml_mul(ctx, rms, weight);
        REQUIRE(input != nullptr);
        REQUIRE(weight != nullptr);
        REQUIRE(rms != nullptr);
        REQUIRE(out != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, out);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        REQUIRE(imported.graph.nodes().size() == 2);
        REQUIRE(imported.graph.nodes()[0].op == GGML_OP_RMS_NORM);
        REQUIRE(imported.graph.nodes()[1].op == GGML_OP_MUL);

        const std::vector<size_t> order = traversal_indices(imported.graph);
        REQUIRE(order.size() == 2);
        REQUIRE(order[0] == 0);
        REQUIRE(order[1] == 1);

        ggml_free(ctx);
    }

    {
        ggml_init_params params = {};
        params.mem_size         = 256 * 1024;
        params.no_alloc         = true;
        ggml_context * ctx      = ggml_init(params);
        REQUIRE(ctx != nullptr);

        ggml_tensor * a     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_tensor * b     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_tensor * c     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_tensor * left  = ggml_add(ctx, a, b);
        ggml_tensor * right = ggml_mul(ctx, a, c);
        ggml_tensor * join  = ggml_add(ctx, left, right);
        REQUIRE(a != nullptr);
        REQUIRE(b != nullptr);
        REQUIRE(c != nullptr);
        REQUIRE(left != nullptr);
        REQUIRE(right != nullptr);
        REQUIRE(join != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, join);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        REQUIRE(imported.graph.nodes().size() == 3);

        const size_t left_index  = producer_index_for_tensor(imported.graph, left);
        const size_t right_index = producer_index_for_tensor(imported.graph, right);
        const size_t join_index  = producer_index_for_tensor(imported.graph, join);

        const std::vector<size_t> order = traversal_indices(imported.graph);
        REQUIRE(order.size() == 3);
        REQUIRE(find_position(order, join_index) > find_position(order, left_index));
        REQUIRE(find_position(order, join_index) > find_position(order, right_index));

        ggml_free(ctx);
    }
}

static void schedule_single_matmul_command(ggml_context * ctx,
                                           ggml_tensor *  output,
                                           const char *   expected_kernel_name,
                                           int64_t        expected_token_count,
                                           int64_t        expected_input_size,
                                           int64_t        expected_output_size) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 1);
    REQUIRE(imported.graph.nodes()[0].op == GGML_OP_MUL_MAT);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 1);

    const ggml::hrx::Dispatch & dispatch    = scheduler.plan().dispatches.front();
    const std::string           kernel_name = kernel_name_for_id(dispatch.kernel.kernel_id);
    REQUIRE(kernel_name == expected_kernel_name);
    REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == expected_token_count);
    REQUIRE(dispatch.bindings.size() == 3);
    require_compile_parameter(dispatch, "qwen3_moe.workload.token_capacity", std::to_string(expected_token_count));
    if (string_contains(kernel_name, "dense_linear")) {
        require_compile_parameter(dispatch, "qwen3_moe.dense_quantized.input_size",
                                  std::to_string(expected_input_size));
        require_compile_parameter(dispatch, "qwen3_moe.dense_quantized.output_size",
                                  std::to_string(expected_output_size));
        require_compile_parameter(dispatch, "qwen3_moe.dense_quantized.output_accumulation", "0");
    } else {
        require_compile_parameter(dispatch, "qwen3_moe.model.hidden_size", std::to_string(expected_input_size));
        require_compile_parameter(dispatch, "qwen3_moe.router.expert_count", std::to_string(expected_output_size));
    }

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 1);
    REQUIRE(command_program_verifies(commands));
    REQUIRE(commands.commands.front().bindings.size() == 3);
    REQUIRE(commands.commands.front().bindings[0].name == "input");
    REQUIRE(commands.commands.front().bindings[1].name == "weight");
    REQUIRE(commands.commands.front().bindings[2].name == "output");
}

static bool matmul_graph_is_supported(ggml_context * ctx, ggml_tensor * output) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    return ggml::hrx::DispatchScheduler::can_schedule_graph(imported.graph, test_dispatch_target());
}

static bool graph_is_supported(ggml_context * ctx, ggml_tensor * output) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    return ggml::hrx::DispatchScheduler::can_schedule_graph(imported.graph, test_dispatch_target());
}

static void schedule_qwen_flash_attention_command(ggml_context * ctx, ggml_tensor * output) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 1);
    REQUIRE(imported.graph.nodes()[0].op == GGML_OP_FLASH_ATTN_EXT);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 1);

    const ggml::hrx::Dispatch & dispatch    = scheduler.plan().dispatches.front();
    const std::string           kernel_name = kernel_name_for_id(dispatch.kernel.kernel_id);
    REQUIRE(kernel_name == "qwen3_moe:qwen3_moe_flash_attention_f32_f16_wmma");
    REQUIRE(dispatch.kernel.integer_parameters.at("query_token_count") == 4);
    REQUIRE(dispatch.kernel.integer_parameters.at("key_value_token_count") == 8);
    REQUIRE(dispatch.bindings.size() == 5);
    require_compile_parameter(dispatch, "qwen3_moe.attention.query_head_count", "4");
    require_compile_parameter(dispatch, "qwen3_moe.attention.key_value_head_count", "2");
    require_compile_parameter(dispatch, "qwen3_moe.workload.token_capacity", "4");

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 1);
    REQUIRE(command_program_verifies(commands));
    REQUIRE(commands.commands.front().bindings.size() == 5);
    REQUIRE(commands.commands.front().bindings[0].name == "query");
    REQUIRE(commands.commands.front().bindings[1].name == "key");
    REQUIRE(commands.commands.front().bindings[2].name == "value");
    REQUIRE(commands.commands.front().bindings[3].name == "mask");
    REQUIRE(commands.commands.front().bindings[4].name == "output");
}

static void run_qwen_flash_attention_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size         = 4 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    {
        ggml_tensor * output = build_qwen_flash_attention_graph(ctx, 4, 8, 4, 2);
        schedule_qwen_flash_attention_command(ctx, output);
    }
    {
        ggml_tensor * output = build_qwen_flash_attention_graph(ctx, 4, 8, 4, 2, GGML_TYPE_F32, GGML_TYPE_F32);
        REQUIRE(!graph_is_supported(ctx, output));
    }
    {
        ggml_tensor * output = build_qwen_flash_attention_graph(ctx, 4, 8, 4, 2, GGML_TYPE_F32, GGML_TYPE_F16, false);
        REQUIRE(!graph_is_supported(ctx, output));
    }
    {
        ggml_tensor * output =
            build_qwen_flash_attention_graph(ctx, 4, 8, 4, 2, GGML_TYPE_F32, GGML_TYPE_F16, true, true);
        REQUIRE(!graph_is_supported(ctx, output));
    }
    {
        ggml_tensor * output = build_qwen_flash_attention_graph(ctx, 4, 8, 4, 2, GGML_TYPE_F32, GGML_TYPE_F16, true,
                                                                false, 64, 1.0f / std::sqrt(64.0f));
        REQUIRE(!graph_is_supported(ctx, output));
    }
    {
        ggml_tensor * output = build_qwen_flash_attention_graph(ctx, 4, 8, 4, 2, GGML_TYPE_F32, GGML_TYPE_F16, true,
                                                                false, kQwenFlashHeadSize, 1.0f);
        REQUIRE(!graph_is_supported(ctx, output));
    }
    {
        ggml_tensor * output =
            build_qwen_flash_attention_graph(ctx, 4, 8, 4, 2, GGML_TYPE_F32, GGML_TYPE_F16, true, false,
                                             kQwenFlashHeadSize, 1.0f / std::sqrt(128.0f), 1.0f);
        REQUIRE(!graph_is_supported(ctx, output));
    }

    ggml_free(ctx);
}

static void run_qwen_matmul_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size         = 2 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "qwen3_moe:qwen3_moe_dense_linear_q4k_f16_wmma", 4, 2048, 128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 2);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "qwen3_moe:qwen3_moe_dense_linear_q6k_f16_wmma", 2, 2048, 128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "qwen3_moe:qwen3_moe_router_projection_f32_four_row_wave32", 4,
                                       2048, 128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        REQUIRE(!matmul_graph_is_supported(ctx, output));
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2048, 4);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        REQUIRE(!matmul_graph_is_supported(ctx, output));
    }

    {
        ggml_tensor * weight = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2048, 128, 2);
        ggml_tensor * input  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2048, 4, 2);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        REQUIRE(!matmul_graph_is_supported(ctx, output));
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 4);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        REQUIRE(!matmul_graph_is_supported(ctx, output));
    }

    ggml_free(ctx);
}

static void schedule_qwen_router_top8_command(ggml_context * ctx, ggml_tensor * output, ggml_tensor * route_ids) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 10);

    const ggml::hrx::Value * route_ids_value = imported.graph.values().find_tensor(route_ids);
    const ggml::hrx::Value * output_value    = imported.graph.values().find_tensor(output);
    REQUIRE(route_ids_value != nullptr);
    REQUIRE(output_value != nullptr);
    REQUIRE(route_ids_value->kind == ggml::hrx::ValueKind::Transient);
    REQUIRE(output_value->kind == ggml::hrx::ValueKind::External);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 3);
    REQUIRE(scheduler.plan().transients.size() == 2);

    const int64_t                           token_count           = output->ne[2];
    const size_t                            route_id_length       = static_cast<size_t>(token_count) * route_ids->nb[1];
    const size_t                            expert_table_bytes    = qwen_expert_table_size(token_count);
    const size_t                            partition_table_bytes = qwen_partition_table_size(token_count);
    const ggml::hrx::CommandPlanTransient & expert_table_transient    = scheduler.plan().transients[0];
    const ggml::hrx::CommandPlanTransient & partition_table_transient = scheduler.plan().transients[1];
    REQUIRE(expert_table_transient.value.value == static_cast<int32_t>(imported.graph.values().size()));
    REQUIRE(expert_table_transient.name == "qwen.router.expert_table");
    REQUIRE(expert_table_transient.size == expert_table_bytes);
    REQUIRE(partition_table_transient.value.value == expert_table_transient.value.value + 1);
    REQUIRE(partition_table_transient.name == "qwen.router.partition_table");
    REQUIRE(partition_table_transient.size == partition_table_bytes);

    const ggml::hrx::Dispatch & dispatch    = scheduler.plan().dispatches[0];
    const std::string           kernel_name = kernel_name_for_id(dispatch.kernel.kernel_id);
    REQUIRE(kernel_name == "qwen3_moe:qwen3_moe_router_top8_f32");
    REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == token_count);
    REQUIRE(dispatch.kernel.integer_parameters.at("route_id_stride") == route_ids->nb[1] / sizeof(int32_t));
    REQUIRE(dispatch.bindings.size() == 3);
    REQUIRE(dispatch.bindings[1].value == route_ids_value->id);
    REQUIRE(dispatch.bindings[1].length == route_id_length);
    REQUIRE(dispatch.bindings[2].value == output_value->id);
    require_compile_parameter(dispatch, "qwen3_moe.router.expert_count", "128");
    require_compile_parameter(dispatch, "qwen3_moe.router.route_count", "8");
    require_compile_parameter(dispatch, "qwen3_moe.workload.token_capacity", std::to_string(token_count));

    const ggml::hrx::Dispatch & expert_table_dispatch = scheduler.plan().dispatches[1];
    REQUIRE(kernel_name_for_id(expert_table_dispatch.kernel.kernel_id) == "qwen3_moe:qwen3_moe_build_expert_table");
    REQUIRE(expert_table_dispatch.kernel.integer_parameters.at("token_count") == token_count);
    REQUIRE(expert_table_dispatch.kernel.integer_parameters.at("route_count") == kQwenRouterRouteCount);
    REQUIRE(expert_table_dispatch.kernel.integer_parameters.at("route_stride") == route_ids->nb[1] / sizeof(int32_t));
    REQUIRE(expert_table_dispatch.kernel.integer_parameters.at("expert_count") == kQwenRouterExpertCount);
    REQUIRE(expert_table_dispatch.bindings.size() == 2);
    REQUIRE(expert_table_dispatch.bindings[0].value == route_ids_value->id);
    REQUIRE(expert_table_dispatch.bindings[0].length == route_id_length);
    REQUIRE(expert_table_dispatch.bindings[1].value == expert_table_transient.value);
    REQUIRE(expert_table_dispatch.bindings[1].length == expert_table_bytes);
    require_compile_parameter(expert_table_dispatch, "qwen3_moe.routed_gate_up.expert_count", "128");
    require_compile_parameter(expert_table_dispatch, "qwen3_moe.routed_gate_up.route_count", "8");
    require_compile_parameter(expert_table_dispatch, "qwen3_moe.workload.token_capacity", std::to_string(token_count));

    const ggml::hrx::Dispatch & partition_table_dispatch = scheduler.plan().dispatches[2];
    REQUIRE(kernel_name_for_id(partition_table_dispatch.kernel.kernel_id) ==
            "qwen3_moe:qwen3_moe_build_expert_partition_table");
    REQUIRE(partition_table_dispatch.kernel.integer_parameters.at("token_count") == token_count);
    REQUIRE(partition_table_dispatch.kernel.integer_parameters.at("route_count") == kQwenRouterRouteCount);
    REQUIRE(partition_table_dispatch.kernel.integer_parameters.at("expert_count") == kQwenRouterExpertCount);
    REQUIRE(partition_table_dispatch.bindings.size() == 2);
    REQUIRE(partition_table_dispatch.bindings[0].value == expert_table_transient.value);
    REQUIRE(partition_table_dispatch.bindings[0].length == expert_table_bytes);
    REQUIRE(partition_table_dispatch.bindings[1].value == partition_table_transient.value);
    REQUIRE(partition_table_dispatch.bindings[1].length == partition_table_bytes);
    require_compile_parameter(partition_table_dispatch, "qwen3_moe.routed_gate_up.expert_count", "128");
    require_compile_parameter(partition_table_dispatch, "qwen3_moe.routed_gate_up.route_count", "8");
    require_compile_parameter(partition_table_dispatch, "qwen3_moe.workload.token_capacity",
                              std::to_string(token_count));

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 3);
    REQUIRE(command_program_verifies(commands));
    REQUIRE(commands.commands[0].bindings.size() == 3);
    REQUIRE(commands.commands[0].bindings[0].name == "logits");
    REQUIRE(commands.commands[0].bindings[1].name == "route_ids");
    REQUIRE(commands.commands[0].bindings[1].origin == ggml::hrx::CommandBindingOrigin::Transient);
    REQUIRE(commands.commands[0].bindings[1].length == dispatch.bindings[1].length);
    REQUIRE(commands.commands[0].bindings[2].name == "route_weights");
    REQUIRE(commands.commands[1].dependencies.size() == 1);
    REQUIRE(commands.commands[1].dependencies[0] == 0);
    REQUIRE(commands.commands[1].bindings.size() == 2);
    REQUIRE(commands.commands[1].bindings[0].name == "route_ids");
    REQUIRE(commands.commands[1].bindings[0].origin == ggml::hrx::CommandBindingOrigin::Transient);
    REQUIRE(commands.commands[1].bindings[1].name == "expert_table");
    REQUIRE(commands.commands[1].bindings[1].origin == ggml::hrx::CommandBindingOrigin::Transient);
    REQUIRE(commands.commands[1].bindings[1].length == expert_table_bytes);
    REQUIRE(commands.commands[2].dependencies.size() == 1);
    REQUIRE(commands.commands[2].dependencies[0] == 1);
    REQUIRE(commands.commands[2].bindings.size() == 2);
    REQUIRE(commands.commands[2].bindings[0].name == "expert_table");
    REQUIRE(commands.commands[2].bindings[0].origin == ggml::hrx::CommandBindingOrigin::Transient);
    REQUIRE(commands.commands[2].bindings[1].name == "partition_table");
    REQUIRE(commands.commands[2].bindings[1].origin == ggml::hrx::CommandBindingOrigin::Transient);
    REQUIRE(commands.commands[2].bindings[1].length == partition_table_bytes);
    REQUIRE(commands.transients.allocations.size() == 3);
    const ggml::hrx::TransientAllocation * route_ids_allocation =
        ggml::hrx::find_transient_allocation(commands.transients, route_ids_value->id);
    REQUIRE(route_ids_allocation != nullptr);
    REQUIRE(route_ids_allocation->size == dispatch.bindings[1].length);
    const ggml::hrx::TransientAllocation * expert_table_allocation =
        ggml::hrx::find_transient_allocation(commands.transients, expert_table_transient.value);
    REQUIRE(expert_table_allocation != nullptr);
    REQUIRE(expert_table_allocation->size == expert_table_bytes);
    const ggml::hrx::TransientAllocation * partition_table_allocation =
        ggml::hrx::find_transient_allocation(commands.transients, partition_table_transient.value);
    REQUIRE(partition_table_allocation != nullptr);
    REQUIRE(partition_table_allocation->size == partition_table_bytes);
}

struct QwenRoutedGateUpTensors {
    ggml_tensor *              route_ids     = nullptr;
    ggml_tensor *              route_weights = nullptr;
    ggml_tensor *              gate          = nullptr;
    ggml_tensor *              up            = nullptr;
    ggml_tensor *              glu           = nullptr;
    ggml_tensor *              output        = nullptr;
    ggml_tensor *              weighted      = nullptr;
    ggml_tensor *              residual      = nullptr;
    ggml_tensor *              next_rms      = nullptr;
    ggml_tensor *              next_output   = nullptr;
    std::vector<ggml_tensor *> route_views;
};

static QwenRoutedGateUpTensors build_qwen_routed_gate_up_graph(ggml_context * ctx,
                                                               int64_t        token_count,
                                                               ggml_glu_op    glu_op           = GGML_GLU_OP_SWIGLU,
                                                               ggml_type      up_weight_type   = GGML_TYPE_Q4_K,
                                                               bool           include_down     = true,
                                                               ggml_type      down_weight_type = GGML_TYPE_Q6_K) {
    QwenRoutedGateUpTensors tensors;
    ggml_tensor *           logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenRouterExpertCount, token_count);
    REQUIRE(logits != nullptr);
    tensors.route_weights = build_qwen_router_top8_graph(ctx, logits, &tensors.route_ids);
    REQUIRE(tensors.route_weights != nullptr);
    REQUIRE(tensors.route_ids != nullptr);

    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kQwenMoeHiddenSize, 1, token_count);
    ggml_tensor * gate_weight =
        ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, kQwenMoeHiddenSize, kQwenMoeIntermediateSize, kQwenRouterExpertCount);
    ggml_tensor * up_weight =
        ggml_new_tensor_3d(ctx, up_weight_type, kQwenMoeHiddenSize, kQwenMoeIntermediateSize, kQwenRouterExpertCount);
    REQUIRE(input != nullptr);
    REQUIRE(gate_weight != nullptr);
    REQUIRE(up_weight != nullptr);

    tensors.gate = ggml_mul_mat_id(ctx, gate_weight, input, tensors.route_ids);
    tensors.up   = ggml_mul_mat_id(ctx, up_weight, input, tensors.route_ids);
    REQUIRE(tensors.gate != nullptr);
    REQUIRE(tensors.up != nullptr);
    tensors.glu = ggml_glu_split(ctx, tensors.gate, tensors.up, glu_op);
    REQUIRE(tensors.glu != nullptr);

    if (include_down) {
        ggml_tensor * down_weight = ggml_new_tensor_3d(ctx, down_weight_type, kQwenMoeIntermediateSize,
                                                       kQwenMoeHiddenSize, kQwenRouterExpertCount);
        REQUIRE(down_weight != nullptr);
        tensors.output = ggml_mul_mat_id(ctx, down_weight, tensors.glu, tensors.route_ids);
        REQUIRE(tensors.output != nullptr);
    } else {
        tensors.output = tensors.glu;
    }
    return tensors;
}

static void append_qwen_weighted_reduce_tail(ggml_context *            ctx,
                                             QwenRoutedGateUpTensors & tensors,
                                             bool                      include_next_rmsnorm = false) {
    REQUIRE(tensors.output != nullptr);
    REQUIRE(tensors.route_weights != nullptr);
    tensors.weighted = ggml_mul(ctx, tensors.output, tensors.route_weights);
    REQUIRE(tensors.weighted != nullptr);
    tensors.route_views.clear();
    for (int64_t route = 0; route < kQwenRouterRouteCount; ++route) {
        ggml_tensor * view =
            ggml_view_2d(ctx, tensors.weighted, kQwenMoeHiddenSize, tensors.weighted->ne[2], tensors.weighted->nb[2],
                         static_cast<size_t>(route) * tensors.weighted->nb[1]);
        REQUIRE(view != nullptr);
        tensors.route_views.push_back(view);
    }

    ggml_tensor * reduced = tensors.route_views.front();
    for (size_t i = 1; i < tensors.route_views.size(); ++i) {
        reduced = ggml_add(ctx, reduced, tensors.route_views[i]);
        REQUIRE(reduced != nullptr);
    }

    ggml_tensor * hidden_state = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenMoeHiddenSize, tensors.output->ne[2]);
    REQUIRE(hidden_state != nullptr);
    tensors.residual = ggml_add(ctx, hidden_state, reduced);
    REQUIRE(tensors.residual != nullptr);

    if (include_next_rmsnorm) {
        ggml_tensor * next_norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kQwenMoeHiddenSize);
        REQUIRE(next_norm_weight != nullptr);
        tensors.next_rms = ggml_rms_norm(ctx, tensors.residual, 0.000001f);
        REQUIRE(tensors.next_rms != nullptr);
        tensors.next_output = ggml_mul(ctx, tensors.next_rms, next_norm_weight);
        REQUIRE(tensors.next_output != nullptr);
    }
}

static ggml::hrx::GraphImportResult import_qwen_routed_gate_up_graph(ggml_context *                  ctx,
                                                                     const QwenRoutedGateUpTensors & tensors) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, tensors.route_weights);
    ggml_build_forward_expand(graph, tensors.output);
    if (tensors.residual != nullptr) {
        ggml_build_forward_expand(graph, tensors.residual);
    }
    if (tensors.next_output != nullptr) {
        ggml_build_forward_expand(graph, tensors.next_output);
    }

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    return imported;
}

static bool match_dispatch_at_index(const ggml::hrx::Graph &       graph,
                                    const ggml::hrx::CommandPlan & plan,
                                    const std::vector<bool> &      covered_nodes,
                                    size_t                         node_index,
                                    ggml::hrx::DispatchMatch &     match) {
    REQUIRE(node_index < graph.nodes().size());
    const ggml::hrx::DispatchMatchContext context = {
        graph,      &graph.nodes()[node_index],
        node_index, covered_nodes,
        plan,       ggml::hrx::ValueId(static_cast<int32_t>(graph.values().size() + plan.transients.size())),
    };
    return test_dispatch_registry().match(context, match);
}

static void append_match_to_plan(ggml::hrx::CommandPlan &   plan,
                                 ggml::hrx::DispatchMatch & match,
                                 std::vector<bool> &        covered_nodes) {
    for (ggml::hrx::Dispatch & dispatch : match.dispatches) {
        plan.dispatches.push_back(std::move(dispatch));
    }
    for (ggml::hrx::CommandPlanTransient & transient : match.transients) {
        plan.transients.push_back(std::move(transient));
    }
    REQUIRE(plan.metadata.append(std::move(match.metadata), plan.status));
    for (const size_t covered_node : match.covered_nodes) {
        REQUIRE(covered_node < covered_nodes.size());
        REQUIRE(!covered_nodes[covered_node]);
        covered_nodes[covered_node] = true;
    }
}

static ggml::hrx::CommandPlan build_qwen_router_plan_for_graph(const ggml::hrx::Graph & graph,
                                                               std::vector<bool> &      covered_nodes) {
    ggml::hrx::CommandPlan plan;
    size_t                 softmax_index = graph.nodes().size();
    for (size_t i = 0; i < graph.nodes().size(); ++i) {
        if (graph.nodes()[i].op == GGML_OP_SOFT_MAX) {
            softmax_index = i;
            break;
        }
    }
    REQUIRE(softmax_index < graph.nodes().size());
    ggml::hrx::DispatchMatch router_match;
    REQUIRE(match_dispatch_at_index(graph, plan, covered_nodes, softmax_index, router_match));
    append_match_to_plan(plan, router_match, covered_nodes);
    return plan;
}

static void append_qwen_routed_gate_up_for_graph(const ggml::hrx::Graph &        graph,
                                                 const QwenRoutedGateUpTensors & tensors,
                                                 std::vector<bool> &             covered_nodes,
                                                 ggml::hrx::CommandPlan &        plan) {
    const size_t             gate_index = producer_index_for_tensor(graph, tensors.gate);
    ggml::hrx::DispatchMatch gate_up_match;
    REQUIRE(match_dispatch_at_index(graph, plan, covered_nodes, gate_index, gate_up_match));
    append_match_to_plan(plan, gate_up_match, covered_nodes);
}

static void append_qwen_routed_down_for_graph(const ggml::hrx::Graph &        graph,
                                              const QwenRoutedGateUpTensors & tensors,
                                              std::vector<bool> &             covered_nodes,
                                              ggml::hrx::CommandPlan &        plan,
                                              const char *                    expected_kernel_name) {
    const size_t             down_index = producer_index_for_tensor(graph, tensors.output);
    ggml::hrx::DispatchMatch down_match;
    REQUIRE(match_dispatch_at_index(graph, plan, covered_nodes, down_index, down_match));
    append_match_to_plan(plan, down_match, covered_nodes);

    REQUIRE(plan.dispatches.size() >= 1);
    REQUIRE(plan.transients.size() >= 1);
    const ggml::hrx::Value * route_ids_value = graph.values().find_tensor(tensors.route_ids);
    const ggml::hrx::Value * glu_value       = graph.values().find_tensor(tensors.glu);
    const ggml::hrx::Value * output_value    = graph.values().find_tensor(tensors.output);
    REQUIRE(route_ids_value != nullptr);
    REQUIRE(glu_value != nullptr);
    REQUIRE(output_value != nullptr);
    const ggml::hrx::CommandPlanQwenRoutingBundle * routing_bundle =
        plan.metadata.find_qwen_routing_bundle(route_ids_value->id);
    const ggml::hrx::CommandPlanAlternateValue * gate_up_alternate = plan.metadata.find_alternate_value(
        glu_value->id, GGML_TYPE_F16, qwen_routed_gate_up_f16_output_size(tensors.output->ne[2]));
    const ggml::hrx::CommandPlanAlternateValue * routed_down_alternate = plan.metadata.find_alternate_value(
        output_value->id, GGML_TYPE_F16, qwen_routed_down_f16_output_size(tensors.output->ne[2]));
    REQUIRE(routing_bundle != nullptr);
    REQUIRE(gate_up_alternate != nullptr);
    REQUIRE(routed_down_alternate != nullptr);
    const ggml::hrx::Dispatch &             dispatch              = plan.dispatches.back();
    const ggml::hrx::CommandPlanTransient & routed_down_transient = plan.transients.back();
    REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == expected_kernel_name);
    REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == tensors.output->ne[2]);
    REQUIRE(dispatch.bindings.size() == 4);
    REQUIRE(routed_down_transient.name == "qwen.moe.routed_down_f16");
    REQUIRE(routed_down_transient.size == qwen_routed_down_f16_output_size(tensors.output->ne[2]));
    REQUIRE(dispatch.bindings[0].value == gate_up_alternate->alternate_value);
    REQUIRE(dispatch.bindings[0].length == qwen_routed_gate_up_f16_output_size(tensors.output->ne[2]));
    REQUIRE(dispatch.bindings[1].value == routing_bundle->expert_table);
    REQUIRE(dispatch.bindings[1].length == routing_bundle->expert_table_byte_count);
    REQUIRE(routed_down_alternate->alternate_value == routed_down_transient.value);
    REQUIRE(routed_down_alternate->byte_count == routed_down_transient.size);
    REQUIRE(dispatch.bindings[3].value == routed_down_transient.value);
    REQUIRE(dispatch.bindings[3].length == routed_down_transient.size);
    require_compile_parameter(dispatch, "qwen3_moe.routed_down.input_size", "768");
    require_compile_parameter(dispatch, "qwen3_moe.routed_down.route_count", "8");
    require_compile_parameter(dispatch, "qwen3_moe.routed_down.expert_count", "128");
    require_compile_parameter(dispatch, "qwen3_moe.routed_down.output_size", "2048");
    require_compile_parameter(dispatch, "qwen3_moe.workload.token_capacity", std::to_string(tensors.output->ne[2]));
}

static void append_qwen_weighted_reduce_for_graph(const ggml::hrx::Graph &        graph,
                                                  const QwenRoutedGateUpTensors & tensors,
                                                  std::vector<bool> &             covered_nodes,
                                                  ggml::hrx::CommandPlan &        plan,
                                                  const char *                    expected_kernel_name) {
    REQUIRE(tensors.weighted != nullptr);
    REQUIRE(tensors.residual != nullptr);
    const size_t             weighted_index = producer_index_for_tensor(graph, tensors.weighted);
    ggml::hrx::DispatchMatch weighted_match;
    REQUIRE(match_dispatch_at_index(graph, plan, covered_nodes, weighted_index, weighted_match));
    append_match_to_plan(plan, weighted_match, covered_nodes);

    const ggml::hrx::Value * route_weights_value = graph.values().find_tensor(tensors.route_weights);
    const ggml::hrx::Value * routed_output_value = graph.values().find_tensor(tensors.output);
    const ggml::hrx::Value * residual_value      = graph.values().find_tensor(tensors.residual);
    REQUIRE(route_weights_value != nullptr);
    REQUIRE(routed_output_value != nullptr);
    REQUIRE(residual_value != nullptr);

    const ggml::hrx::Dispatch & dispatch = plan.dispatches.back();
    REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == expected_kernel_name);
    REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == tensors.output->ne[2]);
    REQUIRE(dispatch.bindings.size() == (tensors.next_output != nullptr ? 5 : 3));
    REQUIRE(dispatch.bindings[0].value == route_weights_value->id);
    REQUIRE(dispatch.bindings[0].length == route_weights_value->byte_count);
    REQUIRE(dispatch.bindings[1].value == plan.metadata.alternate_values().back().alternate_value);
    REQUIRE(dispatch.bindings[1].length == qwen_routed_down_f16_output_size(tensors.output->ne[2]));
    REQUIRE(dispatch.bindings[2].value == residual_value->id);
    REQUIRE(dispatch.bindings[2].length == residual_value->byte_count);
    require_compile_parameter(dispatch, "qwen3_moe.routed_down.route_count", "8");
    require_compile_parameter(dispatch, "qwen3_moe.routed_down.output_size", "2048");
    require_compile_parameter(dispatch, "qwen3_moe.workload.token_capacity", std::to_string(tensors.output->ne[2]));

    if (tensors.next_output != nullptr) {
        const ggml::hrx::Value * next_output_value = graph.values().find_tensor(tensors.next_output);
        REQUIRE(next_output_value != nullptr);
        REQUIRE(dispatch.bindings[4].value == next_output_value->id);
        REQUIRE(dispatch.bindings[4].length == next_output_value->byte_count);
        require_compile_parameter(dispatch, "qwen3_moe.model.hidden_size", "2048");
        require_compile_parameter(dispatch, "qwen3_moe.model.rms_epsilon", "0.000001");
    }
}

static void run_qwen_routed_gate_up_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size         = 4 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    {
        constexpr int64_t             token_count = 4;
        const QwenRoutedGateUpTensors tensors     = build_qwen_routed_gate_up_graph(ctx, token_count);
        ggml::hrx::GraphImportResult  imported    = import_qwen_routed_gate_up_graph(ctx, tensors);
        REQUIRE(imported.graph.nodes().size() == 14);

        const ggml::hrx::Value * route_ids_value     = imported.graph.values().find_tensor(tensors.route_ids);
        const ggml::hrx::Value * route_weights_value = imported.graph.values().find_tensor(tensors.route_weights);
        const ggml::hrx::Value * glu_value           = imported.graph.values().find_tensor(tensors.glu);
        REQUIRE(route_ids_value != nullptr);
        REQUIRE(route_weights_value != nullptr);
        REQUIRE(glu_value != nullptr);
        REQUIRE(route_ids_value->kind == ggml::hrx::ValueKind::Transient);
        REQUIRE(route_weights_value->kind == ggml::hrx::ValueKind::External);
        REQUIRE(glu_value->kind == ggml::hrx::ValueKind::Transient);

        std::vector<bool>      covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan plan = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        const ggml::hrx::CommandPlanGeneratedResource * expert_table_resource = plan.metadata.find_generated_resource(
            route_ids_value->id, ggml::hrx::GeneratedResourceRole::QwenMoeExpertTable);
        const ggml::hrx::CommandPlanGeneratedResource * partition_table_resource =
            plan.metadata.find_generated_resource(route_ids_value->id,
                                                  ggml::hrx::GeneratedResourceRole::QwenMoePartitionTable);
        const ggml::hrx::CommandPlanQwenRoutingBundle * routing_bundle =
            plan.metadata.find_qwen_routing_bundle(route_ids_value->id);
        REQUIRE(expert_table_resource != nullptr);
        REQUIRE(partition_table_resource != nullptr);
        REQUIRE(routing_bundle != nullptr);
        REQUIRE(routing_bundle->route_ids == route_ids_value->id);
        REQUIRE(routing_bundle->route_weights == route_weights_value->id);
        REQUIRE(routing_bundle->expert_table == expert_table_resource->generated_value);
        REQUIRE(routing_bundle->partition_table == partition_table_resource->generated_value);
        REQUIRE(routing_bundle->expert_table_byte_count == qwen_expert_table_size(token_count));
        REQUIRE(routing_bundle->partition_table_byte_count == qwen_partition_table_size(token_count));
        REQUIRE(routing_bundle->route_count == kQwenRouterRouteCount);
        REQUIRE(routing_bundle->expert_count == kQwenRouterExpertCount);
        ggml::hrx::QwenMoeRoutingResourceMetadata expert_metadata;
        ggml::hrx::QwenMoeRoutingResourceMetadata partition_metadata;
        REQUIRE(expert_table_resource->metadata.read(expert_metadata));
        REQUIRE(partition_table_resource->metadata.read(partition_metadata));
        REQUIRE(expert_metadata.token_count == token_count);
        REQUIRE(expert_metadata.route_count == kQwenRouterRouteCount);
        REQUIRE(expert_metadata.expert_count == kQwenRouterExpertCount);
        REQUIRE(partition_metadata.route_stride == expert_metadata.route_stride);
        REQUIRE(expert_table_resource->byte_count == qwen_expert_table_size(token_count));
        REQUIRE(partition_table_resource->byte_count == qwen_partition_table_size(token_count));

        const size_t             gate_index = producer_index_for_tensor(imported.graph, tensors.gate);
        ggml::hrx::DispatchMatch gate_up_match;
        REQUIRE(match_dispatch_at_index(imported.graph, plan, covered_nodes, gate_index, gate_up_match));
        append_match_to_plan(plan, gate_up_match, covered_nodes);

        REQUIRE(plan.dispatches.size() == 4);
        REQUIRE(plan.transients.size() == 3);
        REQUIRE(plan.metadata.alternate_values().size() == 1);
        const ggml::hrx::CommandPlanTransient & f16_output_transient = plan.transients.back();
        REQUIRE(f16_output_transient.name == "qwen.moe.gate_up_swiglu_f16");
        REQUIRE(f16_output_transient.size == qwen_routed_gate_up_f16_output_size(token_count));
        REQUIRE(plan.metadata.alternate_values().front().graph_value == glu_value->id);
        REQUIRE(plan.metadata.alternate_values().front().alternate_value == f16_output_transient.value);
        REQUIRE(plan.metadata.alternate_values().front().type == GGML_TYPE_F16);
        REQUIRE(plan.metadata.alternate_values().front().byte_count == f16_output_transient.size);

        const ggml::hrx::Dispatch & dispatch = plan.dispatches.back();
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) ==
                "qwen3_moe:qwen3_moe_routed_gate_up_swiglu_q4k_f16_wmma");
        REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == token_count);
        REQUIRE(dispatch.bindings.size() == 6);
        REQUIRE(dispatch.bindings[1].value == routing_bundle->expert_table);
        REQUIRE(dispatch.bindings[1].length == qwen_expert_table_size(token_count));
        REQUIRE(dispatch.bindings[2].value == routing_bundle->partition_table);
        REQUIRE(dispatch.bindings[2].length == qwen_partition_table_size(token_count));
        REQUIRE(dispatch.bindings[5].value == f16_output_transient.value);
        REQUIRE(dispatch.bindings[5].length == f16_output_transient.size);
        require_compile_parameter(dispatch, "qwen3_moe.routed_gate_up.input_size", "2048");
        require_compile_parameter(dispatch, "qwen3_moe.routed_gate_up.expert_count", "128");
        require_compile_parameter(dispatch, "qwen3_moe.routed_gate_up.route_count", "8");
        require_compile_parameter(dispatch, "qwen3_moe.routed_gate_up.output_size", "768");
        require_compile_parameter(dispatch, "qwen3_moe.workload.token_capacity", std::to_string(token_count));

        const ggml::hrx::CommandProgram commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(commands.commands.size() == 4);
        REQUIRE(command_program_verifies(commands));
        REQUIRE(commands.commands[3].bindings.size() == 6);
        REQUIRE(commands.commands[3].bindings[0].name == "input");
        REQUIRE(commands.commands[3].bindings[0].origin == ggml::hrx::CommandBindingOrigin::GraphValue);
        REQUIRE(commands.commands[3].bindings[1].name == "expert_table");
        REQUIRE(commands.commands[3].bindings[1].origin == ggml::hrx::CommandBindingOrigin::Transient);
        REQUIRE(commands.commands[3].bindings[2].name == "partition_table");
        REQUIRE(commands.commands[3].bindings[2].origin == ggml::hrx::CommandBindingOrigin::Transient);
        REQUIRE(commands.commands[3].bindings[3].name == "gate_weight");
        REQUIRE(commands.commands[3].bindings[3].origin == ggml::hrx::CommandBindingOrigin::GraphValue);
        REQUIRE(commands.commands[3].bindings[4].name == "up_weight");
        REQUIRE(commands.commands[3].bindings[4].origin == ggml::hrx::CommandBindingOrigin::GraphValue);
        REQUIRE(commands.commands[3].bindings[5].name == "output");
        REQUIRE(commands.commands[3].bindings[5].origin == ggml::hrx::CommandBindingOrigin::Transient);
        REQUIRE(ggml::hrx::find_transient_allocation(commands.transients, f16_output_transient.value) != nullptr);
    }

    {
        constexpr int64_t token_count     = 4;
        ggml_tensor *     first_logits    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenRouterExpertCount, token_count);
        ggml_tensor *     first_route_ids = nullptr;
        REQUIRE(first_logits != nullptr);
        ggml_tensor * first_route_weights = build_qwen_router_top8_graph(ctx, first_logits, &first_route_ids);
        REQUIRE(first_route_ids != nullptr);
        REQUIRE(first_route_weights != nullptr);

        const QwenRoutedGateUpTensors tensors = build_qwen_routed_gate_up_graph(ctx, token_count);
        ggml_cgraph *                 graph   = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, first_route_weights);
        ggml_build_forward_expand(graph, tensors.route_weights);
        ggml_build_forward_expand(graph, tensors.output);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());

        const ggml::hrx::Value * first_route_ids_value  = imported.graph.values().find_tensor(first_route_ids);
        const ggml::hrx::Value * second_route_ids_value = imported.graph.values().find_tensor(tensors.route_ids);
        REQUIRE(first_route_ids_value != nullptr);
        REQUIRE(second_route_ids_value != nullptr);
        REQUIRE(first_route_ids_value->id != second_route_ids_value->id);

        std::vector<bool>      covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan plan;
        size_t                 router_matches = 0;
        for (size_t i = 0; i < imported.graph.nodes().size(); ++i) {
            if (imported.graph.nodes()[i].op != GGML_OP_SOFT_MAX) {
                continue;
            }
            ggml::hrx::DispatchMatch router_match;
            REQUIRE(match_dispatch_at_index(imported.graph, plan, covered_nodes, i, router_match));
            append_match_to_plan(plan, router_match, covered_nodes);
            ++router_matches;
        }
        REQUIRE(router_matches == 2);
        REQUIRE(plan.metadata.generated_resources().size() == 4);
        REQUIRE(plan.metadata.qwen_routing_bundles().size() == 2);

        const ggml::hrx::CommandPlanGeneratedResource * first_expert_table = plan.metadata.find_generated_resource(
            first_route_ids_value->id, ggml::hrx::GeneratedResourceRole::QwenMoeExpertTable);
        const ggml::hrx::CommandPlanGeneratedResource * second_expert_table = plan.metadata.find_generated_resource(
            second_route_ids_value->id, ggml::hrx::GeneratedResourceRole::QwenMoeExpertTable);
        const ggml::hrx::CommandPlanGeneratedResource * second_partition_table = plan.metadata.find_generated_resource(
            second_route_ids_value->id, ggml::hrx::GeneratedResourceRole::QwenMoePartitionTable);
        const ggml::hrx::CommandPlanQwenRoutingBundle * second_routing_bundle =
            plan.metadata.find_qwen_routing_bundle(second_route_ids_value->id);
        REQUIRE(first_expert_table != nullptr);
        REQUIRE(second_expert_table != nullptr);
        REQUIRE(second_partition_table != nullptr);
        REQUIRE(second_routing_bundle != nullptr);
        REQUIRE(second_routing_bundle->expert_table == second_expert_table->generated_value);
        REQUIRE(second_routing_bundle->partition_table == second_partition_table->generated_value);
        REQUIRE(first_expert_table->generated_value != second_expert_table->generated_value);

        const size_t             gate_index = producer_index_for_tensor(imported.graph, tensors.gate);
        ggml::hrx::DispatchMatch gate_up_match;
        REQUIRE(match_dispatch_at_index(imported.graph, plan, covered_nodes, gate_index, gate_up_match));
        append_match_to_plan(plan, gate_up_match, covered_nodes);

        REQUIRE(plan.dispatches.size() == 7);
        REQUIRE(plan.transients.size() == 5);
        const ggml::hrx::Dispatch & dispatch = plan.dispatches.back();
        REQUIRE(dispatch.bindings.size() == 6);
        REQUIRE(dispatch.bindings[1].value == second_routing_bundle->expert_table);
        REQUIRE(dispatch.bindings[1].value != first_expert_table->generated_value);
        REQUIRE(dispatch.bindings[2].value == second_routing_bundle->partition_table);

        const ggml::hrx::CommandProgram commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(commands.commands.size() == 7);
        REQUIRE(command_program_verifies(commands));
    }

    {
        constexpr int64_t             token_count = 4;
        const QwenRoutedGateUpTensors tensors     = build_qwen_routed_gate_up_graph(ctx, token_count);
        ggml::hrx::GraphImportResult  imported    = import_qwen_routed_gate_up_graph(ctx, tensors);
        std::vector<bool>             covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan        plan = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        append_qwen_routed_gate_up_for_graph(imported.graph, tensors, covered_nodes, plan);
        append_qwen_routed_down_for_graph(imported.graph, tensors, covered_nodes, plan,
                                          "qwen3_moe:qwen3_moe_routed_down_q6k_f16_wmma_grouped");

        REQUIRE(plan.dispatches.size() == 5);
        REQUIRE(plan.transients.size() == 4);
        const ggml::hrx::CommandProgram commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(commands.commands.size() == 5);
        REQUIRE(command_program_verifies(commands));
        REQUIRE(ggml::hrx::find_transient_allocation(commands.transients, plan.transients.back().value) != nullptr);
    }

    {
        constexpr int64_t       token_count = 4;
        QwenRoutedGateUpTensors tensors     = build_qwen_routed_gate_up_graph(ctx, token_count);
        append_qwen_weighted_reduce_tail(ctx, tensors);
        ggml::hrx::GraphImportResult imported = import_qwen_routed_gate_up_graph(ctx, tensors);
        std::vector<bool>            covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan       plan = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        append_qwen_routed_gate_up_for_graph(imported.graph, tensors, covered_nodes, plan);
        append_qwen_routed_down_for_graph(imported.graph, tensors, covered_nodes, plan,
                                          "qwen3_moe:qwen3_moe_routed_down_q6k_f16_wmma_grouped");
        append_qwen_weighted_reduce_for_graph(imported.graph, tensors, covered_nodes, plan,
                                              "qwen3_moe:qwen3_moe_routed_down_weighted_reduce_f16_f32");

        REQUIRE(plan.dispatches.size() == 6);
        REQUIRE(plan.transients.size() == 4);
        const ggml::hrx::CommandProgram commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(commands.commands.size() == 6);
        REQUIRE(command_program_verifies(commands));
        REQUIRE(commands.commands.back().bindings.size() == 3);
        REQUIRE(commands.commands.back().bindings[0].name == "route_weights");
        REQUIRE(commands.commands.back().bindings[1].name == "routed_output");
        REQUIRE(commands.commands.back().bindings[1].origin == ggml::hrx::CommandBindingOrigin::Transient);
        REQUIRE(commands.commands.back().bindings[2].name == "output");
    }

    {
        constexpr int64_t       token_count = 4;
        QwenRoutedGateUpTensors tensors     = build_qwen_routed_gate_up_graph(ctx, token_count);
        append_qwen_weighted_reduce_tail(ctx, tensors, true);
        ggml::hrx::GraphImportResult imported = import_qwen_routed_gate_up_graph(ctx, tensors);
        std::vector<bool>            covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan       plan = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        append_qwen_routed_gate_up_for_graph(imported.graph, tensors, covered_nodes, plan);
        append_qwen_routed_down_for_graph(imported.graph, tensors, covered_nodes, plan,
                                          "qwen3_moe:qwen3_moe_routed_down_q6k_f16_wmma_grouped");
        append_qwen_weighted_reduce_for_graph(imported.graph, tensors, covered_nodes, plan,
                                              "qwen3_moe:qwen3_moe_routed_down_weighted_reduce_next_rmsnorm_f32");

        REQUIRE(plan.dispatches.size() == 6);
        REQUIRE(plan.transients.size() == 4);
        const ggml::hrx::CommandProgram commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(commands.commands.size() == 6);
        REQUIRE(command_program_verifies(commands));
        REQUIRE(commands.commands.back().bindings.size() == 5);
        REQUIRE(commands.commands.back().bindings[0].name == "route_weights");
        REQUIRE(commands.commands.back().bindings[1].name == "routed_output");
        REQUIRE(commands.commands.back().bindings[1].origin == ggml::hrx::CommandBindingOrigin::Transient);
        REQUIRE(commands.commands.back().bindings[2].name == "hidden_state");
        REQUIRE(commands.commands.back().bindings[3].name == "next_norm_weight");
        REQUIRE(commands.commands.back().bindings[4].name == "next_projection_input");
    }

    {
        constexpr int64_t             token_count = 4;
        const QwenRoutedGateUpTensors tensors =
            build_qwen_routed_gate_up_graph(ctx, token_count, GGML_GLU_OP_SWIGLU, GGML_TYPE_Q4_K, true, GGML_TYPE_Q4_K);
        ggml::hrx::GraphImportResult imported = import_qwen_routed_gate_up_graph(ctx, tensors);
        std::vector<bool>            covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan       plan = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        append_qwen_routed_gate_up_for_graph(imported.graph, tensors, covered_nodes, plan);
        append_qwen_routed_down_for_graph(imported.graph, tensors, covered_nodes, plan,
                                          "qwen3_moe:qwen3_moe_routed_down_q4k_f16_wmma_grouped");

        REQUIRE(plan.dispatches.size() == 5);
        REQUIRE(plan.transients.size() == 4);
        const ggml::hrx::CommandProgram commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(commands.commands.size() == 5);
        REQUIRE(command_program_verifies(commands));
    }

    {
        const QwenRoutedGateUpTensors tensors    = build_qwen_routed_gate_up_graph(ctx, 4);
        ggml::hrx::GraphImportResult  imported   = import_qwen_routed_gate_up_graph(ctx, tensors);
        const size_t                  gate_index = producer_index_for_tensor(imported.graph, tensors.gate);
        std::vector<bool>             covered_nodes(imported.graph.nodes().size(), false);
        const ggml::hrx::CommandPlan  empty_plan;
        ggml::hrx::DispatchMatch      gate_up_match;
        REQUIRE(!match_dispatch_at_index(imported.graph, empty_plan, covered_nodes, gate_index, gate_up_match));
    }

    {
        const QwenRoutedGateUpTensors tensors  = build_qwen_routed_gate_up_graph(ctx, 4, GGML_GLU_OP_GEGLU);
        ggml::hrx::GraphImportResult  imported = import_qwen_routed_gate_up_graph(ctx, tensors);
        std::vector<bool>             covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan        plan       = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        const size_t                  gate_index = producer_index_for_tensor(imported.graph, tensors.gate);
        ggml::hrx::DispatchMatch      gate_up_match;
        REQUIRE(!match_dispatch_at_index(imported.graph, plan, covered_nodes, gate_index, gate_up_match));
    }

    {
        const QwenRoutedGateUpTensors tensors =
            build_qwen_routed_gate_up_graph(ctx, 4, GGML_GLU_OP_SWIGLU, GGML_TYPE_Q6_K);
        ggml::hrx::GraphImportResult imported = import_qwen_routed_gate_up_graph(ctx, tensors);
        std::vector<bool>            covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan       plan       = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        const size_t                 gate_index = producer_index_for_tensor(imported.graph, tensors.gate);
        ggml::hrx::DispatchMatch     gate_up_match;
        REQUIRE(!match_dispatch_at_index(imported.graph, plan, covered_nodes, gate_index, gate_up_match));
    }

    {
        const QwenRoutedGateUpTensors tensors =
            build_qwen_routed_gate_up_graph(ctx, 4, GGML_GLU_OP_SWIGLU, GGML_TYPE_Q4_K, false);
        ggml::hrx::GraphImportResult imported = import_qwen_routed_gate_up_graph(ctx, tensors);
        std::vector<bool>            covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan       plan       = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        const size_t                 gate_index = producer_index_for_tensor(imported.graph, tensors.gate);
        ggml::hrx::DispatchMatch     gate_up_match;
        REQUIRE(!match_dispatch_at_index(imported.graph, plan, covered_nodes, gate_index, gate_up_match));
    }

    {
        const QwenRoutedGateUpTensors tensors  = build_qwen_routed_gate_up_graph(ctx, 4);
        ggml::hrx::GraphImportResult  imported = import_qwen_routed_gate_up_graph(ctx, tensors);
        std::vector<bool>             covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan        plan       = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        const size_t                  down_index = producer_index_for_tensor(imported.graph, tensors.output);
        ggml::hrx::DispatchMatch      down_match;
        REQUIRE(!match_dispatch_at_index(imported.graph, plan, covered_nodes, down_index, down_match));
    }

    {
        const QwenRoutedGateUpTensors tensors =
            build_qwen_routed_gate_up_graph(ctx, 4, GGML_GLU_OP_SWIGLU, GGML_TYPE_Q4_K, true, GGML_TYPE_Q5_K);
        ggml::hrx::GraphImportResult imported = import_qwen_routed_gate_up_graph(ctx, tensors);
        std::vector<bool>            covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan       plan = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        append_qwen_routed_gate_up_for_graph(imported.graph, tensors, covered_nodes, plan);
        const size_t             down_index = producer_index_for_tensor(imported.graph, tensors.output);
        ggml::hrx::DispatchMatch down_match;
        REQUIRE(!match_dispatch_at_index(imported.graph, plan, covered_nodes, down_index, down_match));
    }

    ggml_free(ctx);
}

static void run_qwen_router_top8_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size         = 2 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    {
        ggml_tensor * logits    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenRouterExpertCount, 4);
        ggml_tensor * route_ids = nullptr;
        REQUIRE(logits != nullptr);
        ggml_tensor * output = build_qwen_router_top8_graph(ctx, logits, &route_ids);
        schedule_qwen_router_top8_command(ctx, output, route_ids);
    }
    {
        ggml_tensor * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenRouterExpertCount, 4);
        REQUIRE(logits != nullptr);
        ggml_tensor * output = build_qwen_router_top8_graph(ctx, logits, nullptr, GGML_SORT_ORDER_ASC);
        REQUIRE(!graph_is_supported(ctx, output));
    }
    {
        ggml_tensor * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenRouterExpertCount, 4);
        REQUIRE(logits != nullptr);
        ggml_tensor * output = build_qwen_router_top8_graph(ctx, logits, nullptr, GGML_SORT_ORDER_DESC, 4);
        REQUIRE(!graph_is_supported(ctx, output));
    }
    {
        ggml_tensor * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 4);
        REQUIRE(logits != nullptr);
        ggml_tensor * output = build_qwen_router_top8_graph(ctx, logits);
        REQUIRE(!graph_is_supported(ctx, output));
    }
    {
        ggml_tensor * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kQwenRouterExpertCount, 4);
        REQUIRE(logits != nullptr);
        ggml_tensor * output = build_qwen_router_top8_graph(ctx, logits);
        REQUIRE(!graph_is_supported(ctx, output));
    }

    ggml_free(ctx);
}

static void bind_external_values(ggml::hrx::ValueMap & values) {
    uintptr_t buffer = 0x1000;
    for (const ggml::hrx::ValueId id : values.external_value_ids()) {
        const ggml::hrx::Value * value = values.find(id);
        REQUIRE(value != nullptr);
        REQUIRE(values.bind_buffer(id, { dummy_hrx_buffer(buffer), 0, value->byte_count }));
        buffer += 0x1000;
    }
}

static void run_multi_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * a    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * c    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * d    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * out0 = ggml_add(ctx, a, b);
    ggml_tensor * out1 = ggml_add(ctx, c, d);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(c != nullptr);
    REQUIRE(d != nullptr);
    REQUIRE(out0 != nullptr);
    REQUIRE(out1 != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out0);
    ggml_build_forward_expand(graph, out1);
    graph->uid = 1001;

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 2);
    const ggml::hrx::Value * out0_value = imported.graph.values().find_tensor(out0);
    const ggml::hrx::Value * out1_value = imported.graph.values().find_tensor(out1);
    REQUIRE(out0_value != nullptr);
    REQUIRE(out1_value != nullptr);
    REQUIRE(imported.graph.nodes()[0].output == out0_value->id);
    REQUIRE(imported.graph.nodes()[1].output == out1_value->id);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 2);

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 2);
    REQUIRE(commands.commands[0].ordinal == 0);
    REQUIRE(commands.commands[0].dependencies.empty());
    REQUIRE(commands.commands[1].ordinal == 1);
    REQUIRE(commands.commands[1].dependencies.size() == 1);
    REQUIRE(commands.commands[1].dependencies[0] == 0);
    REQUIRE(command_program_verifies(commands));

    bind_external_values(imported.graph.values());
    const ggml::hrx::CommandProgramBindings bindings =
        ggml::hrx::CommandProgramBindings::from_value_map(imported.graph.values());
    REQUIRE(bindings.valid());
    ggml::hrx::ResolvedCommandProgram resolved = ggml::hrx::resolve_command_program_bindings(commands, bindings);
    REQUIRE(resolved.valid());
    REQUIRE(resolved.commands.size() == 2);

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

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(!scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(!scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.empty());
    REQUIRE(status_contains(scheduler.plan().status, "unsupported HRX node 1"));

    ggml_free(ctx);
}

static void run_chained_dispatch_requires_transients() {
    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * a   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * b   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * c   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * sum = ggml_add(ctx, a, b);
    ggml_tensor * out = ggml_add(ctx, sum, c);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(c != nullptr);
    REQUIRE(sum != nullptr);
    REQUIRE(out != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 2);
    REQUIRE(imported.graph.nodes()[0].op == GGML_OP_ADD);
    REQUIRE(imported.graph.nodes()[1].op == GGML_OP_ADD);

    const ggml::hrx::Value * sum_value = imported.graph.values().find_tensor(sum);
    REQUIRE(sum_value != nullptr);
    REQUIRE(sum_value->kind == ggml::hrx::ValueKind::Transient);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 2);

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 2);
    REQUIRE(commands.commands[1].dependencies.size() == 1);
    REQUIRE(commands.commands[1].dependencies[0] == 0);
    REQUIRE(commands.commands[0].bindings.size() == 3);
    REQUIRE(commands.commands[1].bindings.size() == 3);
    REQUIRE(commands.commands[0].bindings[0].origin == ggml::hrx::CommandBindingOrigin::GraphValue);
    REQUIRE(commands.commands[0].bindings[1].origin == ggml::hrx::CommandBindingOrigin::GraphValue);
    REQUIRE(commands.commands[0].bindings[2].value == sum_value->id);
    REQUIRE(commands.commands[0].bindings[2].origin == ggml::hrx::CommandBindingOrigin::Transient);
    REQUIRE(commands.commands[1].bindings[0].value == sum_value->id);
    REQUIRE(commands.commands[1].bindings[0].origin == ggml::hrx::CommandBindingOrigin::Transient);
    REQUIRE(commands.commands[1].bindings[1].origin == ggml::hrx::CommandBindingOrigin::GraphValue);
    REQUIRE(commands.commands[1].bindings[2].origin == ggml::hrx::CommandBindingOrigin::GraphValue);
    REQUIRE(commands.transients.allocations.size() == 1);
    const ggml::hrx::TransientAllocation * sum_allocation =
        ggml::hrx::find_transient_allocation(commands.transients, sum_value->id);
    REQUIRE(sum_allocation != nullptr);
    REQUIRE(sum_allocation->value == sum_value->id);
    REQUIRE(sum_allocation->size == sum_value->byte_count);
    REQUIRE(sum_allocation->alignment == 256);
    REQUIRE(sum_allocation->arena_offset == 0);
    REQUIRE(commands.transients.arena_size == 256);
    REQUIRE(command_program_verifies(commands));

    const std::string transient_binding_text = ggml::hrx::format_command_binding(commands.commands[1].bindings[0]);
    REQUIRE(string_contains(transient_binding_text, "origin=Transient"));

    bind_external_values(imported.graph.values());
    const ggml::hrx::CommandProgramBindings bindings =
        ggml::hrx::CommandProgramBindings::from_value_map(imported.graph.values());
    REQUIRE(bindings.valid());
    REQUIRE(bindings.find(sum_value->id) == nullptr);

    const ggml::hrx::ResolvedCommandProgram resolved = ggml::hrx::resolve_command_program_bindings(commands, bindings);
    REQUIRE(!resolved.valid());
    REQUIRE(status_contains(resolved.status, "no transient arena"));
    REQUIRE(status_contains(resolved.status, "origin=Transient"));
    REQUIRE(status_contains(resolved.status, "value="));

    const ggml::hrx::TransientArenaAllocationRef transient_arena = {
        dummy_hrx_buffer(0x8000),
        commands.transients.arena_size,
        7,
    };
    const ggml::hrx::ResolvedCommandProgram resolved_with_transients =
        ggml::hrx::resolve_command_program_bindings(commands, bindings, &transient_arena);
    REQUIRE(resolved_with_transients.valid());
    REQUIRE(resolved_with_transients.commands.size() == 2);
    REQUIRE(resolved_with_transients.commands[0].bindings[2].ref.buffer == dummy_hrx_buffer(0x8000));
    REQUIRE(resolved_with_transients.commands[0].bindings[2].ref.offset == 0);
    REQUIRE(resolved_with_transients.commands[0].bindings[2].ref.length == sum_value->byte_count);
    REQUIRE(resolved_with_transients.commands[1].bindings[0].ref.buffer == dummy_hrx_buffer(0x8000));
    REQUIRE(resolved_with_transients.commands[1].bindings[0].ref.offset == 0);
    REQUIRE(resolved_with_transients.commands[1].bindings[0].ref.length == sum_value->byte_count);

    ggml::hrx::PreparedCommandProgram prepared_shape;
    for (const ggml::hrx::Command & prepared_source : commands.commands) {
        ggml::hrx::PreparedCommand prepared_command;
        prepared_command.ordinal               = prepared_source.ordinal;
        prepared_command.kind                  = prepared_source.kind;
        prepared_command.kernel.specialization = prepared_source.kernel;
        for (const ggml::hrx::CommandBinding & binding : prepared_source.bindings) {
            prepared_command.kernel.bindings.push_back({
                binding, { dummy_hrx_buffer(0x4000), 123, binding.length }
            });
        }
        prepared_shape.commands.push_back(prepared_command);
    }
    prepared_shape.bound_transient_arena_allocation_id = 1;

    REQUIRE(ggml::hrx::bind_prepared_command_program_transients(commands, transient_arena, prepared_shape));
    REQUIRE(prepared_shape.bound_transient_arena_allocation_id == transient_arena.allocation_id);
    REQUIRE(prepared_shape.commands[0].kernel.bindings[2].ref.buffer == dummy_hrx_buffer(0x8000));
    REQUIRE(prepared_shape.commands[0].kernel.bindings[2].ref.offset == 0);
    REQUIRE(prepared_shape.commands[1].kernel.bindings[0].ref.buffer == dummy_hrx_buffer(0x8000));
    REQUIRE(prepared_shape.commands[1].kernel.bindings[0].ref.offset == 0);

    const ggml::hrx::TransientArenaAllocationRef grown_transient_arena = {
        dummy_hrx_buffer(0x9000),
        commands.transients.arena_size + 256,
        8,
    };
    REQUIRE(ggml::hrx::bind_prepared_command_program_transients(commands, grown_transient_arena, prepared_shape));
    REQUIRE(prepared_shape.bound_transient_arena_allocation_id == grown_transient_arena.allocation_id);
    REQUIRE(prepared_shape.commands[0].kernel.bindings[2].ref.buffer == dummy_hrx_buffer(0x9000));
    REQUIRE(prepared_shape.commands[1].kernel.bindings[0].ref.buffer == dummy_hrx_buffer(0x9000));

    const ggml::hrx::TransientArenaAllocationRef invalid_transient_arena = {
        dummy_hrx_buffer(0xa000),
        commands.transients.arena_size,
        ggml::hrx::kInvalidTransientArenaAllocationId,
    };
    const ggml::hrx::ResolvedCommandProgram invalid_transient_resolved =
        ggml::hrx::resolve_command_program_bindings(commands, bindings, &invalid_transient_arena);
    REQUIRE(!invalid_transient_resolved.valid());
    REQUIRE(status_contains(invalid_transient_resolved.status, "no transient arena allocation id"));

    ggml::hrx::CommandProgram missing_allocation = copy_command_program_shape(commands);
    missing_allocation.transients.allocations.clear();
    ggml::hrx::VerificationResult verification =
        ggml::hrx::verify_command_program(missing_allocation, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(status_contains(verification.status, "no transient allocation"));

    ggml::hrx::CommandProgram out_of_range      = copy_command_program_shape(commands);
    out_of_range.commands[0].bindings[2].length = sum_value->byte_count + 1;
    verification = ggml::hrx::verify_command_program(out_of_range, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(status_contains(verification.status, "outside transient allocation length"));

    ggml_free(ctx);
}

static void run_multiple_transient_plan_checks() {
    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * a    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * c    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * d    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * sum0 = ggml_add(ctx, a, b);
    ggml_tensor * sum1 = ggml_add(ctx, c, d);
    ggml_tensor * out  = ggml_add(ctx, sum0, sum1);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(c != nullptr);
    REQUIRE(d != nullptr);
    REQUIRE(sum0 != nullptr);
    REQUIRE(sum1 != nullptr);
    REQUIRE(out != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 3);

    const ggml::hrx::Value * sum0_value = imported.graph.values().find_tensor(sum0);
    const ggml::hrx::Value * sum1_value = imported.graph.values().find_tensor(sum1);
    REQUIRE(sum0_value != nullptr);
    REQUIRE(sum1_value != nullptr);
    REQUIRE(sum0_value->kind == ggml::hrx::ValueKind::Transient);
    REQUIRE(sum1_value->kind == ggml::hrx::ValueKind::Transient);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 3);

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.transients.allocations.size() == 2);
    REQUIRE(commands.transients.arena_size == 512);
    const ggml::hrx::TransientAllocation * sum0_allocation =
        ggml::hrx::find_transient_allocation(commands.transients, sum0_value->id);
    const ggml::hrx::TransientAllocation * sum1_allocation =
        ggml::hrx::find_transient_allocation(commands.transients, sum1_value->id);
    REQUIRE(sum0_allocation != nullptr);
    REQUIRE(sum1_allocation != nullptr);
    REQUIRE(sum0_allocation->arena_offset != sum1_allocation->arena_offset);
    REQUIRE(sum0_allocation->arena_offset % 256 == 0);
    REQUIRE(sum1_allocation->arena_offset % 256 == 0);

    ggml_free(ctx);
}

static void run_graph_program_cache_uid_mismatch_checks() {
    ggml::hrx::GraphProgramCache    cache;
    const ggml::hrx::KernelCorpus & corpus = ggml::hrx::get_qwen_kernel_corpus();

    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * a    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * c    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * d    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * out0 = ggml_add(ctx, a, b);
    ggml_tensor * out1 = ggml_add(ctx, c, d);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(c != nullptr);
    REQUIRE(d != nullptr);
    REQUIRE(out0 != nullptr);
    REQUIRE(out1 != nullptr);

    ggml_cgraph * graph0 = ggml_new_graph(ctx);
    REQUIRE(graph0 != nullptr);
    ggml_build_forward_expand(graph0, out0);
    graph0->uid = 3001;

    ggml::hrx::GraphProgramLookup lookup = cache.get_or_build(*graph0, corpus, "gfx1151");
    REQUIRE(lookup.valid());
    ggml::hrx::GraphProgramCacheStats stats = cache.stats();
    REQUIRE(stats.builds == 1);
    REQUIRE(stats.hits == 0);

    ggml_cgraph * graph1 = ggml_new_graph(ctx);
    REQUIRE(graph1 != nullptr);
    ggml_build_forward_expand(graph1, out0);
    ggml_build_forward_expand(graph1, out1);
    graph1->uid = 3001;

    lookup = cache.get_or_build(*graph1, corpus, "gfx1151");
    REQUIRE(lookup.valid());
    stats = cache.stats();
    REQUIRE(stats.builds == 2);
    REQUIRE(stats.hits == 0);

    ggml_tensor * unsupported = ggml_sqr(ctx, a);
    REQUIRE(unsupported != nullptr);
    ggml_cgraph * graph2 = ggml_new_graph(ctx);
    REQUIRE(graph2 != nullptr);
    ggml_build_forward_expand(graph2, unsupported);
    graph2->uid = 3001;

    lookup = cache.get_or_build(*graph2, corpus, "gfx1151");
    REQUIRE(!lookup.valid());
    stats = cache.stats();
    REQUIRE(stats.builds == 2);
    REQUIRE(stats.hits == 0);

    ggml_free(ctx);
}

static void run_graph_executor_contract_checks() {
    ggml_backend_hrx_device_context device_context  = {};
    ggml_backend_hrx_context        backend_context = {};
    device_context.architecture                     = "gfx1151";
    backend_context.device                          = &device_context;
    const ggml::hrx::GraphExecutor executor(backend_context);

    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * a       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * b       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * add_out = ggml_add(ctx, a, b);
    ggml_tensor * sqr_out = ggml_sqr(ctx, a);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(add_out != nullptr);
    REQUIRE(sqr_out != nullptr);

    ggml_cgraph * add_graph = ggml_new_graph(ctx);
    REQUIRE(add_graph != nullptr);
    ggml_build_forward_expand(add_graph, add_out);
    const ggml::hrx::GraphSupportResult add_support = executor.can_execute(*add_graph);
    REQUIRE(add_support.supported);
    REQUIRE(add_support.status.success());

    const ggml::hrx::GraphExecutionResult missing_binding = executor.execute(*add_graph);
    REQUIRE(!missing_binding.success());
    REQUIRE(missing_binding.code == GGML_STATUS_FAILED);
    REQUIRE(status_contains(missing_binding.status, "external value"));
    REQUIRE(status_contains(missing_binding.status, "not bound"));

    ggml_cgraph * sqr_graph = ggml_new_graph(ctx);
    REQUIRE(sqr_graph != nullptr);
    ggml_build_forward_expand(sqr_graph, sqr_out);
    const ggml::hrx::GraphSupportResult sqr_support = executor.can_execute(*sqr_graph);
    REQUIRE(!sqr_support.supported);
    REQUIRE(status_contains(sqr_support.status, "unsupported HRX node 0"));
    REQUIRE(status_contains(sqr_support.status, "SQR"));

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

static void run_two_independent_add_f32() {
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
    ggml_tensor *     c             = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, element_count);
    ggml_tensor *     d             = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, element_count);
    ggml_tensor *     out0          = ggml_add(ctx, a, b);
    ggml_tensor *     out1          = ggml_add(ctx, c, d);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(c != nullptr);
    REQUIRE(d != nullptr);
    REQUIRE(out0 != nullptr);
    REQUIRE(out1 != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out0);
    ggml_build_forward_expand(graph, out1);
    graph->uid = 1002;

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);

    std::vector<float> a_data(element_count);
    std::vector<float> b_data(element_count);
    std::vector<float> c_data(element_count);
    std::vector<float> d_data(element_count);
    std::vector<float> expected0(element_count);
    std::vector<float> expected1(element_count);
    for (int64_t i = 0; i < element_count; ++i) {
        a_data[i]    = static_cast<float>(i % 17) * 0.25f - 2.0f;
        b_data[i]    = static_cast<float>(i % 13) * -0.5f + 3.0f;
        c_data[i]    = static_cast<float>(i % 19) * 0.125f + 1.0f;
        d_data[i]    = static_cast<float>(i % 11) * 0.75f - 4.0f;
        expected0[i] = a_data[i] + b_data[i];
        expected1[i] = c_data[i] + d_data[i];
    }

    ggml_backend_tensor_set(a, a_data.data(), 0, a_data.size() * sizeof(float));
    ggml_backend_tensor_set(b, b_data.data(), 0, b_data.size() * sizeof(float));
    ggml_backend_tensor_set(c, c_data.data(), 0, c_data.size() * sizeof(float));
    ggml_backend_tensor_set(d, d_data.data(), 0, d_data.size() * sizeof(float));

    ggml_backend_hrx_cache_stats cache_stats = {};
    REQUIRE(ggml_backend_hrx_get_cache_stats(backend, &cache_stats));
    REQUIRE(cache_stats.graph_program_builds == 0);
    REQUIRE(cache_stats.graph_program_hits == 0);
    REQUIRE(cache_stats.prepared_program_builds == 0);
    REQUIRE(cache_stats.prepared_program_hits == 0);

    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    REQUIRE(ggml_backend_hrx_get_cache_stats(backend, &cache_stats));
    REQUIRE(cache_stats.graph_program_builds == 1);
    REQUIRE(cache_stats.graph_program_hits == 0);
    REQUIRE(cache_stats.prepared_program_builds == 1);
    REQUIRE(cache_stats.prepared_program_hits == 0);

    std::vector<float> actual0(element_count);
    std::vector<float> actual1(element_count);
    ggml_backend_tensor_get(out0, actual0.data(), 0, actual0.size() * sizeof(float));
    ggml_backend_tensor_get(out1, actual1.data(), 0, actual1.size() * sizeof(float));
    for (int64_t i = 0; i < element_count; ++i) {
        REQUIRE(actual0[i] == expected0[i]);
        REQUIRE(actual1[i] == expected1[i]);
    }

    for (int64_t i = 0; i < element_count; ++i) {
        a_data[i]    = static_cast<float>(i % 23) * -0.25f + 5.0f;
        b_data[i]    = static_cast<float>(i % 7) * 0.5f - 1.0f;
        c_data[i]    = static_cast<float>(i % 5) * -0.125f + 2.0f;
        d_data[i]    = static_cast<float>(i % 29) * 0.75f - 6.0f;
        expected0[i] = a_data[i] + b_data[i];
        expected1[i] = c_data[i] + d_data[i];
    }
    ggml_backend_tensor_set(a, a_data.data(), 0, a_data.size() * sizeof(float));
    ggml_backend_tensor_set(b, b_data.data(), 0, b_data.size() * sizeof(float));
    ggml_backend_tensor_set(c, c_data.data(), 0, c_data.size() * sizeof(float));
    ggml_backend_tensor_set(d, d_data.data(), 0, d_data.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    REQUIRE(ggml_backend_hrx_get_cache_stats(backend, &cache_stats));
    REQUIRE(cache_stats.graph_program_builds == 1);
    REQUIRE(cache_stats.graph_program_hits == 1);
    REQUIRE(cache_stats.prepared_program_builds == 1);
    REQUIRE(cache_stats.prepared_program_hits == 1);

    ggml_backend_tensor_get(out0, actual0.data(), 0, actual0.size() * sizeof(float));
    ggml_backend_tensor_get(out1, actual1.data(), 0, actual1.size() * sizeof(float));
    for (int64_t i = 0; i < element_count; ++i) {
        REQUIRE(actual0[i] == expected0[i]);
        REQUIRE(actual1[i] == expected1[i]);
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static void run_chained_add_f32() {
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
    ggml_tensor *     c             = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, element_count);
    ggml_tensor *     sum           = ggml_add(ctx, a, b);
    ggml_tensor *     out           = ggml_add(ctx, sum, c);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(c != nullptr);
    REQUIRE(sum != nullptr);
    REQUIRE(out != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out);
    graph->uid = 1004;

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);

    std::vector<float> a_data(element_count);
    std::vector<float> b_data(element_count);
    std::vector<float> c_data(element_count);
    std::vector<float> expected(element_count);
    for (int64_t i = 0; i < element_count; ++i) {
        a_data[i]   = static_cast<float>(i % 17) * 0.25f - 2.0f;
        b_data[i]   = static_cast<float>(i % 13) * -0.5f + 3.0f;
        c_data[i]   = static_cast<float>(i % 7) * 0.125f + 1.0f;
        expected[i] = a_data[i] + b_data[i] + c_data[i];
    }

    ggml_backend_tensor_set(a, a_data.data(), 0, a_data.size() * sizeof(float));
    ggml_backend_tensor_set(b, b_data.data(), 0, b_data.size() * sizeof(float));
    ggml_backend_tensor_set(c, c_data.data(), 0, c_data.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    std::vector<float> actual(element_count);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    for (int64_t i = 0; i < element_count; ++i) {
        REQUIRE(actual[i] == expected[i]);
    }

    ggml_backend_hrx_cache_stats cache_stats = {};
    REQUIRE(ggml_backend_hrx_get_cache_stats(backend, &cache_stats));
    REQUIRE(cache_stats.graph_program_builds == 1);
    REQUIRE(cache_stats.prepared_program_builds == 1);

    for (int64_t i = 0; i < element_count; ++i) {
        a_data[i]   = static_cast<float>(i % 11) * -0.25f + 5.0f;
        b_data[i]   = static_cast<float>(i % 5) * 0.5f - 1.0f;
        c_data[i]   = static_cast<float>(i % 19) * 0.75f - 6.0f;
        expected[i] = a_data[i] + b_data[i] + c_data[i];
    }
    ggml_backend_tensor_set(a, a_data.data(), 0, a_data.size() * sizeof(float));
    ggml_backend_tensor_set(b, b_data.data(), 0, b_data.size() * sizeof(float));
    ggml_backend_tensor_set(c, c_data.data(), 0, c_data.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    REQUIRE(ggml_backend_hrx_get_cache_stats(backend, &cache_stats));
    REQUIRE(cache_stats.graph_program_hits == 1);
    REQUIRE(cache_stats.prepared_program_hits == 1);

    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    for (int64_t i = 0; i < element_count; ++i) {
        REQUIRE(actual[i] == expected[i]);
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static void run_same_uid_distinct_graph_reuses_graph_program() {
    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);

    constexpr int64_t  element_count = 1024;
    std::vector<float> a_data(element_count);
    std::vector<float> b_data(element_count);
    std::vector<float> expected(element_count);

    ggml_init_params params0 = {};
    params0.mem_size         = 256 * 1024;
    params0.no_alloc         = true;
    ggml_context * ctx0      = ggml_init(params0);
    REQUIRE(ctx0 != nullptr);

    ggml_tensor * a0   = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, element_count);
    ggml_tensor * b0   = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, element_count);
    ggml_tensor * out0 = ggml_add(ctx0, a0, b0);
    REQUIRE(a0 != nullptr);
    REQUIRE(b0 != nullptr);
    REQUIRE(out0 != nullptr);

    ggml_cgraph * graph0 = ggml_new_graph(ctx0);
    REQUIRE(graph0 != nullptr);
    ggml_build_forward_expand(graph0, out0);
    graph0->uid = 1003;

    ggml_backend_buffer_t buffer0 = ggml_backend_alloc_ctx_tensors(ctx0, backend);
    REQUIRE(buffer0 != nullptr);

    for (int64_t i = 0; i < element_count; ++i) {
        a_data[i]   = static_cast<float>(i % 17) * 0.25f - 2.0f;
        b_data[i]   = static_cast<float>(i % 13) * -0.5f + 3.0f;
        expected[i] = a_data[i] + b_data[i];
    }
    ggml_backend_tensor_set(a0, a_data.data(), 0, a_data.size() * sizeof(float));
    ggml_backend_tensor_set(b0, b_data.data(), 0, b_data.size() * sizeof(float));
    REQUIRE(ggml_backend_graph_compute(backend, graph0) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    ggml_backend_hrx_cache_stats cache_stats = {};
    REQUIRE(ggml_backend_hrx_get_cache_stats(backend, &cache_stats));
    REQUIRE(cache_stats.graph_program_builds == 1);
    REQUIRE(cache_stats.graph_program_hits == 0);
    REQUIRE(cache_stats.prepared_program_builds == 1);
    REQUIRE(cache_stats.prepared_program_hits == 0);

    ggml_init_params params1 = {};
    params1.mem_size         = 256 * 1024;
    params1.no_alloc         = true;
    ggml_context * ctx1      = ggml_init(params1);
    REQUIRE(ctx1 != nullptr);

    ggml_tensor * a1   = ggml_new_tensor_1d(ctx1, GGML_TYPE_F32, element_count);
    ggml_tensor * b1   = ggml_new_tensor_1d(ctx1, GGML_TYPE_F32, element_count);
    ggml_tensor * out1 = ggml_add(ctx1, a1, b1);
    REQUIRE(a1 != nullptr);
    REQUIRE(b1 != nullptr);
    REQUIRE(out1 != nullptr);

    ggml_cgraph * graph1 = ggml_new_graph(ctx1);
    REQUIRE(graph1 != nullptr);
    ggml_build_forward_expand(graph1, out1);
    graph1->uid = 1003;

    ggml_backend_buffer_t buffer1 = ggml_backend_alloc_ctx_tensors(ctx1, backend);
    REQUIRE(buffer1 != nullptr);

    for (int64_t i = 0; i < element_count; ++i) {
        a_data[i]   = static_cast<float>(i % 23) * -0.25f + 5.0f;
        b_data[i]   = static_cast<float>(i % 7) * 0.5f - 1.0f;
        expected[i] = a_data[i] + b_data[i];
    }
    ggml_backend_tensor_set(a1, a_data.data(), 0, a_data.size() * sizeof(float));
    ggml_backend_tensor_set(b1, b_data.data(), 0, b_data.size() * sizeof(float));
    REQUIRE(ggml_backend_graph_compute(backend, graph1) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    REQUIRE(ggml_backend_hrx_get_cache_stats(backend, &cache_stats));
    REQUIRE(cache_stats.graph_program_builds == 1);
    REQUIRE(cache_stats.graph_program_hits == 1);
    REQUIRE(cache_stats.prepared_program_builds == 2);
    REQUIRE(cache_stats.prepared_program_hits == 0);

    std::vector<float> actual(element_count);
    ggml_backend_tensor_get(out1, actual.data(), 0, actual.size() * sizeof(float));
    for (int64_t i = 0; i < element_count; ++i) {
        REQUIRE(actual[i] == expected[i]);
    }

    ggml_backend_buffer_free(buffer1);
    ggml_free(ctx1);
    ggml_backend_buffer_free(buffer0);
    ggml_free(ctx0);
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
    run_status_checks();
    run_command_plan_metadata_checks();
    run_dispatch_registry_checks();
    run_graph_import_checks();
    run_graph_index_checks();
    run_graph_traversal_checks();
    run_qwen_flash_attention_dispatch_checks();
    run_qwen_matmul_dispatch_checks();
    run_qwen_router_top8_dispatch_checks();
    run_qwen_routed_gate_up_dispatch_checks();
    run_multi_dispatch_checks();
    run_transient_import_checks();
    run_chained_dispatch_requires_transients();
    run_multiple_transient_plan_checks();
    run_graph_program_cache_uid_mismatch_checks();
    run_graph_executor_contract_checks();

    if (ggml_backend_hrx_get_device_count() == 0) {
        std::fprintf(stderr, "test skipped: no HRX devices available\n");
        return 0;
    }

    run_add_f32();
    run_two_independent_add_f32();
    run_chained_add_f32();
    run_same_uid_distinct_graph_reuses_graph_program();
    run_unsupported_op_fails();
    return 0;
}
