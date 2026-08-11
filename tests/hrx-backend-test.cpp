#include "dispatch/command-program-bindings.h"
#include "dispatch/command-program-diagnostics.h"
#include "dispatch/command-program-resolver.h"
#include "dispatch/command-program.h"
#include "dispatch/dispatch-scheduler.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-hrx.h"
#include "ggml-impl.h"
#include "ggml.h"
#include "graph/graph.h"
#include "runtime/command-program-executor.h"
#include "runtime/graph-executor.h"
#include "runtime/graph-program-cache.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
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

static bool error_log_contains(const ggml::hrx::ErrorLog & errors, const char * text) {
    for (const std::string & message : errors.messages()) {
        if (message.find(text) != std::string::npos) {
            return true;
        }
    }
    return false;
}

static bool string_contains(const std::string & value, const char * text) {
    return value.find(text) != std::string::npos;
}

static void run_error_log_checks() {
    ggml::hrx::ErrorLog errors;
    REQUIRE(errors.success());
    REQUIRE(errors.empty());
    REQUIRE(errors.size() == 0);

    errors.log("first");
    REQUIRE(!errors.success());
    REQUIRE(!errors.empty());
    REQUIRE(errors.size() == 1);
    REQUIRE(errors.front() == "first");

    errors.log("value %d", 7);
    REQUIRE(errors.size() == 2);
    REQUIRE(errors.messages()[1] == "value 7");

    ggml::hrx::ErrorLog other;
    other.log("third");
    errors.append(other);
    REQUIRE(errors.size() == 3);
    REQUIRE(errors.messages()[2] == "third");
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
    REQUIRE(scheduler.schedule_graph(imported.graph));
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

    ggml::hrx::CommandProgram offset_command           = commands;
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
    REQUIRE(error_log_contains(resolved.errors, "is not bound"));
    REQUIRE(error_log_contains(resolved.errors, "binding output"));
    REQUIRE(error_log_contains(resolved.errors, "value="));

    resolved = ggml::hrx::resolve_command_program_bindings(commands, partial_bindings);
    REQUIRE(!resolved.valid());
    REQUIRE(error_log_contains(resolved.errors, "is not bound"));

    resolved = ggml::hrx::resolve_command_program_bindings(commands, empty_runtime_binding);
    REQUIRE(!resolved.valid());
    REQUIRE(error_log_contains(resolved.errors, "empty binding"));

    ggml::hrx::ValueMap null_values = imported.graph.values();
    REQUIRE(null_values.bind_buffer(a_value->id, { nullptr, 0, a_value->byte_count }));
    REQUIRE(null_values.bind_buffer(out_value->id, { dummy_hrx_buffer(0x2000), 0, out_value->byte_count }));
    const ggml::hrx::CommandProgramBindings null_bindings =
        ggml::hrx::CommandProgramBindings::from_value_map(null_values);
    REQUIRE(!null_bindings.valid());
    resolved = ggml::hrx::resolve_command_program_bindings(commands, null_bindings);
    REQUIRE(!resolved.valid());
    REQUIRE(error_log_contains(resolved.errors, "null buffer"));
    REQUIRE(error_log_contains(resolved.errors, "binding a"));

    ggml::hrx::CommandProgram empty_resolve_binding           = commands;
    empty_resolve_binding.commands.front().bindings[0].length = 0;
    resolved = ggml::hrx::resolve_command_program_bindings(empty_resolve_binding, runtime_bindings);
    REQUIRE(!resolved.valid());
    REQUIRE(error_log_contains(resolved.errors, "empty range"));
    REQUIRE(error_log_contains(resolved.errors, "range=[0, 0)"));

    ggml::hrx::CommandProgram out_of_range_binding           = commands;
    out_of_range_binding.commands.front().bindings[0].offset = a_value->byte_count;
    out_of_range_binding.commands.front().bindings[0].length = 4;
    resolved = ggml::hrx::resolve_command_program_bindings(out_of_range_binding, runtime_bindings);
    REQUIRE(!resolved.valid());
    REQUIRE(error_log_contains(resolved.errors, "outside runtime binding length"));
    REQUIRE(error_log_contains(resolved.errors, "binding a"));

    ggml::hrx::CommandProgram unsupported_origin           = commands;
    unsupported_origin.commands.front().bindings[0].origin = static_cast<ggml::hrx::CommandBindingOrigin>(255);
    resolved = ggml::hrx::resolve_command_program_bindings(unsupported_origin, runtime_bindings);
    REQUIRE(!resolved.valid());
    REQUIRE(error_log_contains(resolved.errors, "unsupported binding origin"));
    REQUIRE(error_log_contains(resolved.errors, "origin=Unknown(255)"));
    ggml::hrx::VerificationResult verification =
        ggml::hrx::verify_command_program(unsupported_origin, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(error_log_contains(verification.errors, "unsupported binding origin"));
    REQUIRE(error_log_contains(verification.errors, "origin=Unknown(255)"));

    ggml::hrx::CommandProgram invalid_kernel         = commands;
    invalid_kernel.commands.front().kernel.kernel_id = ggml::hrx::kUncatalogedKernelId;
    verification = ggml::hrx::verify_command_program(invalid_kernel, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(error_log_contains(verification.errors, "command 0"));
    REQUIRE(error_log_contains(verification.errors, "kernel_id="));

    ggml::hrx::CommandProgram empty_bindings = commands;
    empty_bindings.commands.front().bindings.clear();
    verification = ggml::hrx::verify_command_program(empty_bindings, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(error_log_contains(verification.errors, "bindings=0"));

    ggml::hrx::CommandProgram empty_binding           = commands;
    empty_binding.commands.front().bindings[0].length = 0;
    verification = ggml::hrx::verify_command_program(empty_binding, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(error_log_contains(verification.errors, "binding a"));
    REQUIRE(error_log_contains(verification.errors, "range=[0, 0)"));

    ggml::hrx::CommandProgram invalid_value          = commands;
    invalid_value.commands.front().bindings[0].value = ggml::hrx::ValueId(-1);
    verification = ggml::hrx::verify_command_program(invalid_value, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(error_log_contains(verification.errors, "value=-1"));

    ggml::hrx::CommandProgram forward_dependency = commands;
    forward_dependency.commands.front().dependencies.push_back(0);
    verification =
        ggml::hrx::verify_command_program(forward_dependency, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(error_log_contains(verification.errors, "forward dependency 0"));

    ggml::hrx::CommandProgram wrong_binding_name         = commands;
    wrong_binding_name.commands.front().bindings[0].name = "wrong";
    verification =
        ggml::hrx::verify_command_program(wrong_binding_name, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(error_log_contains(verification.errors, "binding wrong"));

    ggml::hrx::CommandProgram wrong_binding_access           = commands;
    wrong_binding_access.commands.front().bindings[0].access = ggml::hrx::ResourceAccess::Write;
    verification =
        ggml::hrx::verify_command_program(wrong_binding_access, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(error_log_contains(verification.errors, "access=Write"));

    ggml_hrx_loom_jit_amdgpu *                      jit             = nullptr;
    const ggml::hrx::KernelCorpus &                 corpus          = ggml::hrx::get_qwen_kernel_corpus();
    const ggml::hrx::CommandProgramExecutionContext prepare_context = {
        nullptr, nullptr, "gfx1151", &corpus, &jit, nullptr, nullptr,
    };

    ggml::hrx::PreparedCommandProgram prepared =
        ggml::hrx::prepare_command_program(prepare_context, invalid_kernel, runtime_bindings);
    REQUIRE(!prepared.valid());
    REQUIRE(error_log_contains(prepared.errors, "kernel_id="));

    prepared = ggml::hrx::prepare_command_program(prepare_context, commands, missing_bindings);
    REQUIRE(!prepared.valid());
    REQUIRE(error_log_contains(prepared.errors, "is not bound"));

    prepared = ggml::hrx::prepare_command_program(prepare_context, commands, runtime_bindings);
    REQUIRE(!prepared.valid());
    REQUIRE(error_log_contains(prepared.errors, "missing HRX device"));

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
    REQUIRE(scheduler.schedule_graph(imported.graph));
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
    REQUIRE(!scheduler.schedule_graph(imported.graph));
    REQUIRE(!scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.empty());
    REQUIRE(error_log_contains(scheduler.plan().errors, "unsupported HRX node 1"));

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
    REQUIRE(scheduler.schedule_graph(imported.graph));
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
    REQUIRE(error_log_contains(resolved.errors, "no transient arena"));
    REQUIRE(error_log_contains(resolved.errors, "origin=Transient"));
    REQUIRE(error_log_contains(resolved.errors, "value="));

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
    REQUIRE(error_log_contains(invalid_transient_resolved.errors, "no transient arena allocation id"));

    ggml::hrx::CommandProgram missing_allocation = commands;
    missing_allocation.transients.allocations.clear();
    ggml::hrx::VerificationResult verification =
        ggml::hrx::verify_command_program(missing_allocation, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(error_log_contains(verification.errors, "no transient allocation"));

    ggml::hrx::CommandProgram out_of_range      = commands;
    out_of_range.commands[0].bindings[2].length = sum_value->byte_count + 1;
    verification = ggml::hrx::verify_command_program(out_of_range, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(error_log_contains(verification.errors, "outside transient allocation length"));

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
    REQUIRE(scheduler.schedule_graph(imported.graph));
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
    REQUIRE(add_support.errors.success());

    const ggml::hrx::GraphExecutionResult missing_binding = executor.execute(*add_graph);
    REQUIRE(!missing_binding.success());
    REQUIRE(missing_binding.status == GGML_STATUS_FAILED);
    REQUIRE(error_log_contains(missing_binding.errors, "external value"));
    REQUIRE(error_log_contains(missing_binding.errors, "not bound"));

    ggml_cgraph * sqr_graph = ggml_new_graph(ctx);
    REQUIRE(sqr_graph != nullptr);
    ggml_build_forward_expand(sqr_graph, sqr_out);
    const ggml::hrx::GraphSupportResult sqr_support = executor.can_execute(*sqr_graph);
    REQUIRE(!sqr_support.supported);
    REQUIRE(error_log_contains(sqr_support.errors, "unsupported HRX node 0"));
    REQUIRE(error_log_contains(sqr_support.errors, "SQR"));

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
    run_error_log_checks();
    run_graph_import_checks();
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
