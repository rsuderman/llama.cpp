#include "ggml-hrx-loom-catalog-runtime-internal.h"
#include "ggml-hrx-runtime-util.h"

#include <loomc/loomc.h>
#include <loomc/target/amdgpu.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <vector>

namespace {

struct ggml_backend_hrx_loaded_loom_route {
    std::string                  cache_key;
    hrx_executable_t             executable     = nullptr;
    uint32_t                     export_ordinal = 0;
    hrx_executable_export_info_t export_info    = {};
    std::string                  report_json;

    ~ggml_backend_hrx_loaded_loom_route() {
        if (executable) {
            hrx_executable_release(executable);
        }
    }
};

#define GGML_HRX_LOOM_CHECK(expr) ggml_backend_hrx_log_hrx_status((expr), #expr, __FILE__, __LINE__)

static void ggml_backend_hrx_loom_log_status(loomc_status_t status, const char * expr, const char * file, int line) {
    if (loomc_status_is_ok(status)) {
        return;
    }

    char               buffer[1024] = {};
    loomc_host_size_t  length       = 0;
    const bool         formatted    = loomc_status_format(status, sizeof(buffer), buffer, &length);
    const char * const message      = formatted ? buffer : loomc_status_code_string(loomc_status_code(status));
    GGML_LOG_ERROR("%s:%d: %s failed: %s\n", file, line, expr, message ? message : "unknown Loom error");
    loomc_status_free(status);
}

#define GGML_HRX_LOOMC_CHECK(expr) ggml_backend_hrx_loom_log_status((expr), #expr, __FILE__, __LINE__)

static ggml_backend_hrx_loom_op_response ggml_backend_hrx_loom_response(
    ggml_backend_hrx_loom_result             result,
    ggml_backend_hrx_loom_unsupported_reason unsupported_reason,
    const char *                             route_id) {
    return {
        /* .result             = */ result,
        /* .unsupported_reason = */ unsupported_reason,
        /* .route_id           = */ route_id,
    };
}

static void ggml_backend_hrx_loom_compile_output_free(ggml_backend_hrx_loom_compile_output * output) {
    if (!output) {
        return;
    }

    std::free(output->executable_data);
    std::free(output->report_json);
    *output = {};
}

static void ggml_backend_hrx_loom_log_result_diagnostics(const loomc_result_t * result) {
    if (!result) {
        return;
    }

    for (loomc_host_size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
        const loomc_diagnostic_t * diagnostic = loomc_result_diagnostic_at(result, i);
        if (!diagnostic) {
            continue;
        }
        GGML_LOG_ERROR("Loom diagnostic: %.*s: %.*s\n", static_cast<int>(diagnostic->code.size),
                       diagnostic->code.data, static_cast<int>(diagnostic->message.size), diagnostic->message.data);
    }
}

static const loomc_artifact_t * ggml_backend_hrx_loom_find_artifact(const loomc_result_t *     result,
                                                                    loomc_artifact_kind_t     kind,
                                                                    loomc_string_view_t       format) {
    if (!result) {
        return nullptr;
    }

    for (loomc_host_size_t i = 0; i < loomc_result_artifact_count(result); ++i) {
        const loomc_artifact_t * artifact = loomc_result_artifact_at(result, i);
        if (!artifact) {
            continue;
        }
        if (artifact->kind == kind && loomc_string_view_equal(artifact->format, format)) {
            return artifact;
        }
    }
    return nullptr;
}

static bool ggml_backend_hrx_loom_require_success(loomc_result_t * result, const char * phase) {
    if (result && loomc_result_succeeded(result)) {
        return true;
    }

    GGML_LOG_ERROR("%s: Loom %s failed\n", __func__, phase);
    ggml_backend_hrx_loom_log_result_diagnostics(result);
    return false;
}

static bool ggml_backend_hrx_loom_copy_artifact_bytes(const loomc_artifact_t * artifact,
                                                      void **                  out_data,
                                                      size_t *                 out_size) {
    if (!artifact || !out_data || !out_size || !artifact->contents.data || artifact->contents.data_length == 0) {
        return false;
    }

    void * data = std::malloc(artifact->contents.data_length);
    if (!data) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes for Loom artifact\n", __func__,
                       static_cast<size_t>(artifact->contents.data_length));
        return false;
    }

    std::memcpy(data, artifact->contents.data, artifact->contents.data_length);
    *out_data = data;
    *out_size = artifact->contents.data_length;
    return true;
}

struct ggml_backend_hrx_loom_compile_state {
    loomc_target_environment_t * target_environment = nullptr;
    loomc_context_t *            context            = nullptr;
    loomc_workspace_t *          workspace          = nullptr;
    loomc_source_t *             source             = nullptr;
    loomc_module_t *             module             = nullptr;
    loomc_target_profile_t *     target_profile     = nullptr;
    loomc_target_selection_t *   target_selection   = nullptr;
    loomc_compiler_t *           compiler           = nullptr;
    loomc_pass_program_t *       pass_program       = nullptr;
    loomc_result_t *             result             = nullptr;

    ~ggml_backend_hrx_loom_compile_state() {
        loomc_result_release(result);
        loomc_pass_program_release(pass_program);
        loomc_compiler_release(compiler);
        loomc_target_selection_release(target_selection);
        loomc_target_profile_release(target_profile);
        loomc_module_release(module);
        loomc_source_release(source);
        loomc_workspace_release(workspace);
        loomc_context_release(context);
        loomc_target_environment_release(target_environment);
    }

    void reset_result() {
        loomc_result_release(result);
        result = nullptr;
    }
};

}  // namespace

ggml_backend_hrx_loom_op_response ggml_backend_hrx_loom_unsupported(ggml_backend_hrx_loom_unsupported_reason reason) {
    return ggml_backend_hrx_loom_response(GGML_BACKEND_HRX_LOOM_UNSUPPORTED, reason, nullptr);
}

ggml_backend_hrx_loom_op_response ggml_backend_hrx_loom_supported(const char * route_id) {
    return ggml_backend_hrx_loom_response(GGML_BACKEND_HRX_LOOM_INVOKED, GGML_BACKEND_HRX_LOOM_UNSUPPORTED_NONE,
                                          route_id);
}

ggml_backend_hrx_loom_op_response ggml_backend_hrx_loom_failed(const char * route_id) {
    return ggml_backend_hrx_loom_response(GGML_BACKEND_HRX_LOOM_FAILED, GGML_BACKEND_HRX_LOOM_UNSUPPORTED_NONE,
                                          route_id);
}

int64_t ggml_backend_hrx_loom_next_power_of_2(int64_t value) {
    int64_t result = 1;
    while (result < value) {
        result <<= 1;
    }
    return result;
}

struct ggml_backend_hrx_loom_catalog {
    hrx_device_t                                                     device = nullptr;
    std::string                                                      architecture;
    std::string                                                      target;
    std::mutex                                                       routes_mutex;
    std::vector<std::unique_ptr<ggml_backend_hrx_loaded_loom_route>> routes;

    ~ggml_backend_hrx_loom_catalog() {
        routes.clear();
        if (device) {
            hrx_device_release(device);
        }
    }
};

const ggml_backend_hrx_loom_catalog_entry * ggml_backend_hrx_loom_find_entry(
    const ggml_backend_hrx_loom_catalog * catalog,
    const char *                          route_id) {
    if (!catalog || !route_id) {
        return nullptr;
    }

    size_t                                      count   = 0;
    const ggml_backend_hrx_loom_catalog_entry * entries = ggml_backend_hrx_loom_catalog_entries(&count);
    for (size_t i = 0; i < count; ++i) {
        if (std::strcmp(entries[i].id, route_id) == 0 && catalog->target == entries[i].target) {
            return &entries[i];
        }
    }
    return nullptr;
}

namespace {

static ggml_backend_hrx_loaded_loom_route * ggml_backend_hrx_loom_get_loaded_route(
    ggml_backend_hrx_loom_catalog *              catalog,
    const ggml_backend_hrx_loom_execution_plan * plan) {
    if (!catalog || !plan || !plan->entry) {
        return nullptr;
    }
    if (plan->config_binding_count > GGML_BACKEND_HRX_LOOM_MAX_CONFIG_BINDINGS) {
        GGML_LOG_ERROR("%s: route %s has too many config bindings: %zu\n", __func__, plan->entry->id,
                       plan->config_binding_count);
        return nullptr;
    }

    const std::string cache_key = ggml_backend_hrx_loom_cache_key(catalog->target.c_str(), plan);

    std::lock_guard<std::mutex> lock(catalog->routes_mutex);
    for (const auto & route : catalog->routes) {
        if (route->cache_key == cache_key) {
            return route.get();
        }
    }

    ggml_backend_hrx_loom_compile_input compile_input = {
        /* .source_data          = */ plan->entry->source_data,
        /* .source_size          = */ plan->entry->source_size,
        /* .source_format        = */ plan->entry->source_format,
        /* .source_name          = */ plan->entry->source_name,
        /* .target               = */ catalog->target.c_str(),
        /* .symbol               = */ plan->entry->symbol,
        /* .config_bindings      = */ plan->config_bindings,
        /* .config_binding_count = */ plan->config_binding_count,
    };
    ggml_backend_hrx_loom_compile_output compile_output = {};
    if (!ggml_backend_hrx_loom_compile(&compile_input, &compile_output)) {
        ggml_backend_hrx_loom_compile_output_free(&compile_output);
        return nullptr;
    }

    hrx_executable_t executable = nullptr;
    if (!compile_output.executable_data || compile_output.executable_size == 0 ||
        !GGML_HRX_LOOM_CHECK(hrx_executable_load_data(catalog->device, compile_output.executable_data,
                                                      compile_output.executable_size, "amdgpu",
                                                      catalog->target.c_str(), &executable))) {
        ggml_backend_hrx_loom_compile_output_free(&compile_output);
        return nullptr;
    }

    uint32_t export_ordinal = 0;
    if (!GGML_HRX_LOOM_CHECK(hrx_executable_lookup_export_by_name(executable, plan->entry->symbol, &export_ordinal))) {
        hrx_executable_release(executable);
        ggml_backend_hrx_loom_compile_output_free(&compile_output);
        return nullptr;
    }

    hrx_executable_export_info_t export_info = {};
    if (!GGML_HRX_LOOM_CHECK(hrx_executable_export_info(executable, export_ordinal, &export_info))) {
        hrx_executable_release(executable);
        ggml_backend_hrx_loom_compile_output_free(&compile_output);
        return nullptr;
    }

    if (!ggml_backend_hrx_export_abi_matches(__func__, plan->entry->id, export_info, plan->entry->binding_count,
                                             plan->entry->parameter_count, plan->entry->constant_byte_length)) {
        hrx_executable_release(executable);
        ggml_backend_hrx_loom_compile_output_free(&compile_output);
        return nullptr;
    }

    auto route            = std::make_unique<ggml_backend_hrx_loaded_loom_route>();
    route->cache_key      = cache_key;
    route->executable     = executable;
    route->export_ordinal = export_ordinal;
    route->export_info    = export_info;
    if (compile_output.report_json && compile_output.report_json_size > 0) {
        route->report_json.assign(compile_output.report_json, compile_output.report_json_size);
    }
    catalog->routes.push_back(std::move(route));
    ggml_backend_hrx_loom_compile_output_free(&compile_output);
    return catalog->routes.back().get();
}

static bool ggml_backend_hrx_loom_dispatch_plan(ggml_backend_hrx_loom_catalog *              catalog,
                                                hrx_stream_t                                 stream,
                                                const ggml_backend_hrx_loom_execution_plan * plan) {
    if (!plan || !plan->entry) {
        return false;
    }

    auto * route = ggml_backend_hrx_loom_get_loaded_route(catalog, plan);
    if (!route || !stream) {
        return false;
    }

    if (!ggml_backend_hrx_dispatch_abi_matches(__func__, plan->entry->id, plan->binding_count,
                                               plan->entry->binding_count, plan->constants_size,
                                               plan->entry->constant_byte_length)) {
        return false;
    }

    return GGML_HRX_LOOM_CHECK(hrx_stream_dispatch(stream, route->executable, route->export_ordinal, &plan->dispatch,
                                                   plan->constants, plan->constants_size, plan->bindings,
                                                   plan->binding_count, HRX_DISPATCH_FLAG_NONE));
}

}  // namespace

bool ggml_backend_hrx_loom_bind_tensor(const ggml_backend_hrx_loom_op_request * request,
                                       const ggml_tensor *                      tensor,
                                       hrx_buffer_ref_t *                       out_ref) {
    return request && request->bind_tensor && request->bind_tensor(request->bind_tensor_user_data, tensor, out_ref);
}

ggml_backend_hrx_loom_op_response ggml_backend_hrx_loom_match_request(
    ggml_backend_hrx_loom_catalog * catalog, const ggml_backend_hrx_loom_op_request * request) {
    return ggml_backend_hrx_loom_match_or_prepare_request(catalog, request, nullptr);
}

ggml_backend_hrx_loom_op_response ggml_backend_hrx_loom_prepare_plan(
    ggml_backend_hrx_loom_catalog *          catalog,
    const ggml_backend_hrx_loom_op_request * request,
    ggml_backend_hrx_loom_execution_plan *   plan) {
    return ggml_backend_hrx_loom_match_or_prepare_request(catalog, request, plan);
}

bool ggml_backend_hrx_loom_compile(const ggml_backend_hrx_loom_compile_input * input,
                                   ggml_backend_hrx_loom_compile_output *      output) {
    if (output) {
        *output = {};
    }
    if (!input || !output || !input->source_data || input->source_size == 0 || !input->source_format ||
        !input->source_name || !input->target || !input->symbol) {
        GGML_LOG_ERROR("%s: invalid Loom compile input\n", __func__);
        return false;
    }
    if (input->config_binding_count > 0 && !input->config_bindings) {
        GGML_LOG_ERROR("%s: invalid Loom compile config bindings\n", __func__);
        return false;
    }
    if (std::strcmp(input->source_format, "loom-text") != 0) {
        GGML_LOG_ERROR("%s: unsupported Loom source format %s\n", __func__, input->source_format);
        return false;
    }

    std::vector<loomc_config_binding_t> config_bindings;
    config_bindings.reserve(input->config_binding_count);
    for (size_t i = 0; i < input->config_binding_count; ++i) {
        const ggml_backend_hrx_loom_config_binding & binding = input->config_bindings[i];
        config_bindings.push_back({
            /* .key   = */ loomc_make_cstring_view(binding.name),
            /* .value = */ loomc_make_cstring_view(binding.value),
        });
    }

    ggml_backend_hrx_loom_compile_state state;
    loomc_status_t status =
        loomc_target_environment_create_amdgpu(loomc_allocator_system(), &state.target_environment);
    if (!loomc_status_is_ok(status)) {
        GGML_HRX_LOOMC_CHECK(status);
        return false;
    }

    loomc_context_target_options_t context_target_options = {
        /* .type               = */ LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
        /* .structure_size     = */ sizeof(loomc_context_target_options_t),
        /* .next               = */ nullptr,
        /* .target_environment = */ state.target_environment,
    };
    loomc_context_options_t context_options = {
        /* .type           = */ LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
        /* .structure_size = */ sizeof(loomc_context_options_t),
        /* .next           = */ &context_target_options,
    };
    status = loomc_context_create(&context_options, loomc_allocator_system(), &state.context);
    if (!loomc_status_is_ok(status)) {
        GGML_HRX_LOOMC_CHECK(status);
        return false;
    }

    status = loomc_workspace_create(nullptr, loomc_allocator_system(), &state.workspace);
    if (!loomc_status_is_ok(status)) {
        GGML_HRX_LOOMC_CHECK(status);
        return false;
    }

    loomc_source_options_t source_options = {
        /* .type              = */ LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
        /* .structure_size    = */ sizeof(loomc_source_options_t),
        /* .next              = */ nullptr,
        /* .format            = */ LOOMC_SOURCE_FORMAT_TEXT,
        /* .identifier        = */ loomc_make_cstring_view(input->source_name),
        /* .contents          = */ loomc_make_byte_span(input->source_data, input->source_size),
        /* .storage           = */ LOOMC_SOURCE_STORAGE_BORROWED,
        /* .release           = */ nullptr,
        /* .release_user_data = */ nullptr,
    };
    status = loomc_source_create(&source_options, loomc_allocator_system(), &state.source);
    if (!loomc_status_is_ok(status)) {
        GGML_HRX_LOOMC_CHECK(status);
        return false;
    }

    loomc_amdgpu_profile_options_t profile_options = {
        /* .type           = */ LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS,
        /* .structure_size = */ sizeof(loomc_amdgpu_profile_options_t),
        /* .next           = */ nullptr,
        /* .identifier     = */ loomc_make_cstring_view(input->target),
        /* .processor      = */ loomc_make_cstring_view(input->target),
    };
    status = loomc_target_profile_create_amdgpu(state.target_environment, &profile_options, loomc_allocator_system(),
                                                &state.target_profile);
    if (!loomc_status_is_ok(status)) {
        GGML_HRX_LOOMC_CHECK(status);
        return false;
    }
    status = loomc_target_selection_create_from_profile(state.target_profile, loomc_allocator_system(),
                                                        &state.target_selection);
    if (!loomc_status_is_ok(status)) {
        GGML_HRX_LOOMC_CHECK(status);
        return false;
    }

    status = loomc_compiler_create(state.context, nullptr, loomc_allocator_system(), &state.compiler);
    if (!loomc_status_is_ok(status)) {
        GGML_HRX_LOOMC_CHECK(status);
        return false;
    }

    loomc_target_selection_options_t target_options = {
        /* .type             = */ LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
        /* .structure_size   = */ sizeof(loomc_target_selection_options_t),
        /* .next             = */ nullptr,
        /* .target_selection = */ state.target_selection,
    };
    loomc_target_pipeline_options_t pipeline_options = {
        /* .type                     = */ LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
        /* .structure_size           = */ sizeof(loomc_target_pipeline_options_t),
        /* .next                     = */ &target_options,
        /* .identifier               = */ loomc_make_cstring_view("hrx-loom-prepared-low"),
        /* .kind                     = */ LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW,
        /* .control_flow_lowering    = */ LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
        /* .source_to_low_max_errors = */ 20,
    };
    status = loomc_pass_program_create_from_target_pipeline(state.context, &pipeline_options, loomc_allocator_system(),
                                                            &state.pass_program, &state.result);
    if (!loomc_status_is_ok(status)) {
        GGML_HRX_LOOMC_CHECK(status);
        return false;
    }
    if (!ggml_backend_hrx_loom_require_success(state.result, "target pipeline preparation")) {
        return false;
    }
    state.reset_result();

    status = loomc_module_deserialize_from_source(state.context, state.workspace, state.source, nullptr,
                                                  loomc_allocator_system(), &state.module, &state.result);
    if (!loomc_status_is_ok(status)) {
        GGML_HRX_LOOMC_CHECK(status);
        return false;
    }
    if (!ggml_backend_hrx_loom_require_success(state.result, "source deserialization")) {
        return false;
    }
    state.reset_result();

    loomc_compile_options_t compile_options = {
        /* .type           = */ LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
        /* .structure_size = */ sizeof(loomc_compile_options_t),
        /* .next           = */ &target_options,
        /* .module_name    = */ loomc_make_cstring_view(input->symbol),
        /* .artifact_flags = */ 0,
        /* .config         = */
            {
                /* .bindings      = */ config_bindings.empty() ? nullptr : config_bindings.data(),
                /* .binding_count = */ config_bindings.size(),
                /* .json_object   = */ loomc_string_view_empty(),
                /* .flags         = */ LOOMC_CONFIG_POLICY_FLAG_REJECT_UNKNOWN |
                                      LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
            },
    };
    status = loomc_compile_module(state.compiler, state.workspace, state.pass_program, state.module, &compile_options,
                                  loomc_allocator_system(), &state.result);
    if (!loomc_status_is_ok(status)) {
        GGML_HRX_LOOMC_CHECK(status);
        return false;
    }
    if (!ggml_backend_hrx_loom_require_success(state.result, "compilation")) {
        return false;
    }
    state.reset_result();

    const loomc_option_entry_t emit_entries[] = {
        {
            /* .key   = */ loomc_make_cstring_view(LOOMC_EMIT_OPTION_KEY_IDENTIFIER),
            /* .value = */ loomc_make_cstring_view(input->symbol),
        },
    };
    loomc_amdgpu_emit_options_t amdgpu_options = {
        /* .type            = */ LOOMC_STRUCTURE_TYPE_AMDGPU_EMIT_OPTIONS,
        /* .structure_size  = */ sizeof(loomc_amdgpu_emit_options_t),
        /* .next            = */ &target_options,
        /* .runtime_globals = */ LOOMC_AMDGPU_RUNTIME_GLOBAL_NONE,
    };
    loomc_compile_report_options_t report_options = {
        /* .type           = */ LOOMC_STRUCTURE_TYPE_COMPILE_REPORT_OPTIONS,
        /* .structure_size = */ sizeof(loomc_compile_report_options_t),
        /* .next           = */ &amdgpu_options,
        /* .mode           = */ LOOMC_COMPILE_REPORT_MODE_SUMMARY,
        /* .identifier     = */ loomc_string_view_empty(),
    };
    loomc_option_dict_t option_dict = {
        /* .type           = */ LOOMC_STRUCTURE_TYPE_OPTION_DICT,
        /* .structure_size = */ sizeof(loomc_option_dict_t),
        /* .next           = */ &report_options,
        /* .entries        = */ emit_entries,
        /* .entry_count    = */ 1,
    };
    loomc_emit_options_t emit_options = {
        /* .type            = */ LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
        /* .structure_size  = */ sizeof(loomc_emit_options_t),
        /* .next            = */ &option_dict,
        /* .artifact_format = */ loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO),
        /* .identifier      = */ loomc_make_cstring_view(input->symbol),
        /* .artifact_flags  = */ LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
    };
    status = loomc_emit_module(state.target_environment, state.workspace, state.module, &emit_options,
                               loomc_allocator_system(), &state.result);
    if (!loomc_status_is_ok(status)) {
        GGML_HRX_LOOMC_CHECK(status);
        return false;
    }
    if (!ggml_backend_hrx_loom_require_success(state.result, "AMDGPU emission")) {
        return false;
    }

    const loomc_artifact_t * executable =
        ggml_backend_hrx_loom_find_artifact(state.result, LOOMC_ARTIFACT_KIND_EXECUTABLE,
                                            loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO));
    if (!ggml_backend_hrx_loom_copy_artifact_bytes(executable, &output->executable_data, &output->executable_size)) {
        ggml_backend_hrx_loom_compile_output_free(output);
        return false;
    }

    const loomc_artifact_t * report =
        ggml_backend_hrx_loom_find_artifact(state.result, LOOMC_ARTIFACT_KIND_REPORT,
                                            loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_COMPILE_REPORT_JSON));
    if (report && !ggml_backend_hrx_loom_copy_artifact_bytes(report, reinterpret_cast<void **>(&output->report_json),
                                                            &output->report_json_size)) {
        ggml_backend_hrx_loom_compile_output_free(output);
        return false;
    }

    return true;
}

ggml_backend_hrx_loom_catalog * ggml_backend_hrx_loom_catalog_new(hrx_device_t device, const char * architecture) {
    if (!device || !architecture || architecture[0] == '\0') {
        return nullptr;
    }

    auto * catalog = new (std::nothrow) ggml_backend_hrx_loom_catalog();
    if (!catalog) {
        return nullptr;
    }

    hrx_device_retain(device);
    catalog->device       = device;
    catalog->architecture = architecture;
    catalog->target       = ggml_backend_hrx_architecture_base(architecture);
    return catalog;
}

void ggml_backend_hrx_loom_catalog_free(ggml_backend_hrx_loom_catalog * catalog) {
    delete catalog;
}

ggml_backend_hrx_loom_op_response ggml_backend_hrx_loom_supports_op(ggml_backend_hrx_loom_catalog * catalog,
                                                                    const ggml_tensor *             op) {
    if (!catalog || !op) {
        return ggml_backend_hrx_loom_unsupported(GGML_BACKEND_HRX_LOOM_UNSUPPORTED_NO_ROUTE);
    }

    const ggml_backend_hrx_loom_op_request request = {
        /* .op                    = */ op,
        /* .cgraph                = */ nullptr,
        /* .node_index            = */ -1,
        /* .stream                = */ nullptr,
        /* .bind_tensor           = */ nullptr,
        /* .bind_tensor_user_data = */ nullptr,
    };
    return ggml_backend_hrx_loom_match_request(catalog, &request);
}

ggml_backend_hrx_loom_op_response ggml_backend_hrx_loom_invoke(
    ggml_backend_hrx_loom_catalog *          catalog,
    const ggml_backend_hrx_loom_op_request * request,
    ggml_backend_hrx_loom_consumed_nodes *   consumed_nodes) {
    if (consumed_nodes) {
        *consumed_nodes = {};
    }
    if (!catalog || !request || !request->op) {
        return ggml_backend_hrx_loom_failed(nullptr);
    }

    ggml_backend_hrx_loom_execution_plan plan     = {};
    ggml_backend_hrx_loom_op_response    response = ggml_backend_hrx_loom_prepare_plan(catalog, request, &plan);
    if (response.result != GGML_BACKEND_HRX_LOOM_INVOKED) {
        return response;
    }
    if (!ggml_backend_hrx_loom_dispatch_plan(catalog, request->stream, &plan)) {
        return ggml_backend_hrx_loom_failed(response.route_id);
    }
    if (consumed_nodes) {
        consumed_nodes->count = plan.consumed_node_count;
        const int copy_count = plan.consumed_node_count < GGML_BACKEND_HRX_LOOM_MAX_CONSUMED_NODES ?
            plan.consumed_node_count : GGML_BACKEND_HRX_LOOM_MAX_CONSUMED_NODES;
        for (int i = 0; i < copy_count; ++i) {
            consumed_nodes->indices[i] = plan.consumed_node_indices[i];
        }
    }
    return response;
}
