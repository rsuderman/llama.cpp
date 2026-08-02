#include "ggml-hrx.h"

#include "graph/graph-ir.h"
#include "graph/optimizer.h"
#include "graph/qwen-rules.h"
#include "graph/qwen-program.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "hrx_runtime.h"

#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <vector>

namespace {

static constexpr size_t GGML_HRX_ALIGNMENT = 256;
static constexpr uintptr_t GGML_HRX_FAKE_PTR_BASE = 0x1000;

struct ggml_backend_hrx_device_context;

struct ggml_backend_hrx_buffer_type_context {
    ggml_backend_hrx_device_context * device;
    std::string name;
};

struct ggml_backend_hrx_buffer_context {
    ggml_backend_hrx_device_context * device;
    hrx_buffer_t buffer;
    uint8_t * base;
};

struct ggml_backend_hrx_device_context {
    hrx_device_t device = nullptr;
    std::string name;
    std::string description;
    size_t memory_total = 0;
    ggml_backend_buffer_type buft = {};
    ggml_backend_hrx_buffer_type_context buft_context = {};
};

struct ggml_backend_hrx_context {
    ggml_backend_hrx_device_context * device;
    hrx_stream_t stream;
    std::string name;
};

struct ggml_backend_hrx_plan {
    ggml::hrx::Graph graph;
    ggml::hrx::Schedule schedule;
    ggml::hrx::VerificationResult verification;
    size_t uncovered_operations = 0;
};

struct ggml_backend_hrx_reg_context {
    bool initialized = false;
    std::vector<std::unique_ptr<ggml_backend_hrx_device_context>> device_contexts;
    std::vector<ggml_backend_device> devices;

    ~ggml_backend_hrx_reg_context() {
        for (auto & context : device_contexts) {
            if (context->device != nullptr) {
                hrx_device_release(context->device);
            }
        }
        if (initialized) {
            hrx_status_t status = hrx_gpu_shutdown();
            if (!hrx_status_is_ok(status)) {
                hrx_status_ignore(status);
            }
        }
    }
};

static bool hrx_check(hrx_status_t status, const char * expression, const char * file, int line) {
    if (hrx_status_is_ok(status)) {
        return true;
    }
    char * message = nullptr;
    size_t length = 0;
    hrx_status_to_string(status, &message, &length);
    GGML_LOG_ERROR("%s:%d: %s failed: %s\n", file, line, expression, message != nullptr ? message : "unknown HRX error");
    hrx_status_free_message(message);
    hrx_status_ignore(status);
    return false;
}

#define HRX_CHECK(expression) hrx_check((expression), #expression, __FILE__, __LINE__)

static ggml_guid_t ggml_backend_hrx_guid() {
    static ggml_guid guid = {
        0xd2, 0x3d, 0x72, 0x83, 0xb2, 0x82, 0x4d, 0xe0, 0x8a, 0x3e, 0x21, 0x1d, 0x68, 0x87, 0x2f, 0x4b,
    };
    return &guid;
}

static ggml_backend_hrx_device_context * device_context(ggml_backend_dev_t device) {
    return static_cast<ggml_backend_hrx_device_context *>(device->context);
}

static ggml_backend_hrx_buffer_context * buffer_context(ggml_backend_buffer_t buffer) {
    return static_cast<ggml_backend_hrx_buffer_context *>(buffer->context);
}

static size_t tensor_offset(const ggml_backend_hrx_buffer_context * context, const ggml_tensor * tensor) {
    return static_cast<size_t>(static_cast<const uint8_t *>(tensor->data) - context->base);
}

static const char * buffer_type_name(ggml_backend_buffer_type_t buft) {
    return static_cast<ggml_backend_hrx_buffer_type_context *>(buft->context)->name.c_str();
}

static void buffer_free(ggml_backend_buffer_t buffer) {
    auto * context = buffer_context(buffer);
    if (context->buffer != nullptr) {
        hrx_buffer_release(context->buffer);
    }
    delete context;
}

static void * buffer_base(ggml_backend_buffer_t buffer) {
    return buffer_context(buffer)->base;
}

static void buffer_memset(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    if (size == 0) {
        return;
    }
    auto * context = buffer_context(buffer);
    hrx_stream_t stream = nullptr;
    if (!HRX_CHECK(hrx_stream_create(context->device->device, 0, &stream))) {
        return;
    }
    const size_t destination_offset = tensor_offset(context, tensor) + offset;
    const bool ok = HRX_CHECK(hrx_stream_fill_buffer(stream, context->buffer, destination_offset, size, &value, sizeof(value))) &&
        HRX_CHECK(hrx_stream_synchronize(stream));
    GGML_UNUSED(ok);
    hrx_stream_release(stream);
}

static void buffer_set(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    if (size == 0) {
        return;
    }
    auto * context = buffer_context(buffer);
    HRX_CHECK(hrx_synchronous_h2d(context->device->device, data, context->buffer, tensor_offset(context, tensor) + offset, size));
}

static void buffer_get(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    if (size == 0) {
        return;
    }
    auto * context = buffer_context(buffer);
    HRX_CHECK(hrx_synchronous_d2h(context->device->device, context->buffer, tensor_offset(context, tensor) + offset, data, size));
}

static bool buffer_copy(ggml_backend_buffer_t buffer, const ggml_tensor * source, ggml_tensor * destination) {
    ggml_backend_buffer_t source_buffer = source->view_src != nullptr ? source->view_src->buffer : source->buffer;
    if (source_buffer == nullptr || source_buffer->iface.get_base != buffer_base) {
        return false;
    }
    auto * source_context = buffer_context(source_buffer);
    auto * destination_context = buffer_context(buffer);
    if (source_context->device != destination_context->device) {
        return false;
    }
    hrx_stream_t stream = nullptr;
    if (!HRX_CHECK(hrx_stream_create(destination_context->device->device, 0, &stream))) {
        return false;
    }
    bool ok = HRX_CHECK(hrx_stream_copy_buffer(stream, source_context->buffer, tensor_offset(source_context, source),
        destination_context->buffer, tensor_offset(destination_context, destination), ggml_nbytes(source)));
    ok = ok && HRX_CHECK(hrx_stream_synchronize(stream));
    hrx_stream_release(stream);
    return ok;
}

static void buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    if (buffer->size == 0) {
        return;
    }
    auto * context = buffer_context(buffer);
    hrx_stream_t stream = nullptr;
    if (!HRX_CHECK(hrx_stream_create(context->device->device, 0, &stream))) {
        return;
    }
    const bool ok = HRX_CHECK(hrx_stream_fill_buffer(stream, context->buffer, 0, buffer->size, &value, sizeof(value))) &&
        HRX_CHECK(hrx_stream_synchronize(stream));
    GGML_UNUSED(ok);
    hrx_stream_release(stream);
}

static const ggml_backend_buffer_i buffer_i = {
    buffer_free, buffer_base, nullptr, buffer_memset, buffer_set, buffer_get, nullptr, nullptr, buffer_copy, buffer_clear, nullptr,
};

static ggml_backend_buffer_t buffer_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    auto * type_context = static_cast<ggml_backend_hrx_buffer_type_context *>(buft->context);
    hrx_buffer_params_t params = {
        HRX_MEMORY_TYPE_DEVICE_LOCAL, HRX_MEMORY_ACCESS_ALL, HRX_BUFFER_USAGE_DEFAULT, 0,
    };
    hrx_buffer_t allocation = nullptr;
    if (size > 0 && !HRX_CHECK(hrx_allocator_allocate_buffer(hrx_device_allocator(type_context->device->device), params, size, &allocation))) {
        return nullptr;
    }
    auto * context = new (std::nothrow) ggml_backend_hrx_buffer_context {
        type_context->device, allocation, reinterpret_cast<uint8_t *>(GGML_HRX_FAKE_PTR_BASE),
    };
    if (context == nullptr) {
        if (allocation != nullptr) {
            hrx_buffer_release(allocation);
        }
        return nullptr;
    }
    return ggml_backend_buffer_init(buft, buffer_i, context, size);
}

static size_t buffer_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return GGML_HRX_ALIGNMENT;
}

static size_t buffer_max_size(ggml_backend_buffer_type_t buft) {
    return static_cast<ggml_backend_hrx_buffer_type_context *>(buft->context)->device->memory_total;
}

static const ggml_backend_buffer_type_i buffer_type_i = {
    buffer_type_name, buffer_alloc, buffer_alignment, buffer_max_size, nullptr, nullptr,
};

static void dump_graph(const ggml_cgraph * graph, const char * mode, const char * stage) {
    const char * directory = std::getenv("GGML_HRX_DUMP_GRAPH_DIR");
    if (directory == nullptr || directory[0] == '\0') {
        return;
    }
    static std::atomic<uint64_t> sequence = 0;
    static std::mutex mutex;
    const uint64_t id = sequence.fetch_add(1);
    try {
        const ggml::hrx::Graph normalized = ggml::hrx::import_graph(graph);
        std::filesystem::create_directories(directory);
        const std::string stem = std::to_string(id) + "-uid-" + std::to_string(graph->uid) + "-" + mode + "-" + stage;
        const std::filesystem::path dot_path = std::filesystem::path(directory) / (stem + ".dot");
        ggml_graph_dump_dot(graph, graph, dot_path.string().c_str());
        std::lock_guard<std::mutex> lock(mutex);
        const std::filesystem::path normalized_directory = std::filesystem::path(directory) / "normalized";
        std::filesystem::create_directories(normalized_directory);
        const std::filesystem::path json_path = normalized_directory / (normalized.fingerprint + ".json");
        if (!std::filesystem::exists(json_path)) {
            const std::filesystem::path temporary_path = json_path.string() + ".tmp";
            std::ofstream output(temporary_path, std::ios::binary | std::ios::trunc);
            output << ggml::hrx::serialize_graph_json(normalized) << '\n';
            output.close();
            std::filesystem::rename(temporary_path, json_path);
        }
        std::ofstream manifest(std::filesystem::path(directory) / "manifest.tsv", std::ios::app);
        manifest << id << '\t' << graph->uid << '\t' << mode << '\t' << stage << '\t'
                 << graph->n_nodes << '\t' << graph->n_leafs << '\t' << normalized.fingerprint << '\t'
                 << dot_path.filename().string() << '\t' << std::filesystem::relative(json_path, directory).string() << '\n';
    } catch (const std::exception & error) {
        GGML_LOG_ERROR("%s: graph dump failed: %s\n", __func__, error.what());
    }
}

static void dump_schedule(const ggml::hrx::Graph & graph, const ggml::hrx::Schedule & schedule) {
    const char * directory = std::getenv("GGML_HRX_DUMP_GRAPH_DIR");
    if (directory == nullptr || directory[0] == '\0') return;
    static std::mutex mutex;
    try {
        std::lock_guard<std::mutex> lock(mutex);
        const std::filesystem::path schedule_directory = std::filesystem::path(directory) / "candidate-schedules";
        std::filesystem::create_directories(schedule_directory);
        const std::filesystem::path path = schedule_directory / (graph.fingerprint + ".json");
        if (std::filesystem::exists(path)) return;
        const std::filesystem::path temporary_path = path.string() + ".tmp";
        std::ofstream output(temporary_path, std::ios::binary | std::ios::trunc);
        output << ggml::hrx::serialize_schedule_json(schedule) << '\n';
        output.close();
        std::error_code error;
        std::filesystem::rename(temporary_path, path, error);
        if (error) std::filesystem::remove(temporary_path);
    } catch (const std::exception & error) {
        GGML_LOG_ERROR("%s: schedule dump failed: %s\n", __func__, error.what());
    }
}

static void dump_signature(const ggml::hrx::Graph & graph, const ggml::hrx::QwenProgramProof & proof) {
    const char * directory = std::getenv("GGML_HRX_DUMP_GRAPH_DIR");
    if (directory == nullptr || directory[0] == '\0' || !proof.recognized()) return;
    try {
        const std::filesystem::path signature_directory = std::filesystem::path(directory) / "program-signatures";
        std::filesystem::create_directories(signature_directory);
        std::ofstream output(signature_directory / (graph.fingerprint + ".tsv"), std::ios::trunc);
        output << ggml::hrx::qwen_program_signature(proof);
    } catch (const std::exception & error) {
        GGML_LOG_ERROR("%s: signature dump failed: %s\n", __func__, error.what());
    }
}

static ggml_backend_graph_plan_t graph_plan_create(ggml_backend_t backend, const ggml_cgraph * graph) {
    GGML_UNUSED(backend);
    dump_graph(graph, "execute", "owner-split");
    auto * plan = new (std::nothrow) ggml_backend_hrx_plan;
    if (plan == nullptr) {
        return nullptr;
    }
    plan->graph = ggml::hrx::import_graph(graph);
    if (!plan->graph.valid()) {
        GGML_LOG_ERROR("%s: graph import failed: %s\n", __func__, plan->graph.errors.front().c_str());
        delete plan;
        return nullptr;
    }
    const ggml::hrx::QwenProgramProof qwen_proof = ggml::hrx::recover_owned_qwen3_moe_program(plan->graph);
    if (qwen_proof.recognized()) {
        plan->schedule = qwen_proof.schedule;
        plan->uncovered_operations = 0;
        plan->verification = ggml::hrx::verify_owned_qwen3_moe_program(plan->graph, qwen_proof);
        dump_signature(plan->graph, qwen_proof);
        GGML_LOG_WARN("%s: reconstructed %s owned schedule with %zu dispatches, zero CPU fallback, %zu native gaps, and %zu root seams\n",
            __func__, qwen_proof.schedule.workload.c_str(), ggml::hrx::schedule_dispatch_count(qwen_proof.schedule),
            qwen_proof.native_gaps.size(), qwen_proof.root_seams.size());
    } else {
        const std::vector<ggml::hrx::FusionRule> rules = ggml::hrx::canonical_qwen3_moe_rules();
        const ggml::hrx::Selection selection = ggml::hrx::select_regions(plan->graph, rules);
        plan->uncovered_operations = selection.uncovered_operations.size();
        plan->schedule = ggml::hrx::materialize_schedule_with_cpu_fallback(plan->graph, rules, selection);
        plan->verification = ggml::hrx::verify_schedule(plan->graph, plan->schedule);
    }
    dump_schedule(plan->graph, plan->schedule);
    GGML_LOG_WARN("%s: capture-only HRX plan %s has %zu semantic regions, %zu dispatches, and %zu CPU fallbacks; execution is disabled\n",
        __func__, plan->graph.fingerprint.c_str(), plan->schedule.invocations.size(),
        ggml::hrx::schedule_dispatch_count(plan->schedule),
        ggml::hrx::schedule_execution_kind_count(plan->schedule, ggml::hrx::KernelSpecialization::ExecutionKind::CpuFallback));
    if (!plan->verification.valid()) {
        GGML_LOG_ERROR("%s: candidate schedule verification failed: %s\n", __func__, plan->verification.errors.front().c_str());
    }
    return plan;
}

static void graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t opaque_plan) {
    GGML_UNUSED(backend);
    auto * plan = static_cast<ggml_backend_hrx_plan *>(opaque_plan);
    delete plan;
}

static enum ggml_status graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t opaque_plan) {
    GGML_UNUSED(backend);
    GGML_UNUSED(opaque_plan);
    GGML_LOG_ERROR("%s: refusing execution in the capture-only HRX build\n", __func__);
    return GGML_STATUS_FAILED;
}

static enum ggml_backend_graph_claim_result graph_claim(ggml_backend_t backend, const ggml_cgraph * graph, enum ggml_backend_graph_claim_mode mode) {
    GGML_UNUSED(backend);
    dump_graph(graph, mode == GGML_BACKEND_GRAPH_CLAIM_MODE_MEASURE ? "measure" : "execute", "claim");
    return graph->n_nodes > 0 ? GGML_BACKEND_GRAPH_CLAIM_ACCEPTED : GGML_BACKEND_GRAPH_CLAIM_DECLINED;
}

static const char * backend_name(ggml_backend_t backend) {
    return static_cast<ggml_backend_hrx_context *>(backend->context)->name.c_str();
}

static void backend_free(ggml_backend_t backend) {
    auto * context = static_cast<ggml_backend_hrx_context *>(backend->context);
    HRX_CHECK(hrx_stream_synchronize(context->stream));
    hrx_stream_release(context->stream);
    delete context;
    delete backend;
}

static void backend_synchronize(ggml_backend_t backend) {
    auto * context = static_cast<ggml_backend_hrx_context *>(backend->context);
    HRX_CHECK(hrx_stream_synchronize(context->stream));
}

static enum ggml_status graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    GGML_UNUSED(backend);
    GGML_UNUSED(graph);
    GGML_LOG_ERROR("HRX graphs must be executed through an owned graph plan\n");
    return GGML_STATUS_FAILED;
}

static const ggml_backend_i backend_i = {
    backend_name, backend_free,
    nullptr, nullptr, nullptr, nullptr, nullptr,
    backend_synchronize,
    graph_plan_create, graph_plan_free, nullptr, graph_plan_compute,
    graph_compute,
    nullptr, nullptr, nullptr,
    graph_claim,
};

static const char * device_name(ggml_backend_dev_t device) {
    return device_context(device)->name.c_str();
}

static const char * device_description(ggml_backend_dev_t device) {
    return device_context(device)->description.c_str();
}

static void device_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    *free = device_context(device)->memory_total;
    *total = device_context(device)->memory_total;
}

static enum ggml_backend_dev_type device_type(ggml_backend_dev_t device) {
    GGML_UNUSED(device);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void device_props(ggml_backend_dev_t device, ggml_backend_dev_props * props) {
    props->name = device_name(device);
    props->description = device_description(device);
    device_memory(device, &props->memory_free, &props->memory_total);
    props->type = GGML_BACKEND_DEVICE_TYPE_GPU;
    props->device_id = nullptr;
    props->caps = { false, false, false, false };
}

static ggml_backend_t device_init(ggml_backend_dev_t device, const char * parameters) {
    GGML_UNUSED(parameters);
    auto * device_ctx = device_context(device);
    hrx_stream_t stream = nullptr;
    if (!HRX_CHECK(hrx_stream_create(device_ctx->device, 0, &stream))) {
        return nullptr;
    }
    auto * context = new (std::nothrow) ggml_backend_hrx_context { device_ctx, stream, device_ctx->name };
    auto * backend = context != nullptr ? new (std::nothrow) ggml_backend { ggml_backend_hrx_guid(), backend_i, device, context } : nullptr;
    if (backend == nullptr) {
        delete context;
        hrx_stream_release(stream);
    }
    return backend;
}

static ggml_backend_buffer_type_t device_buffer_type(ggml_backend_dev_t device) {
    return &device_context(device)->buft;
}

static bool device_supports_op(ggml_backend_dev_t device, const ggml_tensor * op) {
    GGML_UNUSED(device);
    GGML_UNUSED(op);
    // Graph ownership is the capability boundary. Advertising individual operations ensures
    // model weights are allocated in HRX memory instead of creating a CPU weight island for
    // every logical node before graph_claim can run.
    return true;
}

static bool device_supports_buffer_type(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    return buft == &device_context(device)->buft;
}

static const ggml_backend_device_i device_i = {
    device_name, device_description, device_memory, device_type, device_props, device_init, device_buffer_type,
    nullptr, nullptr, device_supports_op, device_supports_buffer_type, nullptr, nullptr, nullptr, nullptr,
};

static const char * registry_name(ggml_backend_reg_t registry) {
    GGML_UNUSED(registry);
    return "HRX";
}

static size_t registry_device_count(ggml_backend_reg_t registry) {
    return static_cast<ggml_backend_hrx_reg_context *>(registry->context)->devices.size();
}

static ggml_backend_dev_t registry_device(ggml_backend_reg_t registry, size_t index) {
    auto * context = static_cast<ggml_backend_hrx_reg_context *>(registry->context);
    GGML_ASSERT(index < context->devices.size());
    return &context->devices[index];
}

static void * registry_proc(ggml_backend_reg_t registry, const char * name) {
    GGML_UNUSED(registry);
    GGML_UNUSED(name);
    return nullptr;
}

static const ggml_backend_reg_i registry_i = { registry_name, registry_device_count, registry_device, registry_proc };

static std::unique_ptr<ggml_backend_hrx_reg_context> create_registry_context() {
    auto context = std::make_unique<ggml_backend_hrx_reg_context>();
    hrx_status_t status = hrx_gpu_initialize(0);
    if (hrx_status_is_ok(status)) {
        context->initialized = true;
    } else if (hrx_status_code(status) == HRX_STATUS_ALREADY_EXISTS) {
        hrx_status_ignore(status);
    } else {
        hrx_status_ignore(status);
        return context;
    }
    int count = 0;
    if (!HRX_CHECK(hrx_gpu_device_count(&count))) {
        return context;
    }
    context->device_contexts.reserve(count);
    context->devices.reserve(count);
    for (int i = 0; i < count; ++i) {
        hrx_device_t hrx_device = nullptr;
        if (!HRX_CHECK(hrx_gpu_device_get(i, &hrx_device)) || hrx_device == nullptr) {
            continue;
        }
        hrx_device_retain(hrx_device);
        auto device_ctx = std::make_unique<ggml_backend_hrx_device_context>();
        device_ctx->device = hrx_device;
        device_ctx->name = "HRX" + std::to_string(i);
        std::array<char, 128> name = {};
        std::array<char, 128> architecture = {};
        HRX_CHECK(hrx_device_get_property(hrx_device, HRX_DEVICE_PROPERTY_NAME, name.data(), name.size()));
        HRX_CHECK(hrx_device_get_property(hrx_device, HRX_DEVICE_PROPERTY_ARCHITECTURE, architecture.data(), architecture.size()));
        uint64_t memory = 0;
        HRX_CHECK(hrx_device_get_property(hrx_device, HRX_DEVICE_PROPERTY_TOTAL_MEMORY, &memory, sizeof(memory)));
        device_ctx->memory_total = static_cast<size_t>(memory);
        device_ctx->description = std::string(name.data()) + " (" + architecture.data() + ")";
        device_ctx->buft_context = { device_ctx.get(), device_ctx->name };
        device_ctx->buft = { buffer_type_i, nullptr, &device_ctx->buft_context };
        context->device_contexts.emplace_back(std::move(device_ctx));
        context->devices.push_back({ device_i, nullptr, context->device_contexts.back().get() });
        context->device_contexts.back()->buft.device = &context->devices.back();
    }
    return context;
}

} // namespace

ggml_backend_reg_t ggml_backend_hrx_reg() {
    static std::unique_ptr<ggml_backend_hrx_reg_context> context = create_registry_context();
    static ggml_backend_reg registry = { GGML_BACKEND_API_VERSION, registry_i, context.get() };
    for (auto & device : context->devices) {
        device.reg = &registry;
    }
    return &registry;
}

ggml_backend_t ggml_backend_hrx_init(size_t device) {
    ggml_backend_reg_t registry = ggml_backend_hrx_reg();
    if (device >= ggml_backend_reg_dev_count(registry)) {
        return nullptr;
    }
    return ggml_backend_dev_init(ggml_backend_reg_dev_get(registry, device), nullptr);
}

bool ggml_backend_is_hrx(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_hrx_guid());
}

int ggml_backend_hrx_get_device_count() {
    return static_cast<int>(ggml_backend_reg_dev_count(ggml_backend_hrx_reg()));
}

ggml_backend_buffer_type_t ggml_backend_hrx_buffer_type(size_t device) {
    ggml_backend_reg_t registry = ggml_backend_hrx_reg();
    return device < ggml_backend_reg_dev_count(registry) ? ggml_backend_dev_buffer_type(ggml_backend_reg_dev_get(registry, device)) : nullptr;
}

GGML_BACKEND_DL_IMPL(ggml_backend_hrx_reg)
