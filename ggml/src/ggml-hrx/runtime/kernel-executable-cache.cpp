#include "kernel-executable-cache.h"

#include "ggml-impl.h"
#include "hrx-interop-utils.h"

#include <cstring>
#include <limits>
#include <sstream>

namespace ggml::hrx {
namespace {

static ggml_hrx_loom_jit_source_format to_jit_source_format(KernelSourceFormat format) {
    switch (format) {
        case KERNEL_SOURCE_FORMAT_TEXT:
            return GGML_HRX_LOOM_JIT_SOURCE_FORMAT_TEXT;
        case KERNEL_SOURCE_FORMAT_BINARY:
            return GGML_HRX_LOOM_JIT_SOURCE_FORMAT_BYTECODE;
    }
    return GGML_HRX_LOOM_JIT_SOURCE_FORMAT_TEXT;
}

static void append_u32(std::vector<uint8_t> & bytes, uint32_t value) {
    const size_t offset = bytes.size();
    bytes.resize(offset + sizeof(value));
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

static bool pack_kernel_constants(const KernelDefinition & definition,
                                  const Dispatch &         dispatch,
                                  std::vector<uint8_t> &   constants) {
    constants.clear();
    for (const KernelScalarDefinition & parameter : definition.launch_parameters) {
        const char * name = parameter.name != nullptr ? parameter.name : "";
        const char * type = parameter.type != nullptr ? parameter.type : "";
        const auto   item = dispatch.kernel.integer_parameters.find(name);
        if (item == dispatch.kernel.integer_parameters.end() || std::strcmp(type, "index") != 0 || item->second < 0 ||
            static_cast<uint64_t>(item->second) > std::numeric_limits<uint32_t>::max()) {
            constants.clear();
            GGML_LOG_ERROR("%s: invalid launch scalar %s for %s\n", __func__, name,
                           kernel_definition_name(definition).c_str());
            return false;
        }
        append_u32(constants, static_cast<uint32_t>(item->second));
    }
    return true;
}

static std::string kernel_executable_artifact_key(const KernelDefinition & definition,
                                                  const Dispatch &         dispatch,
                                                  const char *             target) {
    std::ostringstream out;
    out << (target != nullptr ? target : "") << '|' << definition.source_digest << '|' << definition.symbol
        << "|recipe=" << definition.compile_recipe.mode;
    for (const KernelScalarDefinition & parameter : definition.workload_parameters) {
        const char * name = parameter.name != nullptr ? parameter.name : "";
        const auto   item = dispatch.kernel.integer_parameters.find(name);
        out << '|' << name << '=';
        if (item == dispatch.kernel.integer_parameters.end()) {
            out << "<missing>";
        } else {
            out << item->second;
        }
    }
    for (const KernelCompileConfig & config : definition.compile_config) {
        out << '|' << (config.key != nullptr ? config.key : "") << '=' << (config.value != nullptr ? config.value : "");
    }
    return out.str();
}

static bool ensure_jit(const KernelExecutablePrepareContext & context) {
    if (context.jit == nullptr) {
        GGML_LOG_ERROR("%s: missing Loom JIT storage\n", __func__);
        return false;
    }
    if (*context.jit != nullptr) {
        return true;
    }
    ggml_hrx_loom_jit_amdgpu_options options = {};
    options.processor                        = context.target;
    options.identifier                       = context.target;
    if (ErrorResult error = take_status(ggml_hrx_loom_jit_amdgpu_create(&options, context.jit))) {
        GGML_LOG_ERROR("%s: create Loom JIT: %s\n", __func__, error->c_str());
        return false;
    }
    return true;
}

}  // namespace

KernelExecutableArtifact::~KernelExecutableArtifact() {
    if (executable != nullptr) {
        hrx_executable_release(executable);
    }
}

std::shared_ptr<KernelExecutableArtifact> KernelExecutableCache::prepare(const KernelExecutablePrepareContext & context,
                                                                         const KernelDefinition & definition,
                                                                         const Dispatch &         dispatch,
                                                                         std::vector<uint8_t> &   constants) {
    if (!pack_kernel_constants(definition, dispatch, constants)) {
        return nullptr;
    }
    const std::string           key = kernel_executable_artifact_key(definition, dispatch, context.target);
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  found = cache_.find(key);
    if (found != cache_.end()) {
        return found->second;
    }
    if (!ensure_jit(context)) {
        return nullptr;
    }
    if (definition.compile_recipe.primary_sources.empty()) {
        GGML_LOG_ERROR("%s: kernel %s has no primary source\n", __func__, kernel_definition_name(definition).c_str());
        return nullptr;
    }
    const KernelSourceRef & primary_source = definition.compile_recipe.primary_sources.front();
    const KernelSource *    source         = primary_source.contents;
    if (source == nullptr) {
        GGML_LOG_ERROR("%s: missing embedded source for %s\n", __func__, primary_source.path);
        return nullptr;
    }
    std::vector<ggml_hrx_loom_jit_source> dependencies;
    dependencies.reserve(definition.compile_recipe.library_sources.size());
    for (const KernelSourceRef & dependency_ref : definition.compile_recipe.library_sources) {
        const KernelSource * dependency = dependency_ref.contents;
        if (dependency == nullptr) {
            GGML_LOG_ERROR("%s: missing embedded dependency for %s\n", __func__, dependency_ref.path);
            return nullptr;
        }
        dependencies.push_back({
            dependency->source.data,
            dependency->source.length,
            to_jit_source_format(dependency->source.format),
            dependency_ref.path,
        });
    }

    std::vector<ggml_hrx_loom_jit_config_binding> configs;
    configs.reserve(definition.compile_config.size());
    for (const KernelCompileConfig & config : definition.compile_config) {
        configs.push_back({ config.key, config.value });
    }
    std::vector<int64_t> workload;
    workload.reserve(definition.workload_parameters.size());
    for (const KernelScalarDefinition & parameter : definition.workload_parameters) {
        const char * name = parameter.name != nullptr ? parameter.name : "";
        const char * type = parameter.type != nullptr ? parameter.type : "";
        const auto   item = dispatch.kernel.integer_parameters.find(name);
        if (item == dispatch.kernel.integer_parameters.end() || std::strcmp(type, "index") != 0) {
            GGML_LOG_ERROR("%s: invalid workload scalar %s for %s\n", __func__, name,
                           kernel_definition_name(definition).c_str());
            return nullptr;
        }
        workload.push_back(item->second);
    }

    ggml_hrx_loom_jit_compile_options compile_options = {};
    compile_options.source_data                       = source->source.data;
    compile_options.source_size                       = source->source.length;
    compile_options.source_format                     = to_jit_source_format(source->source.format);
    compile_options.source_identifier                 = primary_source.path;
    compile_options.root_symbol                       = definition.symbol;
    compile_options.module_name                       = definition.symbol;
    compile_options.artifact_identifier               = definition.symbol;
    compile_options.dependencies                      = dependencies.data();
    compile_options.dependency_count                  = dependencies.size();
    compile_options.config_bindings                   = configs.data();
    compile_options.config_binding_count              = configs.size();
    compile_options.workload_arguments                = workload.data();
    compile_options.workload_argument_count           = workload.size();
    compile_options.evaluate_launch_config            = true;

    ggml_hrx_loom_jit_compile_result compiled;
    if (ErrorResult error = take_status(ggml_hrx_loom_jit_amdgpu_compile(*context.jit, &compile_options, &compiled))) {
        GGML_LOG_ERROR("%s: compile %s: %s\n", __func__, key.c_str(), error->c_str());
        return nullptr;
    }

    auto artifact    = std::make_shared<KernelExecutableArtifact>();
    artifact->launch = compiled.launch_config;
    if (ErrorResult error =
            take_status(hrx_executable_load_data(context.device, compiled.hsaco_data, compiled.hsaco_size, "amdgpu",
                                                 context.target, &artifact->executable))) {
        GGML_LOG_ERROR("%s: load %s: %s\n", __func__, key.c_str(), error->c_str());
        return nullptr;
    }
    if (ErrorResult error = take_status(
            hrx_executable_lookup_export_by_name(artifact->executable, definition.symbol, &artifact->export_ordinal))) {
        GGML_LOG_ERROR("%s: lookup %s: %s\n", __func__, key.c_str(), error->c_str());
        return nullptr;
    }
    if (ErrorResult error = take_status(
            hrx_executable_export_info(artifact->executable, artifact->export_ordinal, &artifact->export_info))) {
        GGML_LOG_ERROR("%s: inspect %s: %s\n", __func__, key.c_str(), error->c_str());
        return nullptr;
    }
    if (artifact->export_info.binding_count != dispatch.bindings.size() ||
        artifact->export_info.constant_byte_length != constants.size() ||
        artifact->export_info.parameter_count != dispatch.bindings.size() + definition.launch_parameters.size()) {
        GGML_LOG_ERROR("%s: compiled ABI does not match manifest for %s\n", __func__, key.c_str());
        return nullptr;
    }
    if (artifact->launch.workgroup_count[0] == 0 || artifact->launch.workgroup_size[0] == 0) {
        GGML_LOG_ERROR("%s: compiled launch geometry is empty for %s\n", __func__, key.c_str());
        return nullptr;
    }
    cache_.emplace(key, artifact);
    return artifact;
}

void KernelExecutableCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
}

}  // namespace ggml::hrx
