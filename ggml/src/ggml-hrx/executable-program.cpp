#include "executable-program.h"
#include "transfer-manager.h"

#include "loom-jit/ggml-hrx-loom-jit.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace ggml::hrx {

struct Artifact {
    ~Artifact() { if (executable != nullptr) hrx_executable_release(executable); }
    hrx_executable_t executable = nullptr;
    uint32_t export_ordinal = 0;
    hrx_executable_export_info_t export_info = {};
    ggml_hrx_loom_jit_launch_config_t launch = {};
    PreparedArtifactDiagnostic diagnostic;
};

struct ExecutableArtifactRepository::Impl {
    std::mutex mutex;
    std::unordered_map<std::string, std::shared_ptr<Artifact>> artifacts;
};

ExecutableArtifactRepository::ExecutableArtifactRepository() : impl_(new Impl()) {}
ExecutableArtifactRepository::~ExecutableArtifactRepository() = default;

namespace {

std::string read_file(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return {};
    return { std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>() };
}

std::string take_status(hrx_status_t status) {
    if (hrx_status_is_ok(status)) return {};
    char * message = nullptr;
    size_t length = 0;
    hrx_status_t format_status = hrx_status_to_string(status, &message, &length);
    if (!hrx_status_is_ok(format_status)) hrx_status_ignore(format_status);
    const std::string result = message != nullptr ? message : "unknown HRX error";
    hrx_status_free_message(message);
    hrx_status_ignore(status);
    return result;
}

const KernelDefinition * find_kernel(const KernelCorpus & corpus, const std::string & id) {
    const auto it = std::find_if(corpus.kernels.begin(), corpus.kernels.end(),
        [&](const KernelDefinition & definition) { return definition.id == id; });
    return it == corpus.kernels.end() ? nullptr : &*it;
}

const TransientAllocation * find_transient(const CommandProgram & commands, uint32_t id) {
    const auto it = std::find_if(commands.transients.allocations.begin(), commands.transients.allocations.end(),
        [&](const TransientAllocation & allocation) { return allocation.id == id; });
    return it == commands.transients.allocations.end() ? nullptr : &*it;
}

void append_u32(std::vector<uint8_t> & bytes, uint32_t value) {
    const size_t offset = bytes.size();
    bytes.resize(offset + sizeof(value));
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

std::string join_key(const std::map<std::string, std::string> & values) {
    std::ostringstream out;
    for (const auto & value : values) out << '|' << value.first << '=' << value.second;
    return out.str();
}

} // namespace

PackedKernelConstants pack_kernel_constants(const KernelDefinition & definition,
                                             const Command & command) {
    PackedKernelConstants result;
    for (const KernelScalarDefinition & parameter : definition.launch_parameters) {
        const auto value = command.scalar_parameters.find(parameter.name);
        if (value == command.scalar_parameters.end()) {
            result.errors.push_back("missing launch scalar " + parameter.name);
            continue;
        }
        if (parameter.type != "index") {
            result.errors.push_back("unsupported launch scalar type " + parameter.type + " for " + parameter.name);
            continue;
        }
        if (value->second < 0 || static_cast<uint64_t>(value->second) > std::numeric_limits<uint32_t>::max()) {
            result.errors.push_back("launch scalar " + parameter.name + " does not fit the index ABI");
            continue;
        }
        append_u32(result.bytes, static_cast<uint32_t>(value->second));
    }
    if (!result.errors.empty()) result.bytes.clear();
    return result;
}

std::string kernel_artifact_key(const KernelDefinition & definition,
                                const Command & command) {
    std::ostringstream out;
    out << definition.target << '|' << definition.source_digest << '|' << definition.symbol
        << "|recipe=" << definition.compile_recipe.mode;
    if (!definition.compile_recipe.link_module.empty()) out << ':' << definition.compile_recipe.link_module;
    for (const KernelScalarDefinition & parameter : definition.workload_parameters) {
        const auto value = command.scalar_parameters.find(parameter.name);
        out << '|' << parameter.name << '=';
        if (value == command.scalar_parameters.end()) out << "<missing>";
        else out << value->second;
    }
    out << join_key(command.compile_parameters);
    return out.str();
}

static const char * binding_class_name(const ExecutableBufferBinding & binding) {
    if (binding.buffer != nullptr) {
        if (binding.weight) return "borrowed_device_weight";
        if (binding.mutable_state) return "device_mutable_state";
        return "device";
    }
    if (binding.weight) return "resident_host_weight";
    if (binding.upload_before_launch && binding.download_after_completion) return "host_input_output";
    if (binding.upload_before_launch) return "host_input";
    if (binding.download_after_completion) return "host_output";
    return "host";
}

struct PreparedExecutableProgram::Impl {
    struct HostStaging {
        StorageId storage = kInvalidId;
        hrx_buffer_t buffer = nullptr;
        void * host_data = nullptr;
        size_t length = 0;
        bool upload = false;
        bool initialize = false;
        bool download = false;
    };

    ~Impl() {
        if (graph_exec != nullptr) hrx_graph_exec_release(graph_exec);
        if (graph != nullptr) hrx_graph_release(graph);
        if (transient_buffer != nullptr) hrx_buffer_release(transient_buffer);
        for (HostStaging & staging : host_staging) {
            if (staging.buffer != nullptr) hrx_buffer_release(staging.buffer);
        }
        for (hrx_buffer_t buffer : retained_buffers) hrx_buffer_release(buffer);
    }
    hrx_graph_t graph = nullptr;
    hrx_graph_exec_t graph_exec = nullptr;
    hrx_buffer_t transient_buffer = nullptr;
    size_t node_count = 0;
    size_t retained_bytes = 0;
    size_t borrowed_device_weight_bytes = 0;
    size_t resident_host_weight_bytes = 0;
    size_t host_staging_bytes = 0;
    size_t transient_bytes = 0;
    AllocationFingerprint allocation_fingerprint;
    hrx_device_t device = nullptr;
    TransferManager * transfers = nullptr;
    std::vector<HostStaging> host_staging;
    std::vector<WeightResidencyLease> resident_weights;
    std::vector<hrx_buffer_t> retained_buffers;
    std::vector<std::shared_ptr<Artifact>> retained_artifacts;
    std::vector<PreparedArtifactDiagnostic> artifacts;
    std::vector<PreparedCommandDiagnostic> commands;
    std::vector<std::string> errors;
    bool launch_in_flight = false;
};

PreparedExecutableProgram::PreparedExecutableProgram() : impl_(new Impl()) {}
PreparedExecutableProgram::~PreparedExecutableProgram() = default;
PreparedExecutableProgram::PreparedExecutableProgram(PreparedExecutableProgram &&) noexcept = default;
PreparedExecutableProgram & PreparedExecutableProgram::operator=(PreparedExecutableProgram &&) noexcept = default;
bool PreparedExecutableProgram::valid() const { return impl_ != nullptr && impl_->errors.empty() && impl_->graph_exec != nullptr; }
size_t PreparedExecutableProgram::node_count() const { return impl_ == nullptr ? 0 : impl_->node_count; }
size_t PreparedExecutableProgram::artifact_count() const { return impl_ == nullptr ? 0 : impl_->artifacts.size(); }
size_t PreparedExecutableProgram::retained_bytes() const { return impl_ == nullptr ? 0 : impl_->retained_bytes; }
size_t PreparedExecutableProgram::borrowed_device_weight_bytes() const { return impl_ == nullptr ? 0 : impl_->borrowed_device_weight_bytes; }
size_t PreparedExecutableProgram::resident_host_weight_bytes() const { return impl_ == nullptr ? 0 : impl_->resident_host_weight_bytes; }
size_t PreparedExecutableProgram::host_staging_bytes() const { return impl_ == nullptr ? 0 : impl_->host_staging_bytes; }
size_t PreparedExecutableProgram::transient_bytes() const { return impl_ == nullptr ? 0 : impl_->transient_bytes; }
const AllocationFingerprint & PreparedExecutableProgram::allocation_fingerprint() const { return impl_->allocation_fingerprint; }
const std::vector<std::string> & PreparedExecutableProgram::errors() const { return impl_->errors; }
const std::vector<PreparedArtifactDiagnostic> & PreparedExecutableProgram::artifacts() const { return impl_->artifacts; }
const std::vector<PreparedCommandDiagnostic> & PreparedExecutableProgram::commands() const { return impl_->commands; }

std::string PreparedExecutableProgram::rebind(const ExecutableBindings & bindings) {
    if (!valid()) return "cannot rebind an invalid prepared executable";
    if (fingerprint_bindings(bindings.snapshot) != impl_->allocation_fingerprint) {
        return "live allocation fingerprint does not match prepared executable";
    }
    for (Impl::HostStaging & staging : impl_->host_staging) {
        const auto found = std::find_if(bindings.storages.begin(), bindings.storages.end(),
            [&](const ExecutableBufferBinding & binding) { return binding.storage == staging.storage; });
        if (found == bindings.storages.end() || found->buffer != nullptr || found->host_data == nullptr ||
            found->offset > found->capacity || found->length > found->capacity - found->offset ||
            found->length != staging.length || found->upload_before_launch != staging.upload ||
            found->download_after_completion != staging.download) {
            return "live host binding does not match prepared storage " + std::to_string(staging.storage);
        }
        staging.host_data = static_cast<uint8_t *>(found->host_data) + found->offset;
    }
    return {};
}

std::string PreparedExecutableProgram::launch(hrx_stream_t stream) {
    if (!valid() || stream == nullptr) return "cannot launch an invalid prepared executable";
    if (impl_->transfers == nullptr) return "prepared executable has no transfer manager";
    const bool has_uploads = std::any_of(impl_->host_staging.begin(), impl_->host_staging.end(),
        [](const Impl::HostStaging & staging) { return staging.upload; });
    if (impl_->launch_in_flight && has_uploads) {
        std::string error = impl_->transfers->wait_for_producer(stream);
        if (!error.empty()) return "order reusable launch bindings: " + error;
    }
    for (const Impl::HostStaging & staging : impl_->host_staging) {
        if (!staging.upload) continue;
        std::string error = impl_->transfers->upload(
            staging.host_data, staging.buffer, 0, staging.length);
        if (!error.empty()) return "upload host staging: " + error;
    }
    std::string error = impl_->transfers->join(stream);
    if (!error.empty()) return "join executable uploads: " + error;
    error = take_status(hrx_graph_exec_launch(impl_->graph_exec, stream));
    if (error.empty()) impl_->launch_in_flight = true;
    return error;
}

std::string PreparedExecutableProgram::complete_after_synchronize() {
    if (!valid()) return "cannot complete an invalid prepared executable";
    if (impl_->transfers == nullptr) return "prepared executable has no transfer manager";
    for (const Impl::HostStaging & staging : impl_->host_staging) {
        if (!staging.download) continue;
        std::string error = impl_->transfers->download(
            nullptr, staging.buffer, 0, staging.host_data, staging.length);
        if (!error.empty()) return "download host staging: " + error;
    }
    impl_->launch_in_flight = false;
    return {};
}

PreparedExecutableProgram prepare_executable_program(
    hrx_device_t device, hrx_stream_t stream, TransferManager & transfers,
    WeightResidencyCache & weights,
    ExecutableArtifactRepository & artifact_repository,
    const ProgramPlan & plan, const KernelCorpus & corpus, const CommandProgram & commands,
    const ExecutableBindings & bindings,
    const ExecutablePreparationOptions & options) {
    PreparedExecutableProgram result;
    auto & impl = *result.impl_;
    if (device == nullptr || stream == nullptr) impl.errors.push_back("device and stream are required");
    if (!transfers.valid()) impl.errors.push_back("valid transfer manager is required: " + transfers.initialization_error());
    if (!weights.valid()) impl.errors.push_back("valid weight residency cache is required: " + weights.initialization_error());
    if (!commands.valid()) impl.errors.insert(impl.errors.end(), commands.errors.begin(), commands.errors.end());
    if (commands.target != options.target) impl.errors.push_back("preparation target does not match command program");
    const VerificationResult verification = verify_command_program(plan, corpus, commands);
    impl.errors.insert(impl.errors.end(), verification.errors.begin(), verification.errors.end());
    const VerificationResult binding_verification = verify_binding_snapshot(plan, bindings.snapshot);
    impl.errors.insert(impl.errors.end(), binding_verification.errors.begin(), binding_verification.errors.end());
    if (bindings.storages.size() != bindings.snapshot.bindings.size()) {
        impl.errors.push_back("executable binding table does not match binding snapshot");
    }
    if (!impl.errors.empty()) return result;

    impl.device = device;
    impl.transfers = &transfers;
    impl.allocation_fingerprint = fingerprint_bindings(bindings.snapshot);
    std::string error;
    if (commands.transients.arena_size != 0) {
        error = take_status(hrx_buffer_allocate(stream, commands.transients.arena_size, HRX_MEMORY_TYPE_DEVICE_LOCAL,
            HRX_BUFFER_USAGE_DEFAULT, &impl.transient_buffer));
        if (!error.empty()) { impl.errors.push_back("allocate transient arena: " + error); return result; }
    }
    impl.transient_bytes = commands.transients.arena_size;
    for (const ConstantInitialization & initialization : commands.initializations) {
        const auto allocation = std::find_if(commands.transients.allocations.begin(), commands.transients.allocations.end(),
            [&](const TransientAllocation & item) { return item.storage == initialization.storage; });
        if (allocation == commands.transients.allocations.end() || impl.transient_buffer == nullptr) {
            impl.errors.push_back("constant initialization does not resolve to transient storage");
            return result;
        }
        error = transfers.upload(initialization.data.data(), impl.transient_buffer,
            allocation->arena_offset, initialization.data.size());
        if (!error.empty()) { impl.errors.push_back("upload " + initialization.label + ": " + error); return result; }
    }

    ggml_hrx_loom_jit_amdgpu_options_t jit_options = {};
    jit_options.structure_size = sizeof(jit_options);
    jit_options.processor = options.target.c_str();
    jit_options.identifier = options.target.c_str();
    ggml_hrx_loom_jit_amdgpu_t jit = nullptr;
    error = take_status(ggml_hrx_loom_jit_amdgpu_create(&jit_options, &jit));
    if (!error.empty()) { impl.errors.push_back("create Loom JIT: " + error); return result; }

    // Compilation and executable publication are serialized per device. This is
    // startup work; steady execution only queries the retained repository.
    std::unique_lock<std::mutex> artifact_lock(artifact_repository.impl_->mutex);
    auto & artifact_cache = artifact_repository.impl_->artifacts;
    std::vector<std::shared_ptr<Artifact>> command_artifacts(commands.commands.size());
    std::vector<std::vector<uint8_t>> command_constants(commands.commands.size());
    std::set<std::string> program_artifact_keys;
    for (const Command & command : commands.commands) {
        if (command.kind != CommandKind::Kernel) continue;
        const KernelDefinition * definition = find_kernel(corpus, command.kernel_id);
        if (definition == nullptr) { impl.errors.push_back("missing kernel " + command.kernel_id); break; }
        const PackedKernelConstants constants = pack_kernel_constants(*definition, command);
        if (!constants.valid()) {
            for (const std::string & item : constants.errors) impl.errors.push_back(command.kernel_id + ": " + item);
            break;
        }
        command_constants[command.ordinal] = constants.bytes;
        const std::string key = kernel_artifact_key(*definition, command);
        auto found = artifact_cache.find(key);
        if (found != artifact_cache.end()) {
            command_artifacts[command.ordinal] = found->second;
            if (program_artifact_keys.insert(key).second) {
                impl.retained_artifacts.push_back(found->second);
                impl.artifacts.push_back(found->second->diagnostic);
            }
            continue;
        }

        const std::filesystem::path source_path = std::filesystem::path(options.corpus_directory) /
            definition->compile_recipe.primary_sources.front();
        const std::string source = read_file(source_path);
        if (source.empty()) { impl.errors.push_back("cannot read Loom source " + source_path.string()); break; }
        std::vector<std::string> dependency_text;
        std::vector<std::string> dependency_paths;
        std::vector<ggml_hrx_loom_jit_source_t> dependencies;
        dependency_text.reserve(definition->compile_recipe.library_sources.size());
        dependency_paths.reserve(definition->compile_recipe.library_sources.size());
        dependencies.reserve(definition->compile_recipe.library_sources.size());
        for (const std::string & relative : definition->compile_recipe.library_sources) {
            dependency_paths.push_back((std::filesystem::path(options.corpus_directory) / relative).string());
            dependency_text.push_back(read_file(dependency_paths.back()));
            if (dependency_text.back().empty()) { impl.errors.push_back("cannot read Loom dependency " + dependency_paths.back()); break; }
        }
        if (!impl.errors.empty()) break;
        for (size_t i = 0; i < dependency_text.size(); ++i) dependencies.push_back({
            dependency_text[i].data(), dependency_text[i].size(), GGML_HRX_LOOM_JIT_SOURCE_FORMAT_TEXT,
            dependency_paths[i].c_str() });
        std::vector<std::string> config_keys;
        std::vector<std::string> config_values;
        std::vector<ggml_hrx_loom_jit_config_binding_t> configs;
        for (const auto & item : command.compile_parameters) {
            config_keys.push_back(item.first);
            config_values.push_back(item.second);
        }
        for (size_t i = 0; i < config_keys.size(); ++i) configs.push_back({ config_keys[i].c_str(), config_values[i].c_str() });
        std::vector<int64_t> workload;
        for (const KernelScalarDefinition & parameter : definition->workload_parameters) {
            const auto value = command.scalar_parameters.find(parameter.name);
            if (value == command.scalar_parameters.end() || parameter.type != "index") {
                impl.errors.push_back("invalid workload ABI for " + command.kernel_id + " parameter " + parameter.name);
                break;
            }
            workload.push_back(value->second);
        }
        if (!impl.errors.empty()) break;
        ggml_hrx_loom_jit_compile_options_t compile_options = {};
        compile_options.structure_size = sizeof(compile_options);
        compile_options.source_data = source.data();
        compile_options.source_size = source.size();
        compile_options.source_format = GGML_HRX_LOOM_JIT_SOURCE_FORMAT_TEXT;
        compile_options.source_identifier = source_path.c_str();
        compile_options.root_symbol = definition->symbol.c_str();
        compile_options.module_name = definition->symbol.c_str();
        compile_options.artifact_identifier = definition->symbol.c_str();
        compile_options.dependencies = dependencies.data();
        compile_options.dependency_count = dependencies.size();
        compile_options.config_bindings = configs.data();
        compile_options.config_binding_count = configs.size();
        compile_options.workload_arguments = workload.data();
        compile_options.workload_argument_count = workload.size();
        compile_options.evaluate_launch_config = true;
        ggml_hrx_loom_jit_compile_result_t compiled = {};
        error = take_status(ggml_hrx_loom_jit_amdgpu_compile(jit, &compile_options, &compiled));
        if (!error.empty()) { impl.errors.push_back("compile " + key + ": " + error); break; }

        auto artifact = std::make_shared<Artifact>();
        artifact->launch = compiled.launch_config;
        if (compiled.manifest_json != nullptr) {
            artifact->diagnostic.manifest_json.assign(compiled.manifest_json, compiled.manifest_json_size);
        }
        if (compiled.compile_report_json != nullptr) {
            artifact->diagnostic.compile_report_json.assign(
                compiled.compile_report_json, compiled.compile_report_json_size);
        }
        if (compiled.final_module_text != nullptr) {
            artifact->diagnostic.final_module_text.assign(
                compiled.final_module_text, compiled.final_module_text_size);
        }
        error = take_status(hrx_executable_load_data(device, compiled.hsaco_data, compiled.hsaco_size,
            "amdgpu", options.target.c_str(), &artifact->executable));
        ggml_hrx_loom_jit_compile_result_deinitialize(&compiled);
        if (!error.empty()) { impl.errors.push_back("load " + key + ": " + error); break; }
        error = take_status(hrx_executable_lookup_export_by_name(artifact->executable, definition->symbol.c_str(),
            &artifact->export_ordinal));
        if (error.empty()) error = take_status(hrx_executable_export_info(artifact->executable,
            artifact->export_ordinal, &artifact->export_info));
        if (!error.empty()) { impl.errors.push_back("inspect " + key + ": " + error); break; }
        if (artifact->export_info.binding_count != definition->bindings.size() ||
            artifact->export_info.constant_byte_length != constants.bytes.size() ||
            artifact->export_info.parameter_count != definition->bindings.size() + definition->launch_parameters.size()) {
            impl.errors.push_back("compiled ABI does not match manifest for " + key +
                ": bindings=" + std::to_string(artifact->export_info.binding_count) + "/" +
                std::to_string(definition->bindings.size()) + " constants=" +
                std::to_string(artifact->export_info.constant_byte_length) + "/" +
                std::to_string(constants.bytes.size()) + " parameters=" +
                std::to_string(artifact->export_info.parameter_count) + "/" +
                std::to_string(definition->bindings.size() + definition->launch_parameters.size()));
            break;
        }
        if (artifact->launch.workgroup_count[0] == 0 || artifact->launch.workgroup_count[1] == 0 ||
            artifact->launch.workgroup_count[2] == 0 || artifact->launch.workgroup_size[0] == 0 ||
            artifact->launch.workgroup_size[1] == 0 || artifact->launch.workgroup_size[2] == 0) {
            impl.errors.push_back("compiled launch geometry is empty for " + key);
            break;
        }
        if (artifact->launch.workgroup_storage_bytes != 0) {
            impl.errors.push_back("HRX graph ABI cannot encode dynamic workgroup storage for " + key);
            break;
        }
        artifact->diagnostic.key = key;
        artifact->diagnostic.kernel_id = definition->id;
        for (size_t i = 0; i < 3; ++i) {
            artifact->diagnostic.workgroup_count[i] = artifact->launch.workgroup_count[i];
            artifact->diagnostic.workgroup_size[i] = artifact->launch.workgroup_size[i];
        }
        artifact->diagnostic.subgroup_size = artifact->launch.subgroup_size;
        artifact->diagnostic.constant_bytes = constants.bytes.size();
        artifact->diagnostic.binding_count = definition->bindings.size();
        artifact_cache.emplace(key, artifact);
        if (program_artifact_keys.insert(key).second) {
            impl.retained_artifacts.push_back(artifact);
            impl.artifacts.push_back(artifact->diagnostic);
        }
        command_artifacts[command.ordinal] = std::move(artifact);
    }
    ggml_hrx_loom_jit_amdgpu_release(jit);
    artifact_lock.unlock();
    if (!impl.errors.empty()) return result;

    error = take_status(hrx_graph_create(device, 0, &impl.graph));
    if (!error.empty()) { impl.errors.push_back("create HRX graph: " + error); return result; }

    std::unordered_map<StorageId, hrx_buffer_ref_t> storage_refs;
    std::set<hrx_buffer_t> retained;
    for (const ExecutableBufferBinding & binding : bindings.storages) {
        const auto snapshot_it = std::find_if(bindings.snapshot.bindings.begin(), bindings.snapshot.bindings.end(),
            [&](const ConcreteBinding & item) { return item.storage == binding.storage; });
        if (snapshot_it == bindings.snapshot.bindings.end() ||
            snapshot_it->buffer_identity != binding.buffer_identity ||
            snapshot_it->generation != binding.generation ||
            snapshot_it->capacity != binding.capacity || snapshot_it->offset != binding.offset ||
            snapshot_it->length != binding.length || binding.length == 0 ||
            binding.offset > binding.capacity || binding.length > binding.capacity - binding.offset ||
            !storage_refs.emplace(binding.storage, hrx_buffer_ref_t {}).second) {
            impl.errors.push_back("concrete binding table disagrees with snapshot for storage " +
                                  std::to_string(binding.storage));
            return result;
        }
        hrx_buffer_t concrete_buffer = binding.buffer;
        size_t concrete_offset = binding.offset;
        const ResourceContract & resource = plan.resources.resources[binding.storage];
        if (concrete_buffer == nullptr) {
            if (binding.host_data == nullptr) {
                impl.errors.push_back("storage " + std::to_string(binding.storage) + " has no device or host allocation");
                return result;
            }
            if (resource.weight) {
                WeightSource source;
                source.host_data = binding.host_data;
                source.buffer_identity = binding.buffer_identity;
                source.generation = binding.generation;
                source.capacity = binding.capacity;
                source.offset = binding.offset;
                source.length = binding.length;
                source.layout = binding.layout;
                WeightResidencyResult resident = weights.acquire(stream, transfers, source);
                if (!resident.valid()) {
                    impl.errors.push_back("materialize host weight storage " + std::to_string(binding.storage) + ": " + resident.error);
                    return result;
                }
                concrete_buffer = resident.lease.buffer();
                concrete_offset = 0;
                impl.resident_host_weight_bytes += binding.length;
                impl.resident_weights.push_back(std::move(resident.lease));
            } else {
                PreparedExecutableProgram::Impl::HostStaging staging;
                staging.storage = binding.storage;
                staging.host_data = static_cast<uint8_t *>(binding.host_data) + binding.offset;
                staging.length = binding.length;
                staging.upload = binding.upload_before_launch;
                staging.initialize = binding.initialize_from_host;
                staging.download = binding.download_after_completion;
                error = take_status(hrx_buffer_allocate(stream, binding.length, HRX_MEMORY_TYPE_DEVICE_LOCAL,
                    HRX_BUFFER_USAGE_DEFAULT, &staging.buffer));
                if (!error.empty()) {
                    impl.errors.push_back("allocate host staging for storage " + std::to_string(binding.storage) + ": " + error);
                    return result;
                }
                concrete_buffer = staging.buffer;
                concrete_offset = 0;
                impl.host_staging_bytes += binding.length;
                impl.retained_bytes += binding.length;
                impl.host_staging.push_back(staging);
            }
        } else if (retained.insert(concrete_buffer).second) {
            hrx_buffer_retain(concrete_buffer);
            impl.retained_buffers.push_back(concrete_buffer);
        }
        if (concrete_buffer != nullptr && binding.buffer != nullptr && resource.weight) {
            impl.borrowed_device_weight_bytes += binding.length;
        }
        storage_refs[binding.storage] = { concrete_buffer, concrete_offset, binding.length };
    }
    impl.retained_bytes += commands.transients.arena_size;
    for (const PreparedExecutableProgram::Impl::HostStaging & staging : impl.host_staging) {
        if (!staging.initialize) continue;
        error = transfers.upload(staging.host_data, staging.buffer, 0, staging.length);
        if (!error.empty()) {
            impl.errors.push_back("initialize retained host staging: " + error);
            return result;
        }
    }
    error = transfers.join(stream);
    if (!error.empty()) {
        impl.errors.push_back("join executable initialization transfers: " + error);
        return result;
    }

    auto resolve_binding = [&](const CommandBinding & binding, hrx_buffer_ref_t & result_ref) -> bool {
        if (binding.origin == BindingOrigin::Transient) {
            const TransientAllocation * allocation = find_transient(commands, binding.transient);
            if (allocation == nullptr || impl.transient_buffer == nullptr ||
                binding.offset > allocation->size || binding.length > allocation->size - binding.offset) {
                impl.errors.push_back("invalid concrete transient binding");
                return false;
            }
            result_ref = { impl.transient_buffer, allocation->arena_offset + binding.offset, binding.length };
        } else {
            const auto concrete = storage_refs.find(binding.storage);
            if (concrete == storage_refs.end() || binding.offset > concrete->second.length ||
                binding.length > concrete->second.length - binding.offset) {
                impl.errors.push_back("graph storage " + std::to_string(binding.storage) + " has no valid concrete range");
                return false;
            }
            result_ref = { concrete->second.buffer, concrete->second.offset + binding.offset, binding.length };
        }
        return true;
    };

    std::vector<hrx_graph_node_t> nodes(commands.commands.size(), nullptr);
    for (const Command & command : commands.commands) {
        std::vector<hrx_graph_node_t> deps;
        for (uint32_t dependency : command.dependencies) {
            if (dependency >= nodes.size() || nodes[dependency] == nullptr) {
                impl.errors.push_back("command dependency was not recorded");
                return result;
            }
            deps.push_back(nodes[dependency]);
        }
        PreparedCommandDiagnostic diagnostic;
        diagnostic.ordinal = command.ordinal;
        diagnostic.kind = command.kind;
        diagnostic.label = command.label;
        diagnostic.binding_count = command.bindings.size();
        if (command.kind == CommandKind::Kernel) {
            const std::shared_ptr<Artifact> & artifact = command_artifacts[command.ordinal];
            std::vector<hrx_buffer_ref_t> bindings;
            for (const CommandBinding & binding : command.bindings) {
                hrx_buffer_ref_t concrete = {};
                if (!resolve_binding(binding, concrete)) return result;
                bindings.push_back(concrete);
            }
            const auto & constants = command_constants[command.ordinal];
            const hrx_graph_kernel_node_attrs_t attrs = {
                artifact->executable, artifact->export_ordinal,
                { { artifact->launch.workgroup_count[0], artifact->launch.workgroup_count[1], artifact->launch.workgroup_count[2] },
                  { artifact->launch.workgroup_size[0], artifact->launch.workgroup_size[1], artifact->launch.workgroup_size[2] },
                  artifact->launch.subgroup_size },
                constants.data(), constants.size(), bindings.data(), bindings.size(), 0,
            };
            error = take_status(hrx_graph_add_kernel_node(impl.graph, deps.data(), deps.size(), &attrs,
                &nodes[command.ordinal]));
            diagnostic.artifact_key = artifact->diagnostic.key;
            diagnostic.constant_bytes = constants.size();
        } else if (command.kind == CommandKind::Copy) {
            hrx_graph_copy_buffer_node_attrs_t attrs = {};
            if (!resolve_binding(command.bindings[0], attrs.src) ||
                !resolve_binding(command.bindings[1], attrs.dst)) return result;
            error = take_status(hrx_graph_add_copy_buffer_node(impl.graph, deps.data(), deps.size(), &attrs,
                &nodes[command.ordinal]));
        } else if (command.kind == CommandKind::Fill) {
            const uint32_t fill_byte = static_cast<uint32_t>(command.scalar_parameters.at("fill_byte")) & 0xffu;
            hrx_graph_fill_buffer_node_attrs_t attrs = {};
            if (!resolve_binding(command.bindings[0], attrs.dst)) return result;
            attrs.pattern = fill_byte;
            attrs.pattern_size = 1;
            error = take_status(hrx_graph_add_fill_buffer_node(impl.graph, deps.data(), deps.size(), &attrs,
                &nodes[command.ordinal]));
        } else {
            error = take_status(hrx_graph_add_empty_node(impl.graph, deps.data(), deps.size(), &nodes[command.ordinal]));
        }
        if (!error.empty()) { impl.errors.push_back("record command " + std::to_string(command.ordinal) + ": " + error); return result; }
        impl.commands.push_back(std::move(diagnostic));
    }
    error = take_status(hrx_graph_size(impl.graph, &impl.node_count));
    if (!error.empty()) { impl.errors.push_back("query HRX graph size: " + error); return result; }
    if (impl.node_count != commands.commands.size()) {
        impl.errors.push_back("recorded HRX graph node count does not match command program");
        return result;
    }
    error = take_status(hrx_graph_instantiate(impl.graph, 0, &impl.graph_exec));
    if (!error.empty()) impl.errors.push_back("instantiate HRX graph: " + error);
    return result;
}

std::string format_prepared_executable_program(const PreparedExecutableProgram & program) {
    std::ostringstream out;
    out << "prepared executable program\n"
        << "valid=" << (program.valid() ? "true" : "false") << '\n'
        << "artifacts=" << program.artifact_count() << '\n'
        << "nodes=" << program.node_count() << '\n'
        << "retained_bytes=" << program.retained_bytes() << '\n'
        << "borrowed_device_weight_bytes=" << program.borrowed_device_weight_bytes() << '\n'
        << "resident_host_weight_bytes=" << program.resident_host_weight_bytes() << '\n'
        << "host_staging_bytes=" << program.host_staging_bytes() << '\n'
        << "transient_bytes=" << program.transient_bytes() << '\n'
        << "allocation_fingerprint=" << program.allocation_fingerprint().value << '\n';
    for (const PreparedArtifactDiagnostic & artifact : program.artifacts()) {
        out << "artifact " << artifact.kernel_id << " key=" << artifact.key
            << " workgroups=" << artifact.workgroup_count[0] << ',' << artifact.workgroup_count[1] << ',' << artifact.workgroup_count[2]
            << " workgroup_size=" << artifact.workgroup_size[0] << ',' << artifact.workgroup_size[1] << ',' << artifact.workgroup_size[2]
            << " subgroup=" << artifact.subgroup_size << " constants=" << artifact.constant_bytes
            << " bindings=" << artifact.binding_count << '\n';
    }
    for (const PreparedCommandDiagnostic & command : program.commands()) {
        out << "command " << command.ordinal << ' ' << command_kind_name(command.kind)
            << " label=" << command.label << " constants=" << command.constant_bytes
            << " bindings=" << command.binding_count;
        if (!command.artifact_key.empty()) out << " artifact=" << command.artifact_key;
        out << '\n';
    }
    for (const std::string & error : program.errors()) out << "error: " << error << '\n';
    return out.str();
}

std::string serialize_prepared_executable_program_json(const PreparedExecutableProgram & program) {
    nlohmann::json root = {
        { "schema", "ggml-hrx-prepared-executable-v1" },
        { "valid", program.valid() }, { "node_count", program.node_count() },
        { "artifact_count", program.artifact_count() }, { "retained_bytes", program.retained_bytes() },
        { "borrowed_device_weight_bytes", program.borrowed_device_weight_bytes() },
        { "resident_host_weight_bytes", program.resident_host_weight_bytes() },
        { "host_staging_bytes", program.host_staging_bytes() },
        { "transient_bytes", program.transient_bytes() },
        { "allocation_fingerprint", program.allocation_fingerprint().value }, { "errors", program.errors() },
    };
    root["artifacts"] = nlohmann::json::array();
    for (const PreparedArtifactDiagnostic & artifact : program.artifacts()) root["artifacts"].push_back({
        { "key", artifact.key }, { "kernel", artifact.kernel_id },
        { "workgroup_count", artifact.workgroup_count }, { "workgroup_size", artifact.workgroup_size },
        { "subgroup_size", artifact.subgroup_size }, { "constant_bytes", artifact.constant_bytes },
        { "binding_count", artifact.binding_count },
    });
    root["commands"] = nlohmann::json::array();
    for (const PreparedCommandDiagnostic & command : program.commands()) root["commands"].push_back({
        { "ordinal", command.ordinal }, { "kind", command_kind_name(command.kind) }, { "label", command.label },
        { "artifact", command.artifact_key }, { "constant_bytes", command.constant_bytes },
        { "binding_count", command.binding_count },
    });
    return root.dump(2);
}

std::string format_executable_bindings(const ExecutableBindings & bindings) {
    std::ostringstream out;
    out << "executable bindings count=" << bindings.storages.size() << '\n';
    for (const ExecutableBufferBinding & binding : bindings.storages) {
        out << "  storage " << binding.storage << " class=" << binding_class_name(binding)
            << " capacity=" << binding.capacity << " range=" << binding.offset << '+' << binding.length;
        if (binding.weight) out << " layout=" << binding.layout;
        out << '\n';
    }
    return out.str();
}

std::string serialize_executable_bindings_json(const ExecutableBindings & bindings) {
    nlohmann::ordered_json root = {
        { "schema", "ggml-hrx-executable-bindings-v1" }, { "bindings", nlohmann::ordered_json::array() },
    };
    for (const ExecutableBufferBinding & binding : bindings.storages) {
        root["bindings"].push_back({
            { "storage", binding.storage }, { "class", binding_class_name(binding) },
            { "capacity", binding.capacity }, { "offset", binding.offset }, { "length", binding.length },
            { "layout", binding.weight ? binding.layout : std::string() },
        });
    }
    return root.dump(2);
}

} // namespace ggml::hrx
