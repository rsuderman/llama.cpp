#include "executable-program.h"

#include "loom-jit/ggml-hrx-loom-jit.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace ggml::hrx {
namespace {

struct Artifact {
    ~Artifact() { if (executable != nullptr) hrx_executable_release(executable); }
    hrx_executable_t executable = nullptr;
    uint32_t export_ordinal = 0;
    hrx_executable_export_info_t export_info = {};
    ggml_hrx_loom_jit_launch_config_t launch = {};
    PreparedArtifactDiagnostic diagnostic;
};

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

struct PreparedExecutableProgram::Impl {
    ~Impl() {
        if (graph_exec != nullptr) hrx_graph_exec_release(graph_exec);
        if (graph != nullptr) hrx_graph_release(graph);
        if (transient_buffer != nullptr) hrx_buffer_release(transient_buffer);
        if (dummy_buffer != nullptr) hrx_buffer_release(dummy_buffer);
    }
    hrx_graph_t graph = nullptr;
    hrx_graph_exec_t graph_exec = nullptr;
    hrx_buffer_t transient_buffer = nullptr;
    hrx_buffer_t dummy_buffer = nullptr;
    size_t node_count = 0;
    std::vector<std::shared_ptr<Artifact>> retained_artifacts;
    std::vector<PreparedArtifactDiagnostic> artifacts;
    std::vector<PreparedCommandDiagnostic> commands;
    std::vector<std::string> errors;
};

PreparedExecutableProgram::PreparedExecutableProgram() : impl_(new Impl()) {}
PreparedExecutableProgram::~PreparedExecutableProgram() = default;
PreparedExecutableProgram::PreparedExecutableProgram(PreparedExecutableProgram &&) noexcept = default;
PreparedExecutableProgram & PreparedExecutableProgram::operator=(PreparedExecutableProgram &&) noexcept = default;
bool PreparedExecutableProgram::valid() const { return impl_ != nullptr && impl_->errors.empty() && impl_->graph_exec != nullptr; }
size_t PreparedExecutableProgram::node_count() const { return impl_ == nullptr ? 0 : impl_->node_count; }
size_t PreparedExecutableProgram::artifact_count() const { return impl_ == nullptr ? 0 : impl_->artifacts.size(); }
const std::vector<std::string> & PreparedExecutableProgram::errors() const { return impl_->errors; }
const std::vector<PreparedArtifactDiagnostic> & PreparedExecutableProgram::artifacts() const { return impl_->artifacts; }
const std::vector<PreparedCommandDiagnostic> & PreparedExecutableProgram::commands() const { return impl_->commands; }

PreparedExecutableProgram prepare_executable_program(
    hrx_device_t device, hrx_stream_t stream, const ProgramPlan & plan,
    const KernelCorpus & corpus, const CommandProgram & commands,
    const ExecutablePreparationOptions & options) {
    PreparedExecutableProgram result;
    auto & impl = *result.impl_;
    if (device == nullptr || stream == nullptr) impl.errors.push_back("device and stream are required");
    if (!commands.valid()) impl.errors.insert(impl.errors.end(), commands.errors.begin(), commands.errors.end());
    if (commands.target != options.target) impl.errors.push_back("preparation target does not match command program");
    const VerificationResult verification = verify_command_program(plan, corpus, commands);
    impl.errors.insert(impl.errors.end(), verification.errors.begin(), verification.errors.end());
    if (!impl.errors.empty()) return result;

    size_t dummy_size = 1;
    for (const Command & command : commands.commands) {
        for (const CommandBinding & binding : command.bindings) {
            if (binding.origin == BindingOrigin::GraphValue) dummy_size = std::max(dummy_size, binding.length);
        }
    }
    if (dummy_size > options.dummy_buffer_limit) {
        impl.errors.push_back("largest imported binding exceeds bounded recorder limit: " + std::to_string(dummy_size));
        return result;
    }
    std::string error = take_status(hrx_buffer_allocate(stream, dummy_size, HRX_MEMORY_TYPE_DEVICE_LOCAL,
        HRX_BUFFER_USAGE_DEFAULT, &impl.dummy_buffer));
    if (!error.empty()) { impl.errors.push_back("allocate dummy import buffer: " + error); return result; }
    if (commands.transients.arena_size != 0) {
        error = take_status(hrx_buffer_allocate(stream, commands.transients.arena_size, HRX_MEMORY_TYPE_DEVICE_LOCAL,
            HRX_BUFFER_USAGE_DEFAULT, &impl.transient_buffer));
        if (!error.empty()) { impl.errors.push_back("allocate transient arena: " + error); return result; }
    }
    for (const ConstantInitialization & initialization : commands.initializations) {
        const auto allocation = std::find_if(commands.transients.allocations.begin(), commands.transients.allocations.end(),
            [&](const TransientAllocation & item) { return item.storage == initialization.storage; });
        if (allocation == commands.transients.allocations.end() || impl.transient_buffer == nullptr) {
            impl.errors.push_back("constant initialization does not resolve to transient storage");
            return result;
        }
        error = take_status(hrx_synchronous_h2d(device, initialization.data.data(), impl.transient_buffer,
            allocation->arena_offset, initialization.data.size()));
        if (!error.empty()) { impl.errors.push_back("upload " + initialization.label + ": " + error); return result; }
    }

    ggml_hrx_loom_jit_amdgpu_options_t jit_options = {};
    jit_options.structure_size = sizeof(jit_options);
    jit_options.processor = options.target.c_str();
    jit_options.identifier = options.target.c_str();
    ggml_hrx_loom_jit_amdgpu_t jit = nullptr;
    error = take_status(ggml_hrx_loom_jit_amdgpu_create(&jit_options, &jit));
    if (!error.empty()) { impl.errors.push_back("create Loom JIT: " + error); return result; }

    std::unordered_map<std::string, std::shared_ptr<Artifact>> artifact_cache;
    std::vector<std::shared_ptr<Artifact>> command_artifacts(commands.commands.size());
    std::vector<std::vector<uint8_t>> command_constants(commands.commands.size());
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
        impl.retained_artifacts.push_back(artifact);
        impl.artifacts.push_back(artifact->diagnostic);
        command_artifacts[command.ordinal] = std::move(artifact);
    }
    ggml_hrx_loom_jit_amdgpu_release(jit);
    if (!impl.errors.empty()) return result;

    error = take_status(hrx_graph_create(device, 0, &impl.graph));
    if (!error.empty()) { impl.errors.push_back("create HRX graph: " + error); return result; }
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
            result_ref = { impl.dummy_buffer, 0, binding.length };
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
        << "nodes=" << program.node_count() << '\n';
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
        { "artifact_count", program.artifact_count() }, { "errors", program.errors() },
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

} // namespace ggml::hrx
