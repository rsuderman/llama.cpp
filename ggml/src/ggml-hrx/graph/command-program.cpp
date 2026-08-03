#include "command-program.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace ggml::hrx {
namespace {

static size_t align_up(size_t value, size_t alignment) {
    return alignment == 0 ? value : (value + alignment - 1) / alignment * alignment;
}

static size_t value_span(const Value & value) {
    const size_t blocks = (static_cast<size_t>(value.access.shape[0]) + ggml_blck_size(value.type) - 1) /
        ggml_blck_size(value.type);
    size_t result = blocks * value.access.strides[0];
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (value.access.shape[i] > 0) result += static_cast<size_t>(value.access.shape[i] - 1) * value.access.strides[i];
    }
    return result;
}

static ValueId find_named_value(const Graph & graph, const std::string & name) {
    const auto position = std::find_if(graph.values.begin(), graph.values.end(), [&](const Value & value) {
        return value.name == name;
    });
    return position == graph.values.end() ? kInvalidId : position->id;
}

static void append_rope_initialization(const ProgramPlan & plan, CommandProgram & program) {
    const ValueId frequencies = find_named_value(plan.graph, "hrx.synthetic.inverse_frequencies");
    if (frequencies == kInvalidId) return;
    const Operation * reference = nullptr;
    for (const Operation & operation : plan.graph.operations) {
        if (operation.op != GGML_OP_ROPE) continue;
        if (operation.inputs.size() != 2) {
            program.errors.push_back("selected attention kernel requires ROPE without a frequency-factor tensor");
            return;
        }
        if (reference == nullptr) reference = &operation;
        else if (operation.raw_params != reference->raw_params) {
            program.errors.push_back("ROPE parameters are not coherent across the owned program");
            return;
        }
    }
    if (reference == nullptr) {
        program.errors.push_back("owned program has no ROPE witness for inverse-frequency initialization");
        return;
    }
    int32_t params[15] = {};
    std::memcpy(params, reference->raw_params.data(), sizeof(params));
    float frequency_base = 0.0f;
    float frequency_scale = 0.0f;
    float extension_factor = 0.0f;
    float attention_factor = 0.0f;
    std::memcpy(&frequency_base, params + 5, sizeof(float));
    std::memcpy(&frequency_scale, params + 6, sizeof(float));
    std::memcpy(&extension_factor, params + 7, sizeof(float));
    std::memcpy(&attention_factor, params + 8, sizeof(float));
    const size_t n_dims = static_cast<size_t>(params[1]);
    if (params[2] != GGML_ROPE_TYPE_NEOX || n_dims == 0 || n_dims % 2 != 0 ||
        !std::isfinite(frequency_base) || frequency_base <= 0.0f ||
        !std::isfinite(frequency_scale) || frequency_scale <= 0.0f ||
        extension_factor != 0.0f || attention_factor != 1.0f) {
        program.errors.push_back("ROPE parameters are outside the selected NEOX inverse-frequency contract");
        return;
    }
    const Value & destination = plan.graph.values[frequencies];
    if (plan.graph.storages[destination.access.storage].size != n_dims / 2 * sizeof(float)) {
        program.errors.push_back("inverse-frequency storage does not agree with recovered ROPE dimensions");
        return;
    }
    ConstantInitialization initialization;
    initialization.label = "rope.inverse_frequencies";
    initialization.storage = destination.access.storage;
    initialization.data.resize(n_dims / 2 * sizeof(float));
    const float theta_scale = std::pow(frequency_base, -2.0f / static_cast<float>(n_dims));
    float theta = frequency_scale;
    for (size_t i = 0; i < n_dims / 2; ++i) {
        std::memcpy(initialization.data.data() + i * sizeof(float), &theta, sizeof(float));
        theta *= theta_scale;
    }
    program.initializations.push_back(std::move(initialization));
}

static std::vector<Command> expand_synthetic_commands(const std::vector<Command> & kernels,
                                                       std::vector<std::string> & errors) {
    std::vector<Command> result;
    std::vector<uint32_t> remap(kernels.size(), UINT32_MAX);
    auto mapped_dependencies = [&](const Command & command) {
        std::vector<uint32_t> dependencies;
        for (uint32_t dependency : command.dependencies) {
            if (dependency >= remap.size() || remap[dependency] == UINT32_MAX) {
                errors.push_back("cannot remap a forward synthetic command dependency");
            } else {
                dependencies.push_back(remap[dependency]);
            }
        }
        return dependencies;
    };
    for (const Command & original : kernels) {
        std::vector<uint32_t> dependencies = mapped_dependencies(original);
        uint32_t synthetic = UINT32_MAX;
        if (original.kernel_id == "qwen_attention_metadata_bringup_workaround" && original.bindings.size() >= 2) {
            Command copy;
            copy.ordinal = static_cast<uint32_t>(result.size());
            copy.kind = CommandKind::Copy;
            copy.label = original.label + ".capture_context_base";
            copy.dependencies = dependencies;
            CommandBinding source = original.bindings[1];
            source.name = "source";
            source.length = sizeof(int32_t);
            source.access = ResourceAccess::Read;
            CommandBinding destination = original.bindings[0];
            destination.name = "destination";
            destination.length = sizeof(int32_t);
            destination.access = ResourceAccess::Write;
            copy.bindings = { source, destination };
            result.push_back(std::move(copy));
            synthetic = result.back().ordinal;
        } else if (original.kernel_id == "qwen3_moe_flash_attention_decode_split_f32_f16_wmma" &&
                   original.bindings.size() >= 8) {
            Command fill;
            fill.ordinal = static_cast<uint32_t>(result.size());
            fill.kind = CommandKind::Fill;
            fill.label = original.label + ".clear_completion_counter";
            fill.scalar_parameters["fill_byte"] = 0;
            fill.dependencies = dependencies;
            CommandBinding destination = original.bindings[7];
            destination.name = "destination";
            destination.access = ResourceAccess::Write;
            fill.bindings = { destination };
            result.push_back(std::move(fill));
            synthetic = result.back().ordinal;
        }
        Command command = original;
        command.ordinal = static_cast<uint32_t>(result.size());
        command.dependencies = std::move(dependencies);
        if (synthetic != UINT32_MAX) command.dependencies.push_back(synthetic);
        result.push_back(std::move(command));
        remap[original.ordinal] = result.back().ordinal;
    }
    return result;
}

static const KernelDefinition * find_kernel(const KernelCorpus & corpus, const std::string & id) {
    const auto position = std::find_if(corpus.kernels.begin(), corpus.kernels.end(),
        [&](const KernelDefinition & kernel) { return kernel.id == id; });
    return position == corpus.kernels.end() ? nullptr : &*position;
}

static std::string stable_hash(const std::string & text) {
    // FNV-1a is used only as an in-process cache witness. Source provenance
    // manifests retain their cryptographic SHA-256 digests separately.
    uint64_t hash = UINT64_C(1469598103934665603);
    for (unsigned char byte : text) {
        hash ^= byte;
        hash *= UINT64_C(1099511628211);
    }
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << hash;
    return out.str();
}

static void pack_transient_plan(TransientPlan & result) {
    result.arena_size = 0;
    for (TransientAllocation & allocation : result.allocations) allocation.arena_offset = 0;
    std::vector<size_t> order(result.allocations.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
        const TransientAllocation & a = result.allocations[lhs];
        const TransientAllocation & b = result.allocations[rhs];
        if (a.first_command != b.first_command) return a.first_command < b.first_command;
        if (a.size != b.size) return a.size > b.size;
        return a.storage < b.storage;
    });
    std::vector<bool> is_placed(result.allocations.size(), false);
    for (size_t index : order) {
        TransientAllocation & allocation = result.allocations[index];
        size_t candidate = 0;
        for (;;) {
            candidate = align_up(candidate, allocation.alignment);
            bool conflict = false;
            size_t next_candidate = candidate;
            for (size_t placed_index = 0; placed_index < result.allocations.size(); ++placed_index) {
                if (!is_placed[placed_index]) continue;
                const TransientAllocation & placed = result.allocations[placed_index];
                const bool lifetime_overlap = allocation.first_command <= placed.last_command &&
                    placed.first_command <= allocation.last_command;
                const bool range_overlap = candidate < placed.arena_offset + placed.size &&
                    placed.arena_offset < candidate + allocation.size;
                if (lifetime_overlap && range_overlap) {
                    conflict = true;
                    next_candidate = std::max(next_candidate, placed.arena_offset + placed.size);
                }
            }
            if (!conflict) break;
            candidate = next_candidate;
        }
        allocation.arena_offset = candidate;
        is_placed[index] = true;
        result.arena_size = std::max(result.arena_size, candidate + allocation.size);
    }
    result.arena_size = align_up(result.arena_size, result.arena_alignment);
    std::sort(result.allocations.begin(), result.allocations.end(),
              [](const TransientAllocation & a, const TransientAllocation & b) { return a.id < b.id; });
}

static TransientPlan build_transient_plan(const ProgramPlan & plan,
                                          const std::vector<uint32_t> & invocation_first_command,
                                          const std::vector<uint32_t> & invocation_last_command) {
    TransientPlan result;
    result.arena_alignment = 256;
    for (const ResourceContract & resource : plan.resources.resources) {
        if (!resource.elidable || resource.first_invocation == UINT32_MAX) continue;
        TransientAllocation allocation;
        allocation.id = static_cast<uint32_t>(result.allocations.size());
        allocation.storage = resource.storage;
        allocation.size = resource.size;
        allocation.alignment = 256;
        allocation.first_command = invocation_first_command[resource.first_invocation];
        allocation.last_command = invocation_last_command[resource.last_invocation];
        result.allocations.push_back(allocation);
    }
    pack_transient_plan(result);
    return result;
}

static nlohmann::ordered_json binding_json(const CommandBinding & binding) {
    return {
        { "name", binding.name },
        { "origin", binding.origin == BindingOrigin::GraphValue ? "graph_value" : "transient" },
        { "value", binding.value },
        { "transient", binding.transient },
        { "storage", binding.storage },
        { "offset", binding.offset },
        { "length", binding.length },
        { "access", resource_access_name(binding.access) },
    };
}

} // namespace

const char * command_kind_name(CommandKind kind) {
    switch (kind) {
        case CommandKind::Kernel: return "kernel";
        case CommandKind::Fill: return "fill";
        case CommandKind::Copy: return "copy";
        case CommandKind::Barrier: return "barrier";
    }
    return "unknown";
}

const char * resource_access_name(ResourceAccess access) {
    switch (access) {
        case ResourceAccess::Read: return "read";
        case ResourceAccess::Write: return "write";
        case ResourceAccess::ReadWrite: return "read_write";
    }
    return "unknown";
}

VerificationResult verify_kernel_corpus(const KernelCorpus & corpus) {
    VerificationResult result;
    if (corpus.schema != "ggml-hrx-kernel-corpus-v1") result.errors.push_back("unsupported kernel corpus schema");
    if (corpus.upstream_revision.empty()) result.errors.push_back("kernel corpus has no upstream revision");
    if (corpus.corpus_digest.empty()) result.errors.push_back("kernel corpus has no digest");
    if (corpus.recipe_digest.empty()) result.errors.push_back("kernel corpus has no BUILD.bazel recipe digest");
    if (corpus.plan_case_count == 0) result.errors.push_back("kernel corpus has no compile plan cases");
    std::set<std::string> ids;
    for (const KernelDefinition & kernel : corpus.kernels) {
        if (kernel.id.empty() || kernel.source.empty() || kernel.symbol.empty() || kernel.target.empty() ||
            kernel.source_digest.empty()) result.errors.push_back("kernel definition is incomplete");
        const bool source_is_primary = std::find(kernel.compile_recipe.primary_sources.begin(),
            kernel.compile_recipe.primary_sources.end(), kernel.source) != kernel.compile_recipe.primary_sources.end();
        const bool source_is_library = std::find(kernel.compile_recipe.library_sources.begin(),
            kernel.compile_recipe.library_sources.end(), kernel.source) != kernel.compile_recipe.library_sources.end();
        if ((kernel.compile_recipe.mode != "direct" && kernel.compile_recipe.mode != "archive") ||
            kernel.compile_recipe.primary_sources.empty() || (!source_is_primary && !source_is_library) ||
            (kernel.compile_recipe.mode == "archive" && kernel.compile_recipe.link_module.empty())) {
            result.errors.push_back("kernel " + kernel.id + " has an invalid BUILD compile recipe");
        }
        if (!ids.insert(kernel.id).second) result.errors.push_back("kernel corpus repeats id " + kernel.id);
        std::set<std::string> names;
        for (const KernelBindingDefinition & binding : kernel.bindings) {
            if (binding.name.empty() || !names.insert(binding.name).second) {
                result.errors.push_back("kernel " + kernel.id + " has invalid binding names");
            }
        }
        if (kernel.bindings.size() == 0) result.errors.push_back("kernel " + kernel.id + " has no binding ABI");
    }
    return result;
}

KernelCorpus load_kernel_corpus_manifest(const std::string & path, const std::string & target,
                                         std::vector<std::string> & errors) {
    KernelCorpus corpus;
    try {
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("cannot open corpus manifest");
        const nlohmann::json root = nlohmann::json::parse(input);
        if (root.at("schema") != "ggml-hrx-qwen-kernel-corpus-v1") throw std::runtime_error("unsupported source corpus schema");
        corpus.upstream_revision = root.at("upstream_revision").get<std::string>();
        corpus.corpus_digest = root.at("corpus_sha256").get<std::string>();
        corpus.recipe_digest = root.at("build_bazel_sha256").get<std::string>();
        corpus.plan_case_count = root.at("plan_cases").size();
        std::map<std::string, std::string> digests;
        for (const nlohmann::json & file : root.at("files")) {
            digests[file.at("path").get<std::string>()] = file.at("sha256").get<std::string>();
        }
        for (const nlohmann::json & item : root.at("exports")) {
            KernelDefinition kernel;
            kernel.id = item.at("symbol").get<std::string>();
            kernel.symbol = kernel.id;
            kernel.source = item.at("source").get<std::string>();
            kernel.target = target;
            kernel.source_digest = digests.at(kernel.source);
            kernel.dependencies = item.at("compile_dependencies").get<std::vector<std::string>>();
            const nlohmann::json & recipe = item.at("compile_recipe");
            kernel.compile_recipe.mode = recipe.at("mode").get<std::string>();
            kernel.compile_recipe.link_module = recipe.value("link_module", std::string());
            kernel.compile_recipe.primary_sources = recipe.at("primary_sources").get<std::vector<std::string>>();
            kernel.compile_recipe.library_sources = recipe.at("library_sources").get<std::vector<std::string>>();
            if (kernel.dependencies != kernel.compile_recipe.library_sources) {
                throw std::runtime_error("legacy dependency closure disagrees with compile recipe");
            }
            for (const char * group : { "workload_parameters", "launch_parameters" }) {
                for (const nlohmann::json & parameter : item.at(group)) {
                    const std::string name = parameter.at("name").get<std::string>();
                    if (std::find(kernel.scalar_parameters.begin(), kernel.scalar_parameters.end(), name) ==
                        kernel.scalar_parameters.end()) kernel.scalar_parameters.push_back(name);
                    KernelScalarDefinition definition { name, parameter.at("type").get<std::string>() };
                    if (std::string(group) == "workload_parameters") kernel.workload_parameters.push_back(std::move(definition));
                    else kernel.launch_parameters.push_back(std::move(definition));
                }
            }
            const std::vector<std::string> binding_names = item.at("bindings").get<std::vector<std::string>>();
            const std::vector<std::string> explicit_access = item.value("binding_access", std::vector<std::string>());
            if (explicit_access.size() != binding_names.size()) {
                throw std::runtime_error("kernel binding access metadata has the wrong arity");
            }
            for (size_t binding_index = 0; binding_index < binding_names.size(); ++binding_index) {
                const std::string & name = binding_names[binding_index];
                ResourceAccess access = ResourceAccess::Read;
                if (explicit_access[binding_index] == "write") access = ResourceAccess::Write;
                else if (explicit_access[binding_index] == "read_write") access = ResourceAccess::ReadWrite;
                else if (explicit_access[binding_index] != "read") throw std::runtime_error("invalid kernel binding access metadata");
                kernel.bindings.push_back({ name, access });
            }
            corpus.kernels.push_back(std::move(kernel));
        }
    } catch (const std::exception & error) {
        errors.push_back(std::string("invalid kernel corpus manifest: ") + error.what());
    }
    const VerificationResult verification = verify_kernel_corpus(corpus);
    errors.insert(errors.end(), verification.errors.begin(), verification.errors.end());
    return corpus;
}

CommandProgram build_command_program(const ProgramPlan & plan, const KernelCorpus & corpus) {
    CommandProgram result;
    result.workload = plan.schedule.workload;
    result.target = plan.target;
    result.graph_fingerprint = plan.graph.fingerprint;
    result.recipe_revision = plan.schedule.oracle_revision;
    result.corpus_digest = corpus.corpus_digest;
    result.roots = plan.schedule.roots;
    const VerificationResult corpus_verification = verify_kernel_corpus(corpus);
    result.errors.insert(result.errors.end(), corpus_verification.errors.begin(), corpus_verification.errors.end());

    std::vector<uint32_t> invocation_first(plan.schedule.invocations.size(), UINT32_MAX);
    std::vector<uint32_t> invocation_last(plan.schedule.invocations.size(), 0);
    uint32_t ordinal = 0;
    for (size_t invocation_id = 0; invocation_id < plan.schedule.invocations.size(); ++invocation_id) {
        const Invocation & invocation = plan.schedule.invocations[invocation_id];
        invocation_first[invocation_id] = ordinal;
        for (const Dispatch & dispatch : invocation.dispatches) {
            Command command;
            command.ordinal = ordinal++;
            command.kind = CommandKind::Kernel;
            command.label = invocation.stage + (invocation.layer >= 0 ? "." + std::to_string(invocation.layer) : "");
            command.kernel_id = dispatch.kernel.variant;
            command.scalar_parameters = dispatch.kernel.integer_parameters;
            command.compile_parameters = dispatch.kernel.compile_parameters;
            command.dependencies = dispatch.dependencies;
            const KernelDefinition * definition = find_kernel(corpus, command.kernel_id);
            if (definition == nullptr) {
                result.errors.push_back("no kernel definition for " + command.kernel_id);
            }
            for (size_t binding_index = 0; binding_index < dispatch.bindings.size(); ++binding_index) {
                const TensorBinding & tensor_binding = dispatch.bindings[binding_index];
                if (tensor_binding.value >= plan.graph.values.size()) continue;
                const Value & value = plan.graph.values[tensor_binding.value];
                CommandBinding binding;
                binding.name = tensor_binding.role;
                binding.value = tensor_binding.value;
                binding.storage = value.access.storage;
                binding.offset = value.access.offset + tensor_binding.offset;
                const size_t span = value_span(value);
                const size_t available = tensor_binding.offset < span ? span - tensor_binding.offset : 0;
                binding.length = tensor_binding.length == 0 ? available : tensor_binding.length;
                binding.access = definition != nullptr && binding_index < definition->bindings.size()
                    ? definition->bindings[binding_index].access : ResourceAccess::ReadWrite;
                command.bindings.push_back(std::move(binding));
            }
            result.commands.push_back(std::move(command));
        }
        invocation_last[invocation_id] = ordinal == 0 ? 0 : ordinal - 1;
    }
    result.commands = expand_synthetic_commands(result.commands, result.errors);
    append_rope_initialization(plan, result);
    result.transients = build_transient_plan(plan, invocation_first, invocation_last);
    for (Command & command : result.commands) {
        for (CommandBinding & binding : command.bindings) {
            const auto position = std::find_if(result.transients.allocations.begin(), result.transients.allocations.end(),
                [&](const TransientAllocation & allocation) { return allocation.storage == binding.storage; });
            if (position != result.transients.allocations.end()) {
                binding.origin = BindingOrigin::Transient;
                binding.transient = position->id;
            }
        }
    }
    for (TransientAllocation & allocation : result.transients.allocations) {
        allocation.first_command = UINT32_MAX;
        allocation.last_command = 0;
    }
    for (const Command & command : result.commands) {
        for (const CommandBinding & binding : command.bindings) {
            if (binding.origin != BindingOrigin::Transient || binding.transient >= result.transients.allocations.size()) continue;
            TransientAllocation & allocation = result.transients.allocations[binding.transient];
            allocation.first_command = std::min(allocation.first_command, command.ordinal);
            allocation.last_command = std::max(allocation.last_command, command.ordinal);
        }
    }
    pack_transient_plan(result.transients);
    // Schedule dependencies describe authored ordering. Add the conservative
    // resource hazards required by concrete command recording so a future
    // recipe may expose independent commands without weakening mutation or
    // alias correctness.
    std::map<StorageId, uint32_t> last_writer;
    std::map<StorageId, std::set<uint32_t>> readers;
    for (Command & command : result.commands) {
        std::set<uint32_t> dependencies(command.dependencies.begin(), command.dependencies.end());
        for (const CommandBinding & binding : command.bindings) {
            auto writer = last_writer.find(binding.storage);
            if (writer != last_writer.end()) dependencies.insert(writer->second);
            if (binding.access != ResourceAccess::Read) {
                const auto storage_readers = readers.find(binding.storage);
                if (storage_readers != readers.end()) dependencies.insert(storage_readers->second.begin(), storage_readers->second.end());
            }
        }
        dependencies.erase(command.ordinal);
        command.dependencies.assign(dependencies.begin(), dependencies.end());
        for (const CommandBinding & binding : command.bindings) {
            if (binding.access == ResourceAccess::Read) {
                readers[binding.storage].insert(command.ordinal);
            } else {
                last_writer[binding.storage] = command.ordinal;
                readers[binding.storage].clear();
            }
        }
    }
    return result;
}

VerificationResult verify_command_program(const ProgramPlan & plan, const KernelCorpus & corpus,
                                          const CommandProgram & commands) {
    VerificationResult result;
    if (!commands.valid()) result.errors.insert(result.errors.end(), commands.errors.begin(), commands.errors.end());
    if (commands.graph_fingerprint != plan.graph.fingerprint || commands.workload != plan.schedule.workload ||
        commands.target != plan.target || commands.corpus_digest != corpus.corpus_digest) {
        result.errors.push_back("command program identity does not match its plan or corpus");
    }
    const size_t kernel_count = std::count_if(commands.commands.begin(), commands.commands.end(), [](const Command & command) {
        return command.kind == CommandKind::Kernel;
    });
    if (kernel_count != schedule_dispatch_count(plan.schedule)) {
        result.errors.push_back("command program does not cover every scheduled dispatch");
    }
    for (size_t i = 0; i < commands.commands.size(); ++i) {
        const Command & command = commands.commands[i];
        if (command.ordinal != i) result.errors.push_back("command ordinals are not contiguous");
        const KernelDefinition * definition = command.kind == CommandKind::Kernel
            ? find_kernel(corpus, command.kernel_id) : nullptr;
        if (command.kind == CommandKind::Kernel && definition == nullptr) {
            result.errors.push_back("command references an unknown kernel " + command.kernel_id);
        } else if (definition != nullptr) {
            for (const std::string & scalar : definition->scalar_parameters) {
                if (command.scalar_parameters.count(scalar) == 0) {
                    result.errors.push_back("command " + std::to_string(command.ordinal) + " kernel " +
                        command.kernel_id + " omits scalar " + scalar);
                }
            }
            if (command.bindings.size() != definition->bindings.size()) {
                result.errors.push_back("command " + std::to_string(command.ordinal) + " kernel " + command.kernel_id +
                    " has " + std::to_string(command.bindings.size()) + " bindings but its ABI requires " +
                    std::to_string(definition->bindings.size()));
            }
            const size_t shared_count = std::min(command.bindings.size(), definition->bindings.size());
            for (size_t binding_index = 0; binding_index < shared_count; ++binding_index) {
                if (command.bindings[binding_index].name != definition->bindings[binding_index].name ||
                    command.bindings[binding_index].access != definition->bindings[binding_index].access) {
                    result.errors.push_back("command " + std::to_string(command.ordinal) + " kernel " +
                        command.kernel_id + " binding " + std::to_string(binding_index) +
                        " does not match the kernel ABI");
                }
            }
        }
        if (command.kind == CommandKind::Copy && command.bindings.size() != 2) {
            result.errors.push_back("copy command does not have source and destination bindings");
        }
        if (command.kind == CommandKind::Fill && command.bindings.size() != 1) {
            result.errors.push_back("fill command does not have one destination binding");
        }
        for (uint32_t dependency : command.dependencies) {
            if (dependency >= command.ordinal) result.errors.push_back("command has a forward dependency");
        }
        for (const CommandBinding & binding : command.bindings) {
            if (binding.storage >= plan.graph.storages.size() || binding.length == 0 ||
                binding.offset + binding.length > plan.graph.storages[binding.storage].size) {
                result.errors.push_back("command has an invalid resource range");
            }
        }
    }
    for (const ConstantInitialization & initialization : commands.initializations) {
        if (initialization.storage >= plan.graph.storages.size() || initialization.data.empty() ||
            initialization.data.size() > plan.graph.storages[initialization.storage].size) {
            result.errors.push_back("invalid constant initialization payload");
        }
    }
    for (const TransientAllocation & a : commands.transients.allocations) {
        if (a.storage >= plan.graph.storages.size() || !plan.resources.resources[a.storage].elidable ||
            a.size != plan.graph.storages[a.storage].size || a.arena_offset % a.alignment != 0 ||
            a.arena_offset + a.size > commands.transients.arena_size) {
            result.errors.push_back("invalid transient allocation");
        }
        for (const TransientAllocation & b : commands.transients.allocations) {
            if (a.id >= b.id) continue;
            const bool lifetime_overlap = a.first_command <= b.last_command && b.first_command <= a.last_command;
            const bool range_overlap = a.arena_offset < b.arena_offset + b.size && b.arena_offset < a.arena_offset + a.size;
            if (lifetime_overlap && range_overlap) result.errors.push_back("live transients overlap");
        }
    }
    return result;
}

VerificationResult verify_binding_snapshot(const ProgramPlan & plan, const BindingSnapshot & snapshot) {
    VerificationResult result;
    if (snapshot.device_identity.empty()) result.errors.push_back("binding snapshot has no device identity");
    std::set<StorageId> storages;
    for (const ConcreteBinding & binding : snapshot.bindings) {
        if (binding.storage >= plan.graph.storages.size() || !storages.insert(binding.storage).second) {
            result.errors.push_back("binding snapshot has invalid or duplicate storage");
            continue;
        }
        if (binding.buffer_identity == 0 || binding.generation == 0 || binding.length == 0 ||
            binding.offset + binding.length > binding.capacity || binding.length < plan.graph.storages[binding.storage].size) {
            result.errors.push_back("binding snapshot range or identity is invalid");
        }
    }
    for (const ResourceContract & resource : plan.resources.resources) {
        if (!resource.elidable && storages.count(resource.storage) == 0) {
            result.errors.push_back("binding snapshot omits pinned storage " + std::to_string(resource.storage));
        }
    }
    return result;
}

AllocationFingerprint fingerprint_bindings(const BindingSnapshot & snapshot) {
    std::ostringstream witness;
    witness << snapshot.device_identity << '\n';
    std::vector<ConcreteBinding> bindings = snapshot.bindings;
    std::sort(bindings.begin(), bindings.end(), [](const ConcreteBinding & a, const ConcreteBinding & b) {
        return a.storage < b.storage;
    });
    for (const ConcreteBinding & binding : bindings) {
        witness << binding.storage << ':' << binding.buffer_identity << ':' << binding.generation << ':'
                << binding.capacity << ':' << binding.offset << ':' << binding.length << '\n';
    }
    return { stable_hash(witness.str()) };
}

std::string format_kernel_corpus(const KernelCorpus & corpus) {
    std::ostringstream out;
    out << "kernel-corpus " << corpus.schema << " revision=" << corpus.upstream_revision
        << " digest=" << corpus.corpus_digest << " recipe=" << corpus.recipe_digest
        << " kernels=" << corpus.kernels.size() << " plan_cases=" << corpus.plan_case_count << '\n';
    for (const KernelDefinition & kernel : corpus.kernels) {
        out << "  kernel " << kernel.id << " target=" << kernel.target << " symbol=@" << kernel.symbol
            << " source=" << kernel.source << " sha256=" << kernel.source_digest << '\n';
        out << "    recipe " << kernel.compile_recipe.mode;
        if (!kernel.compile_recipe.link_module.empty()) out << " module=" << kernel.compile_recipe.link_module;
        out << " primary=";
        for (size_t i = 0; i < kernel.compile_recipe.primary_sources.size(); ++i)
            out << (i ? "," : "") << kernel.compile_recipe.primary_sources[i];
        out << " libraries=";
        for (size_t i = 0; i < kernel.compile_recipe.library_sources.size(); ++i)
            out << (i ? "," : "") << kernel.compile_recipe.library_sources[i];
        out << '\n';
        for (size_t i = 0; i < kernel.bindings.size(); ++i) {
            out << "    binding[" << i << "] " << kernel.bindings[i].name << ' '
                << resource_access_name(kernel.bindings[i].access) << '\n';
        }
        for (const KernelScalarDefinition & parameter : kernel.workload_parameters)
            out << "    workload " << parameter.name << ' ' << parameter.type << '\n';
        for (const KernelScalarDefinition & parameter : kernel.launch_parameters)
            out << "    launch " << parameter.name << ' ' << parameter.type << '\n';
    }
    return out.str();
}

std::string format_resource_program(const ResourceProgram & resources) {
    std::ostringstream out;
    out << "resources count=" << resources.resources.size() << " uses=" << resources.uses.size() << '\n';
    for (const ResourceContract & resource : resources.resources) {
        out << "  storage " << resource.storage << " bytes=" << resource.size << " versions=0.." << resource.final_version
            << " lifetime=";
        if (resource.first_invocation == UINT32_MAX) out << "unused";
        else out << resource.first_invocation << ".." << resource.last_invocation;
        out << " flags=" << (resource.imported ? "I" : "-") << (resource.weight ? "W" : "-")
            << (resource.mutable_state ? "M" : "-") << (resource.exported ? "E" : "-")
            << (resource.elidable ? "T" : "-") << " aliases=" << resource.aliases.size() << '\n';
    }
    return out.str();
}

std::string format_command_program(const CommandProgram & program) {
    std::ostringstream out;
    out << "command-program " << program.schema << " workload=" << program.workload << " target=" << program.target
        << " graph=" << program.graph_fingerprint << " recipe=" << program.recipe_revision
        << " corpus=" << program.corpus_digest << " commands=" << program.commands.size() << '\n';
    for (const Command & command : program.commands) {
        out << "  command " << command.ordinal << ' ' << command_kind_name(command.kind) << ' ' << command.kernel_id
            << " label=" << command.label << " deps=[";
        for (size_t i = 0; i < command.dependencies.size(); ++i) out << (i ? "," : "") << command.dependencies[i];
        out << "] scalars={";
        size_t scalar_index = 0;
        for (const auto & scalar : command.scalar_parameters) out << (scalar_index++ ? "," : "") << scalar.first << '=' << scalar.second;
        out << "} configs={";
        size_t config_index = 0;
        for (const auto & config : command.compile_parameters) out << (config_index++ ? "," : "") << config.first << '=' << config.second;
        out << "}\n";
        for (size_t i = 0; i < command.bindings.size(); ++i) {
            const CommandBinding & binding = command.bindings[i];
            out << "    binding[" << i << "] " << binding.name << ' ' << resource_access_name(binding.access)
                << " storage=" << binding.storage << " range=" << binding.offset << "+" << binding.length;
            if (binding.origin == BindingOrigin::Transient) out << " transient=" << binding.transient;
            out << '\n';
        }
    }
    out << "transients arena=" << program.transients.arena_size << " alignment=" << program.transients.arena_alignment
        << " allocations=" << program.transients.allocations.size() << '\n';
    for (const TransientAllocation & allocation : program.transients.allocations) {
        out << "  transient " << allocation.id << " storage=" << allocation.storage << " range="
            << allocation.arena_offset << "+" << allocation.size << " live=" << allocation.first_command
            << ".." << allocation.last_command << '\n';
    }
    for (const ConstantInitialization & initialization : program.initializations) {
        out << "  initialize " << initialization.label << " storage=" << initialization.storage
            << " bytes=" << initialization.data.size() << '\n';
    }
    return out.str();
}

std::string format_verification_errors(const std::vector<std::string> & errors) {
    std::ostringstream out;
    for (const std::string & error : errors) out << "error=" << error << '\n';
    return out.str();
}

std::string format_verification_summary(const std::vector<std::string> & errors) {
    std::map<std::string, size_t> counts;
    for (std::string error : errors) {
        if (error.rfind("command ", 0) == 0 && error.size() > 8 && std::isdigit(static_cast<unsigned char>(error[8]))) {
            const size_t ordinal_end = error.find(' ', 8);
            if (ordinal_end != std::string::npos) error.replace(8, ordinal_end - 8, "<ordinal>");
        }
        ++counts[error];
    }
    std::ostringstream out;
    out << "error_count=" << errors.size() << '\n';
    for (const auto & item : counts) out << "error_summary=" << item.second << '\t' << item.first << '\n';
    return out.str();
}

std::string format_binding_snapshot(const BindingSnapshot & snapshot, bool include_runtime_identities) {
    std::ostringstream out;
    out << "binding-snapshot device=" << snapshot.device_identity << " bindings=" << snapshot.bindings.size()
        << " fingerprint=" << fingerprint_bindings(snapshot).value << '\n';
    std::vector<ConcreteBinding> bindings = snapshot.bindings;
    std::sort(bindings.begin(), bindings.end(), [](const ConcreteBinding & a, const ConcreteBinding & b) {
        return a.storage < b.storage;
    });
    for (const ConcreteBinding & binding : bindings) {
        out << "  storage " << binding.storage;
        if (include_runtime_identities) out << " buffer=0x" << std::hex << binding.buffer_identity << std::dec;
        else out << " buffer=<runtime>";
        out << " generation=" << binding.generation << " capacity=" << binding.capacity
            << " range=" << binding.offset << "+" << binding.length << '\n';
    }
    return out.str();
}

std::string serialize_kernel_corpus_json(const KernelCorpus & corpus) {
    nlohmann::ordered_json root = {
        { "schema", corpus.schema }, { "upstream_revision", corpus.upstream_revision },
        { "corpus_digest", corpus.corpus_digest }, { "recipe_digest", corpus.recipe_digest },
        { "plan_case_count", corpus.plan_case_count }, { "kernels", nlohmann::ordered_json::array() },
    };
    for (const KernelDefinition & kernel : corpus.kernels) {
        nlohmann::ordered_json item = {
            { "id", kernel.id }, { "source", kernel.source }, { "dependencies", kernel.dependencies },
            { "symbol", kernel.symbol }, { "target", kernel.target }, { "compile_config", kernel.compile_config },
            { "scalar_parameters", kernel.scalar_parameters }, { "source_digest", kernel.source_digest },
            { "compile_recipe", {
                { "mode", kernel.compile_recipe.mode }, { "link_module", kernel.compile_recipe.link_module },
                { "primary_sources", kernel.compile_recipe.primary_sources },
                { "library_sources", kernel.compile_recipe.library_sources },
            } },
            { "workload_parameters", nlohmann::ordered_json::array() },
            { "launch_parameters", nlohmann::ordered_json::array() },
            { "bindings", nlohmann::ordered_json::array() },
        };
        for (const KernelScalarDefinition & parameter : kernel.workload_parameters)
            item["workload_parameters"].push_back({ { "name", parameter.name }, { "type", parameter.type } });
        for (const KernelScalarDefinition & parameter : kernel.launch_parameters)
            item["launch_parameters"].push_back({ { "name", parameter.name }, { "type", parameter.type } });
        for (const KernelBindingDefinition & binding : kernel.bindings) {
            item["bindings"].push_back({ { "name", binding.name }, { "access", resource_access_name(binding.access) } });
        }
        root["kernels"].push_back(std::move(item));
    }
    return root.dump();
}

std::string serialize_command_program_json(const CommandProgram & program) {
    nlohmann::ordered_json root = {
        { "schema", program.schema }, { "workload", program.workload }, { "target", program.target },
        { "graph_fingerprint", program.graph_fingerprint }, { "recipe_revision", program.recipe_revision },
        { "corpus_digest", program.corpus_digest }, { "commands", nlohmann::ordered_json::array() },
        { "initializations", nlohmann::ordered_json::array() },
        { "transients", { { "arena_size", program.transients.arena_size },
            { "arena_alignment", program.transients.arena_alignment }, { "allocations", nlohmann::ordered_json::array() } } },
    };
    for (const Command & command : program.commands) {
        nlohmann::ordered_json item = {
            { "ordinal", command.ordinal }, { "kind", command_kind_name(command.kind) }, { "label", command.label },
            { "kernel", command.kernel_id }, { "scalars", command.scalar_parameters },
            { "compile_parameters", command.compile_parameters },
            { "workgroup_count", command.workgroup_count }, { "workgroup_size", command.workgroup_size },
            { "subgroup_size", command.subgroup_size }, { "dependencies", command.dependencies },
            { "bindings", nlohmann::ordered_json::array() },
        };
        for (const CommandBinding & binding : command.bindings) item["bindings"].push_back(binding_json(binding));
        root["commands"].push_back(std::move(item));
    }
    for (const ConstantInitialization & initialization : program.initializations) {
        std::ostringstream bytes;
        bytes << std::hex << std::setfill('0');
        for (uint8_t byte : initialization.data) bytes << std::setw(2) << static_cast<unsigned>(byte);
        root["initializations"].push_back({ { "label", initialization.label }, { "storage", initialization.storage },
                                             { "bytes_hex", bytes.str() } });
    }
    for (const TransientAllocation & allocation : program.transients.allocations) {
        root["transients"]["allocations"].push_back({
            { "id", allocation.id }, { "storage", allocation.storage }, { "size", allocation.size },
            { "alignment", allocation.alignment }, { "arena_offset", allocation.arena_offset },
            { "first_command", allocation.first_command }, { "last_command", allocation.last_command },
        });
    }
    return root.dump();
}

std::string serialize_binding_snapshot_json(const BindingSnapshot & snapshot, bool include_runtime_identities) {
    nlohmann::ordered_json root = {
        { "schema", "ggml-hrx-binding-snapshot-v1" }, { "device", snapshot.device_identity },
        { "fingerprint", fingerprint_bindings(snapshot).value }, { "bindings", nlohmann::ordered_json::array() },
    };
    std::vector<ConcreteBinding> bindings = snapshot.bindings;
    std::sort(bindings.begin(), bindings.end(), [](const ConcreteBinding & a, const ConcreteBinding & b) {
        return a.storage < b.storage;
    });
    for (const ConcreteBinding & binding : bindings) {
        nlohmann::ordered_json item = {
            { "storage", binding.storage }, { "generation", binding.generation }, { "capacity", binding.capacity },
            { "offset", binding.offset }, { "length", binding.length },
        };
        if (include_runtime_identities) item["buffer_identity"] = binding.buffer_identity;
        root["bindings"].push_back(std::move(item));
    }
    return root.dump();
}

std::string command_program_dot(const CommandProgram & program) {
    std::ostringstream out;
    out << "digraph hrx_commands {\n  rankdir=LR;\n  node [shape=box,fontname=monospace];\n";
    std::vector<std::set<uint32_t>> ancestors(program.commands.size());
    for (const Command & command : program.commands) {
        out << "  c" << command.ordinal << " [label=\"" << command.ordinal << ": " << command.kernel_id
            << "\\n" << command.label << "\"];\n";
        for (uint32_t dependency : command.dependencies) {
            bool redundant = false;
            for (uint32_t other : command.dependencies) {
                if (other != dependency && other < ancestors.size() && ancestors[other].count(dependency) != 0) {
                    redundant = true;
                    break;
                }
            }
            if (!redundant) out << "  c" << dependency << " -> c" << command.ordinal << ";\n";
            if (dependency < ancestors.size()) {
                ancestors[command.ordinal].insert(dependency);
                ancestors[command.ordinal].insert(ancestors[dependency].begin(), ancestors[dependency].end());
            }
        }
    }
    out << "}\n";
    return out.str();
}

} // namespace ggml::hrx
