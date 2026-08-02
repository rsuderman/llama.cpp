#include "command-program.h"

#include <algorithm>
#include <cctype>
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

    std::vector<size_t> order(result.allocations.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
        const TransientAllocation & a = result.allocations[lhs];
        const TransientAllocation & b = result.allocations[rhs];
        if (a.first_command != b.first_command) return a.first_command < b.first_command;
        if (a.size != b.size) return a.size > b.size;
        return a.storage < b.storage;
    });
    for (size_t index : order) {
        TransientAllocation & allocation = result.allocations[index];
        size_t candidate = 0;
        for (;;) {
            candidate = align_up(candidate, allocation.alignment);
            bool conflict = false;
            size_t next_candidate = candidate;
            for (const TransientAllocation & placed : result.allocations) {
                if (&placed == &allocation || placed.arena_offset + placed.size == 0) continue;
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
        result.arena_size = std::max(result.arena_size, candidate + allocation.size);
    }
    result.arena_size = align_up(result.arena_size, result.arena_alignment);
    std::sort(result.allocations.begin(), result.allocations.end(),
              [](const TransientAllocation & a, const TransientAllocation & b) { return a.id < b.id; });
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
        if (!ids.insert(kernel.id).second) result.errors.push_back("kernel corpus repeats id " + kernel.id);
        std::set<std::string> names;
        for (const KernelBindingDefinition & binding : kernel.bindings) {
            if (binding.name.empty() || !names.insert(binding.name).second) {
                result.errors.push_back("kernel " + kernel.id + " has invalid binding names");
            }
        }
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
            for (const char * group : { "workload_parameters", "launch_parameters" }) {
                for (const nlohmann::json & parameter : item.at(group)) {
                    const std::string name = parameter.at("name").get<std::string>();
                    if (std::find(kernel.scalar_parameters.begin(), kernel.scalar_parameters.end(), name) ==
                        kernel.scalar_parameters.end()) kernel.scalar_parameters.push_back(name);
                }
            }
            const std::vector<std::string> binding_names = item.at("bindings").get<std::vector<std::string>>();
            const std::vector<std::string> explicit_access = item.value("binding_access", std::vector<std::string>());
            if (!explicit_access.empty() && explicit_access.size() != binding_names.size()) {
                throw std::runtime_error("kernel binding access metadata has the wrong arity");
            }
            for (size_t binding_index = 0; binding_index < binding_names.size(); ++binding_index) {
                const std::string & name = binding_names[binding_index];
                ResourceAccess access = ResourceAccess::Read;
                if (!explicit_access.empty()) {
                    if (explicit_access[binding_index] == "write") access = ResourceAccess::Write;
                    else if (explicit_access[binding_index] == "read_write") access = ResourceAccess::ReadWrite;
                    else if (explicit_access[binding_index] != "read") throw std::runtime_error("invalid kernel binding access metadata");
                } else if (name.find("output") != std::string::npos || name.find("partial") != std::string::npos ||
                    name.find("counter") != std::string::npos || name == "destination") access = ResourceAccess::ReadWrite;
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
    if (commands.commands.size() != schedule_dispatch_count(plan.schedule)) {
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
        for (size_t i = 0; i < kernel.bindings.size(); ++i) {
            out << "    binding[" << i << "] " << kernel.bindings[i].name << ' '
                << resource_access_name(kernel.bindings[i].access) << '\n';
        }
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
            { "bindings", nlohmann::ordered_json::array() },
        };
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
