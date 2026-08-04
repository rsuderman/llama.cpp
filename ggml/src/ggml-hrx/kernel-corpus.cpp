#include "kernel-corpus.h"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <set>
#include <sstream>
#include <string>

namespace ggml::hrx {
namespace {

const char * kernel_resource_access_name(ResourceAccess access) {
    switch (access) {
        case ResourceAccess::Read:
            return "read";
        case ResourceAccess::Write:
            return "write";
        case ResourceAccess::ReadWrite:
            return "read_write";
    }
    return "unknown";
}

struct KernelSourceRecordEntry {
    const char *         source_path;
    const KernelSource * source;
};

static bool string_equal(const char * lhs, const char * rhs) {
    return std::strcmp(lhs != nullptr ? lhs : "", rhs != nullptr ? rhs : "") == 0;
}

static bool string_empty(const char * value) {
    return value == nullptr || value[0] == 0;
}

static bool contains_source_ref(KernelSpan<KernelSourceRef> values, const char * path) {
    return std::find_if(values.begin(), values.end(), [&](const KernelSourceRef & item) {
        return string_equal(item.path, path);
    }) !=
           values.end();
}

#include "kernel-corpus-sources.inc"
#include "kernel-corpus-qwen.inc"

}  // namespace

const KernelSource * get_kernel_source(const char * source_path) {
    if (source_path == nullptr) {
        return nullptr;
    }
    for (const KernelSourceRecordEntry & entry : kKernelSourceRecords) {
        if (std::strcmp(source_path, entry.source_path) == 0) {
            return entry.source;
        }
    }
    return nullptr;
}

const KernelCorpus & get_qwen_kernel_corpus(const char * target) {
    (void) target;
    return kQwenKernelCorpus;
}

KernelResolveResult resolve_kernel_definition(const KernelCorpus & corpus, const std::string & family,
                                              const std::string & name, uint64_t id,
                                              KernelSpecialization::ExecutionKind execution_kind) {
    if (execution_kind == KernelSpecialization::ExecutionKind::NativeGap) {
        return {
            id == kUncatalogedKernelId ? KernelResolveStatus::NativeGap : KernelResolveStatus::InvalidNativeGap,
            nullptr,
        };
    }
    if (execution_kind != KernelSpecialization::ExecutionKind::Native &&
        execution_kind != KernelSpecialization::ExecutionKind::NativeEager) {
        return { KernelResolveStatus::UncatalogedNative, nullptr };
    }
    if (id == kUncatalogedKernelId) {
        return { KernelResolveStatus::UncatalogedNative, nullptr };
    }
    bool id_match = false;
    for (const KernelDefinition & kernel : corpus.kernels) {
        if (kernel.id != id) {
            continue;
        }
        id_match = true;
        if (string_equal(kernel.family, family.c_str()) && string_equal(kernel.name, name.c_str())) {
            return { KernelResolveStatus::Found, &kernel };
        }
    }
    return { id_match ? KernelResolveStatus::HashCollision : KernelResolveStatus::MissingActiveCorpusEntry, nullptr };
}

KernelResolveResult resolve_kernel_definition(const KernelCorpus & corpus, const KernelSpecialization & kernel) {
    return resolve_kernel_definition(corpus, kernel.family, kernel.variant, kernel.kernel_id, kernel.execution_kind);
}

const char * kernel_resolve_status_name(KernelResolveStatus status) {
    switch (status) {
        case KernelResolveStatus::Found:
            return "found";
        case KernelResolveStatus::NativeGap:
            return "native_gap";
        case KernelResolveStatus::UncatalogedNative:
            return "uncataloged_native";
        case KernelResolveStatus::MissingActiveCorpusEntry:
            return "missing_active_corpus_entry";
        case KernelResolveStatus::HashCollision:
            return "hash_collision";
        case KernelResolveStatus::InvalidNativeGap:
            return "invalid_native_gap";
    }
    return "unknown";
}

std::string format_kernel_resolve_error(const KernelResolveResult & result, const std::string & family,
                                        const std::string & name) {
    KernelSpecialization kernel;
    kernel.family = family;
    kernel.variant = name;
    const std::string label = kernel_specialization_name(kernel);
    switch (result.status) {
        case KernelResolveStatus::Found:
            return "";
        case KernelResolveStatus::NativeGap:
            return "native gap for " + label;
        case KernelResolveStatus::UncatalogedNative:
            return "uncataloged native kernel " + label;
        case KernelResolveStatus::MissingActiveCorpusEntry:
            return "cataloged kernel " + label + " is not available in the active corpus";
        case KernelResolveStatus::HashCollision:
            return "kernel catalog id collision while resolving " + label;
        case KernelResolveStatus::InvalidNativeGap:
            return "native gap " + label + " unexpectedly has a catalog id";
    }
    return "unknown kernel resolution failure for " + label;
}

std::string format_kernel_resolve_error(const KernelResolveResult & result, const KernelSpecialization & kernel) {
    return format_kernel_resolve_error(result, kernel.family, kernel.variant);
}

VerificationResult verify_kernel_corpus(const KernelCorpus & corpus) {
    VerificationResult result;
    if (!string_equal(corpus.schema, "ggml-hrx-kernel-corpus-v1")) {
        result.errors.push_back("unsupported kernel corpus schema");
    }
    if (string_empty(corpus.upstream_revision)) {
        result.errors.push_back("kernel corpus has no upstream revision");
    }
    if (string_empty(corpus.corpus_digest)) {
        result.errors.push_back("kernel corpus has no digest");
    }
    if (string_empty(corpus.recipe_digest)) {
        result.errors.push_back("kernel corpus has no BUILD.bazel recipe digest");
    }
    if (corpus.plan_case_count == 0) {
        result.errors.push_back("kernel corpus has no compile plan cases");
    }
    std::set<std::string> names;
    for (const KernelDefinition & kernel : corpus.kernels) {
        if (string_empty(kernel.family) || string_empty(kernel.name) || string_empty(kernel.source) || string_empty(kernel.symbol) ||
            string_empty(kernel.target) || string_empty(kernel.source_digest)) {
            result.errors.push_back("kernel definition is incomplete");
        }
        if (kernel.id != kernel_catalog_id(kernel.family != nullptr ? kernel.family : "",
                                           kernel.name != nullptr ? kernel.name : "")) {
            result.errors.push_back("kernel " + std::string(kernel.name != nullptr ? kernel.name : "") +
                                    " has an invalid catalog id");
        }
        const bool source_is_primary = contains_source_ref(kernel.compile_recipe.primary_sources, kernel.source);
        const bool source_is_library = contains_source_ref(kernel.compile_recipe.library_sources, kernel.source);
        if ((!string_equal(kernel.compile_recipe.mode, "direct") &&
             !string_equal(kernel.compile_recipe.mode, "archive")) ||
            kernel.compile_recipe.primary_sources.empty() || (!source_is_primary && !source_is_library) ||
            (string_equal(kernel.compile_recipe.mode, "archive") && string_empty(kernel.compile_recipe.link_module))) {
            result.errors.push_back("kernel " + std::string(kernel.name != nullptr ? kernel.name : "") +
                                    " has an invalid BUILD compile recipe");
        }
        for (const KernelSourceRef & source : kernel.compile_recipe.primary_sources) {
            if (string_empty(source.path) || source.contents == nullptr) {
                result.errors.push_back("kernel " + std::string(kernel.name != nullptr ? kernel.name : "") +
                                        " has an invalid embedded primary source reference");
            }
        }
        for (const KernelSourceRef & source : kernel.compile_recipe.library_sources) {
            if (string_empty(source.path) || source.contents == nullptr) {
                result.errors.push_back("kernel " + std::string(kernel.name != nullptr ? kernel.name : "") +
                                        " has an invalid embedded library source reference");
            }
        }
        const std::string full_name = std::string(kernel.family != nullptr ? kernel.family : "") + ":" +
                                      std::string(kernel.name != nullptr ? kernel.name : "");
        if (!names.insert(full_name).second) {
            result.errors.push_back("kernel corpus repeats name " + full_name);
        }
        std::set<std::string> binding_names;
        for (const KernelBindingDefinition & binding : kernel.bindings) {
            if (string_empty(binding.name) || !binding_names.insert(binding.name != nullptr ? binding.name : "").second) {
                result.errors.push_back("kernel " + std::string(kernel.name != nullptr ? kernel.name : "") +
                                        " has invalid binding names");
            }
        }
        if (kernel.bindings.size() == 0) {
            result.errors.push_back("kernel " + std::string(kernel.name != nullptr ? kernel.name : "") +
                                    " has no binding ABI");
        }
    }
    return result;
}

std::string format_kernel_corpus(const KernelCorpus & corpus) {
    std::ostringstream out;
    out << "kernel-corpus " << corpus.schema << " revision=" << corpus.upstream_revision
        << " digest=" << corpus.corpus_digest << " recipe=" << corpus.recipe_digest
        << " kernels=" << corpus.kernels.size() << " plan_cases=" << corpus.plan_case_count << '\n';
    for (const KernelDefinition & kernel : corpus.kernels) {
        out << "  kernel " << kernel.family << ':' << kernel.name << " id=0x" << std::hex << kernel.id << std::dec
            << " target=" << kernel.target << " symbol=@" << kernel.symbol
            << " source=" << kernel.source << " sha256=" << kernel.source_digest << '\n';
        out << "    recipe " << kernel.compile_recipe.mode;
        if (!string_empty(kernel.compile_recipe.link_module)) {
            out << " module=" << kernel.compile_recipe.link_module;
        }
        out << " primary=";
        for (size_t i = 0; i < kernel.compile_recipe.primary_sources.size(); ++i) {
            out << (i ? "," : "") << kernel.compile_recipe.primary_sources[i].path;
        }
        out << " libraries=";
        for (size_t i = 0; i < kernel.compile_recipe.library_sources.size(); ++i) {
            out << (i ? "," : "") << kernel.compile_recipe.library_sources[i].path;
        }
        out << '\n';
        for (size_t i = 0; i < kernel.bindings.size(); ++i) {
            out << "    binding[" << i << "] " << kernel.bindings[i].name << ' '
                << kernel_resource_access_name(kernel.bindings[i].access) << '\n';
        }
        for (const KernelScalarDefinition & parameter : kernel.workload_parameters) {
            out << "    workload " << parameter.name << ' ' << parameter.type << '\n';
        }
        for (const KernelScalarDefinition & parameter : kernel.launch_parameters) {
            out << "    launch " << parameter.name << ' ' << parameter.type << '\n';
        }
    }
    return out.str();
}

}  // namespace ggml::hrx
