#include "kernel-corpus.h"

#include <algorithm>
#include <cstring>
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

static bool contains_string(KernelSpan<const char *> values, const char * value) {
    return std::find_if(values.begin(), values.end(), [&](const char * item) { return string_equal(item, value); }) !=
           values.end();
}

#include "kernel-corpus-qwen.inc"
#include "kernel-corpus-sources.inc"

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
    std::set<std::string> ids;
    for (const KernelDefinition & kernel : corpus.kernels) {
        if (string_empty(kernel.id) || string_empty(kernel.source) || string_empty(kernel.symbol) ||
            string_empty(kernel.target) || string_empty(kernel.source_digest)) {
            result.errors.push_back("kernel definition is incomplete");
        }
        const bool source_is_primary = contains_string(kernel.compile_recipe.primary_sources, kernel.source);
        const bool source_is_library = contains_string(kernel.compile_recipe.library_sources, kernel.source);
        if ((!string_equal(kernel.compile_recipe.mode, "direct") &&
             !string_equal(kernel.compile_recipe.mode, "archive")) ||
            kernel.compile_recipe.primary_sources.empty() || (!source_is_primary && !source_is_library) ||
            (string_equal(kernel.compile_recipe.mode, "archive") && string_empty(kernel.compile_recipe.link_module))) {
            result.errors.push_back("kernel " + std::string(kernel.id != nullptr ? kernel.id : "") +
                                    " has an invalid BUILD compile recipe");
        }
        if (!ids.insert(kernel.id != nullptr ? kernel.id : "").second) {
            result.errors.push_back("kernel corpus repeats id " + std::string(kernel.id != nullptr ? kernel.id : ""));
        }
        std::set<std::string> names;
        for (const KernelBindingDefinition & binding : kernel.bindings) {
            if (string_empty(binding.name) || !names.insert(binding.name != nullptr ? binding.name : "").second) {
                result.errors.push_back("kernel " + std::string(kernel.id != nullptr ? kernel.id : "") +
                                        " has invalid binding names");
            }
        }
        if (kernel.bindings.size() == 0) {
            result.errors.push_back("kernel " + std::string(kernel.id != nullptr ? kernel.id : "") +
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
        out << "  kernel " << kernel.id << " target=" << kernel.target << " symbol=@" << kernel.symbol
            << " source=" << kernel.source << " sha256=" << kernel.source_digest << '\n';
        out << "    recipe " << kernel.compile_recipe.mode;
        if (!string_empty(kernel.compile_recipe.link_module)) {
            out << " module=" << kernel.compile_recipe.link_module;
        }
        out << " primary=";
        for (size_t i = 0; i < kernel.compile_recipe.primary_sources.size(); ++i) {
            out << (i ? "," : "") << kernel.compile_recipe.primary_sources[i];
        }
        out << " libraries=";
        for (size_t i = 0; i < kernel.compile_recipe.library_sources.size(); ++i) {
            out << (i ? "," : "") << kernel.compile_recipe.library_sources[i];
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
