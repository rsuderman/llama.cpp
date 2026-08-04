#pragma once

#include "graph/resource-access.h"
#include "graph/schedule.h"

#include <cstddef>
#include <string>

namespace ggml::hrx {

template <typename T> struct KernelSpan {
    const T * items = nullptr;
    size_t    count = 0;

    const T * begin() const { return items; }

    const T * end() const { return items == nullptr ? nullptr : items + count; }

    const T * data() const { return items; }

    size_t size() const { return count; }

    bool empty() const { return count == 0; }

    const T & operator[](size_t index) const { return items[index]; }

    const T & front() const { return items[0]; }
};

struct KernelCompileConfig {
    const char * key   = "";
    const char * value = "";
};

struct KernelBindingDefinition {
    const char *   name   = "";
    ResourceAccess access = ResourceAccess::Read;
};

struct KernelScalarDefinition {
    const char * name = "";
    const char * type = "";
};

struct KernelCompileRecipe {
    const char *             mode        = "";
    const char *             link_module = "";
    KernelSpan<const char *> primary_sources;
    KernelSpan<const char *> library_sources;
};

struct KernelDefinition {
    const char *                        id     = "";
    const char *                        source = "";
    KernelSpan<const char *>            dependencies;
    const char *                        symbol = "";
    const char *                        target = "";
    KernelSpan<KernelCompileConfig>     compile_config;
    KernelSpan<const char *>            scalar_parameters;
    KernelSpan<KernelBindingDefinition> bindings;
    const char *                        source_digest = "";
    KernelSpan<KernelScalarDefinition>  workload_parameters;
    KernelSpan<KernelScalarDefinition>  launch_parameters;
    KernelCompileRecipe                 compile_recipe;
};

struct KernelCorpus {
    const char *                 schema            = "ggml-hrx-kernel-corpus-v1";
    const char *                 upstream_revision = "";
    const char *                 corpus_digest     = "";
    const char *                 recipe_digest     = "";
    size_t                       plan_case_count   = 0;
    KernelSpan<KernelDefinition> kernels;
};

enum KernelSourceFormat {
    KERNEL_SOURCE_FORMAT_TEXT,
    KERNEL_SOURCE_FORMAT_BINARY,
};

struct KernelSourceSpan {
    const char *       source;
    size_t             length;
    KernelSourceFormat format;
};

struct KernelSource {
    KernelSourceSpan         source;
    const KernelSourceSpan * dependencies;
    size_t                   dependency_count;
};

const KernelSource * get_kernel_source(const char * source_path);
const KernelCorpus & get_qwen_kernel_corpus(const char * target);
VerificationResult   verify_kernel_corpus(const KernelCorpus & corpus);
std::string          format_kernel_corpus(const KernelCorpus & corpus);

}  // namespace ggml::hrx
