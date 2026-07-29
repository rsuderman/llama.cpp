#include "loom-catalog/ggml-hrx-loom-catalog-runtime-internal.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static ggml_backend_hrx_loom_catalog_entry make_entry(const unsigned char * source_data, size_t source_size) {
    return {
        /* .id                   = */ "add_f32",
        /* .op                   = */ "GGML_OP_ADD",
        /* .target               = */ "gfx1100",
        /* .source_name          = */ "sources/add_f32.loom",
        /* .source_data          = */ source_data,
        /* .source_size          = */ source_size,
        /* .source_format        = */ "loom-text",
        /* .symbol               = */ "hrx_add_f32",
        /* .workgroup_size       = */ {256, 1, 1},
        /* .binding_count        = */ 3,
        /* .parameter_count      = */ 4,
        /* .constant_byte_length = */ 8,
    };
}

static void set_binding(ggml_backend_hrx_loom_config_binding & binding, const char * name, const char * value) {
    std::memset(&binding, 0, sizeof(binding));
    std::strncpy(binding.name, name, sizeof(binding.name) - 1);
    std::strncpy(binding.value, value, sizeof(binding.value) - 1);
    binding.type = "i64";
}

static ggml_backend_hrx_loom_execution_plan make_plan(const ggml_backend_hrx_loom_catalog_entry * entry,
                                                      const char *                                 nelements,
                                                      const char *                                 workgroup_size_x) {
    ggml_backend_hrx_loom_execution_plan plan = {};
    plan.entry                                = entry;
    set_binding(plan.config_bindings[0], "nelements", nelements);
    set_binding(plan.config_bindings[1], "workgroup_size_x", workgroup_size_x);
    plan.config_binding_count = 2;
    return plan;
}

static std::string cache_key(const ggml_backend_hrx_loom_execution_plan & plan) {
    std::string key = ggml_backend_hrx_loom_cache_key("gfx1100", &plan);
    if (key.empty()) {
        std::fprintf(stderr, "failed to format cache key\n");
        std::abort();
    }
    return key;
}

static void expect_true(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        std::abort();
    }
}

int main() {
    static const unsigned char source_a[] = "kernel.def @a() { }";
    static const unsigned char source_b[] = "kernel.def @b() { }";

    const ggml_backend_hrx_loom_catalog_entry entry_a  = make_entry(source_a, sizeof(source_a) - 1);
    const ggml_backend_hrx_loom_catalog_entry entry_b  = make_entry(source_b, sizeof(source_b) - 1);
    const ggml_backend_hrx_loom_execution_plan plan_a0 = make_plan(&entry_a, "257", "256");
    const ggml_backend_hrx_loom_execution_plan plan_a1 = make_plan(&entry_a, "257", "256");
    const ggml_backend_hrx_loom_execution_plan plan_b0 = make_plan(&entry_a, "2048", "256");
    const ggml_backend_hrx_loom_execution_plan plan_c0 = make_plan(&entry_b, "257", "256");

    expect_true(cache_key(plan_a0) == cache_key(plan_a1), "identical plans must produce identical cache keys");
    expect_true(cache_key(plan_a0) != cache_key(plan_b0), "config changes must produce different cache keys");
    expect_true(cache_key(plan_a0) != cache_key(plan_c0), "source changes must produce different cache keys");
    expect_true(ggml_backend_hrx_loom_cache_key("gfx1100", nullptr).empty(), "null plan must produce empty cache key");
    return 0;
}
