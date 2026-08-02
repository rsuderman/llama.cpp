#include "loom-catalog/ggml-hrx-loom-catalog-runtime-internal.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static ggml_backend_hrx_loom_catalog_entry make_entry(const unsigned char *                      source_data,
                                                      size_t                                     source_size,
                                                      const ggml_backend_hrx_loom_source_entry * dependencies,
                                                      size_t                                     dependency_count) {
    return {
        /* .id                   = */ "add_f32",
        /* .op                   = */ "GGML_OP_ADD",
        /* .target               = */ "gfx1100",
        /* .source_name          = */ "sources/add_f32.loom",
        /* .source_data          = */ source_data,
        /* .source_size          = */ source_size,
        /* .source_format        = */ "loom-text",
        /* .symbol               = */ "hrx_add_f32",
        /* .dependencies         = */ dependencies,
        /* .dependency_count     = */ dependency_count,
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

static void make_plan(ggml_backend_hrx_loom_execution_plan &    plan,
                      ggml_backend_hrx_loom_config_binding *    storage,
                      const ggml_backend_hrx_loom_catalog_entry * entry,
                      const char *                               nelements,
                      const char *                               workgroup_size_x) {
    plan.config_storage = storage;
    ggml_backend_hrx_loom_reset_plan(&plan);
    plan.dispatches[0].entry = entry;
    set_binding(plan.dispatches[0].config_bindings[0], "nelements", nelements);
    set_binding(plan.dispatches[0].config_bindings[1], "workgroup_size_x", workgroup_size_x);
    plan.dispatches[0].config_binding_count = 2;
    plan.dispatch_count                     = 1;
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
    static const unsigned char                      source_a[]       = "kernel.def @a() { }";
    static const unsigned char                      source_b[]       = "kernel.def @b() { }";
    static const unsigned char                      dependency_a[]   = "kernel.func @helper_a() { }";
    static const unsigned char                      dependency_b[]   = "kernel.func @helper_b() { }";
    static const ggml_backend_hrx_loom_source_entry dependencies_a[] = {
        {
            /* .name   = */ "sources/helper.loom",
            /* .data   = */ dependency_a,
            /* .size   = */ sizeof(dependency_a) - 1,
            /* .format = */ "loom-text",
        },
    };
    static const ggml_backend_hrx_loom_source_entry dependencies_b[] = {
        {
            /* .name   = */ "sources/helper.loom",
            /* .data   = */ dependency_b,
            /* .size   = */ sizeof(dependency_b) - 1,
            /* .format = */ "loom-text",
        },
    };

    const ggml_backend_hrx_loom_catalog_entry  entry_a  = make_entry(source_a, sizeof(source_a) - 1, nullptr, 0);
    const ggml_backend_hrx_loom_catalog_entry  entry_b  = make_entry(source_b, sizeof(source_b) - 1, nullptr, 0);
    const ggml_backend_hrx_loom_catalog_entry  entry_d0 = make_entry(source_a, sizeof(source_a) - 1, dependencies_a, 1);
    const ggml_backend_hrx_loom_catalog_entry  entry_d1 = make_entry(source_a, sizeof(source_a) - 1, dependencies_b, 1);
    ggml_backend_hrx_loom_execution_plan plan_a0 = {};
    ggml_backend_hrx_loom_execution_plan plan_a1 = {};
    ggml_backend_hrx_loom_execution_plan plan_b0 = {};
    ggml_backend_hrx_loom_execution_plan plan_c0 = {};
    ggml_backend_hrx_loom_execution_plan plan_d0 = {};
    ggml_backend_hrx_loom_execution_plan plan_d1 = {};
    ggml_backend_hrx_loom_config_binding storage[6][GGML_BACKEND_HRX_LOOM_MAX_DISPATCHES *
                                                     GGML_BACKEND_HRX_LOOM_MAX_CONFIG_BINDINGS] = {};
    make_plan(plan_a0, storage[0], &entry_a, "257", "256");
    make_plan(plan_a1, storage[1], &entry_a, "257", "256");
    make_plan(plan_b0, storage[2], &entry_a, "2048", "256");
    make_plan(plan_c0, storage[3], &entry_b, "257", "256");
    make_plan(plan_d0, storage[4], &entry_d0, "257", "256");
    make_plan(plan_d1, storage[5], &entry_d1, "257", "256");

    expect_true(cache_key(plan_a0) == cache_key(plan_a1), "identical plans must produce identical cache keys");
    expect_true(cache_key(plan_a0) != cache_key(plan_b0), "config changes must produce different cache keys");
    expect_true(cache_key(plan_a0) != cache_key(plan_c0), "source changes must produce different cache keys");
    expect_true(cache_key(plan_a0) != cache_key(plan_d0), "dependency presence must produce different cache keys");
    expect_true(cache_key(plan_d0) != cache_key(plan_d1), "dependency changes must produce different cache keys");
    expect_true(ggml_backend_hrx_loom_cache_key("gfx1100", nullptr).empty(), "null plan must produce empty cache key");
    return 0;
}
