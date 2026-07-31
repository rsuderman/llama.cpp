message(STATUS "Using HRX backend")

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(GGML_HRX_SOURCE_DIR "" CACHE PATH "HRX source tree containing libhrx and loom")
if (NOT GGML_HRX_SOURCE_DIR)
    message(FATAL_ERROR "GGML_HRX requires GGML_HRX_SOURCE_DIR; installed hrx/loomc packages are not used")
endif()

get_filename_component(GGML_HRX_SOURCE_DIR "${GGML_HRX_SOURCE_DIR}" ABSOLUTE)
if (NOT EXISTS "${GGML_HRX_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "GGML_HRX_SOURCE_DIR does not contain CMakeLists.txt: ${GGML_HRX_SOURCE_DIR}")
endif()
if (NOT EXISTS "${GGML_HRX_SOURCE_DIR}/libhrx/CMakeLists.txt")
    message(FATAL_ERROR "GGML_HRX_SOURCE_DIR does not contain libhrx: ${GGML_HRX_SOURCE_DIR}")
endif()
if (NOT IS_DIRECTORY "${GGML_HRX_SOURCE_DIR}/loom")
    message(FATAL_ERROR "GGML_HRX_SOURCE_DIR does not contain loom: ${GGML_HRX_SOURCE_DIR}")
endif()
if (NOT EXISTS "${GGML_HRX_SOURCE_DIR}/loom/binding/c/include/loomc/loomc.h")
    message(FATAL_ERROR "GGML_HRX_SOURCE_DIR does not contain loomc headers: ${GGML_HRX_SOURCE_DIR}")
endif()

include(ExternalProject)
set(GGML_HRX_SOURCE_BINARY_DIR "${CMAKE_BINARY_DIR}/ggml-hrx-deps")
set(GGML_HRX_SOURCE_HRX_LIBRARY "${GGML_HRX_SOURCE_BINARY_DIR}/libhrx/src/libhrx/${CMAKE_SHARED_LIBRARY_PREFIX}hrx${CMAKE_SHARED_LIBRARY_SUFFIX}")
set(GGML_HRX_SOURCE_LOOMC_LIBRARY "${GGML_HRX_SOURCE_BINARY_DIR}/loom/binding/c/${CMAKE_SHARED_LIBRARY_PREFIX}loomc${CMAKE_SHARED_LIBRARY_SUFFIX}")
set(GGML_HRX_SOURCE_CMAKE_ARGS
    "-DIREE_BUILD_TESTS=OFF"
    "-DIREE_BUILD_BENCHMARKS=OFF"
    "-DIREE_BUILD_DOCS=OFF"
    "-DIREE_BUILD_SAMPLES=OFF"
    "-DIREE_BUILD_PYTHON_BINDINGS=OFF"
    "-DIREE_HAL_DRIVER_DEFAULTS=OFF"
    "-DIREE_HAL_DRIVER_AMDGPU=ON"
    "-DIREE_HAL_DRIVER_LOCAL_SYNC=ON"
    "-DIREE_HAL_DRIVER_LOCAL_TASK=ON"
    "-DIREE_HAL_DRIVER_NULL=ON"
    "-DIREE_HAL_DRIVER_VULKAN=OFF"
    "-DIREE_HAL_DRIVER_HIP=OFF"
    "-DIREE_HAL_DRIVER_HIP_RCCL=OFF"
    "-DLOOM_BUILD=ON"
    "-DLOOM_TARGET_AMDGPU=ON"
    "-DLOOM_EXECUTE_DEFAULTS=ON"
    "-DLIBHRX_BUILD=ON"
    "-DBUILD_SHARED_LIBS=OFF"
)
if (CMAKE_BUILD_TYPE)
    list(APPEND GGML_HRX_SOURCE_CMAKE_ARGS "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
endif()
if (CMAKE_C_COMPILER)
    list(APPEND GGML_HRX_SOURCE_CMAKE_ARGS "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}")
endif()
if (CMAKE_CXX_COMPILER)
    list(APPEND GGML_HRX_SOURCE_CMAKE_ARGS "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")
endif()

message(STATUS "Building HRX and Loom from source: ${GGML_HRX_SOURCE_DIR}")
ExternalProject_Add(ggml-hrx-source-deps
    SOURCE_DIR "${GGML_HRX_SOURCE_DIR}"
    BINARY_DIR "${GGML_HRX_SOURCE_BINARY_DIR}"
    CMAKE_ARGS ${GGML_HRX_SOURCE_CMAKE_ARGS}
    BUILD_COMMAND "${CMAKE_COMMAND}" --build "<BINARY_DIR>" --target hrx loomc_shared
    INSTALL_COMMAND ""
    BUILD_ALWAYS TRUE
    BUILD_BYPRODUCTS
        "${GGML_HRX_SOURCE_HRX_LIBRARY}"
        "${GGML_HRX_SOURCE_LOOMC_LIBRARY}"
)

add_library(hrx::hrx SHARED IMPORTED GLOBAL)
set_target_properties(hrx::hrx PROPERTIES
    IMPORTED_LOCATION "${GGML_HRX_SOURCE_HRX_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${GGML_HRX_SOURCE_DIR}/libhrx/include")
add_dependencies(hrx::hrx ggml-hrx-source-deps)

add_library(loomc::loomc SHARED IMPORTED GLOBAL)
set_target_properties(loomc::loomc PROPERTIES
    IMPORTED_LOCATION "${GGML_HRX_SOURCE_LOOMC_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${GGML_HRX_SOURCE_DIR}/loom/binding/c/include")
add_dependencies(loomc::loomc ggml-hrx-source-deps)

get_target_property(GGML_HRX_IMPORTED_TARGET hrx::hrx IMPORTED)
if (GGML_HRX_IMPORTED_TARGET)
    get_target_property(GGML_HRX_IMPORTED_INCLUDES hrx::hrx INTERFACE_INCLUDE_DIRECTORIES)
    if (GGML_HRX_IMPORTED_INCLUDES)
        set(GGML_HRX_FILTERED_IMPORTED_INCLUDES)
        foreach(GGML_HRX_IMPORTED_INCLUDE IN LISTS GGML_HRX_IMPORTED_INCLUDES)
            set(GGML_HRX_IMPORTED_INCLUDE_PATH "${GGML_HRX_IMPORTED_INCLUDE}")
            if (GGML_HRX_IMPORTED_INCLUDE_PATH MATCHES "^\\$<BUILD_INTERFACE:(.*)>$")
                set(GGML_HRX_IMPORTED_INCLUDE_PATH "${CMAKE_MATCH_1}")
            endif()
            if (EXISTS "${GGML_HRX_IMPORTED_INCLUDE_PATH}")
                list(APPEND GGML_HRX_FILTERED_IMPORTED_INCLUDES "${GGML_HRX_IMPORTED_INCLUDE}")
            endif()
        endforeach()
        set_target_properties(hrx::hrx PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${GGML_HRX_FILTERED_IMPORTED_INCLUDES}")
    endif()
endif()

foreach(GGML_HRX_REQUIRED_TARGET hrx::hrx loomc::loomc)
    if (NOT TARGET "${GGML_HRX_REQUIRED_TARGET}")
        message(FATAL_ERROR "Missing required HRX dependency target: ${GGML_HRX_REQUIRED_TARGET}")
    endif()
endforeach()

option(GGML_HRX_EMBED_ROCM_LIBS "Bundle HRX/ROCm runtime libraries next to the HRX backend" OFF)
set(GGML_HRX_EMBED_LIBRARY_DIRS "" CACHE STRING "Library directories to scan when GGML_HRX_EMBED_ROCM_LIBS=ON")
set(GGML_HRX_LOOM_TARGETS "" CACHE STRING "Targets used for the HRX Loom catalog; empty uses catalog metadata")

if (NOT GGML_HRX_ROCM_PATH AND DEFINED ENV{ROCM_PATH})
    set(GGML_HRX_ROCM_PATH "$ENV{ROCM_PATH}")
endif()

set(GGML_HRX_GENERATED_ROUTER_COMPILE_OPTIONS)
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    list(APPEND GGML_HRX_GENERATED_ROUTER_COMPILE_OPTIONS -Wno-unused-variable)
endif()

set(GGML_HRX_LOOM_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/loom-catalog")
set(GGML_HRX_LOOM_GENERATED_CPP "${GGML_HRX_LOOM_GENERATED_DIR}/ggml-hrx-loom-catalog.cpp")
set(GGML_HRX_LOOM_GENERATED_H "${GGML_HRX_LOOM_GENERATED_DIR}/ggml-hrx-loom-catalog.h")
set(GGML_HRX_LOOM_ROUTE_DISPATCH_CPP "${GGML_HRX_LOOM_GENERATED_DIR}/ggml-hrx-loom-route-dispatch.cpp")
set(GGML_HRX_LOOM_ROUTE_FILES
        routes/gfx1151/scale/f32/generic_wg256.json
        routes/gfx1151/scale/f32/flat_large_wg256.json
        routes/gfx1151/add/f32/rhs_row_broadcast_2d_r64_1048576_wg256.json
        routes/gfx1151/mul/f32/rhs_row_broadcast_2d_r64_1048576_wg256.json
        routes/gfx1151/unary/f32/sigmoid_wide_wg256.json
        routes/gfx1151/unary/f32/softplus_wide_wg256.json
        routes/gfx1151/unary/f32/silu_wide_wg256.json
        routes/gfx1151/unary/f32/exp_wide_wg256.json
        routes/gfx1151/unary/f32/neg_wide_wg256.json
        routes/gfx1151/get_rows/f32/n2048_r1_64_wg256.json
        routes/gfx1151/get_rows/q8_0_f32/n2048_rany.json
        routes/gfx1151/get_rows/f32/wide_wg256.json
        routes/gfx1151/cont/f32/strided_wide_wg256.json
        routes/gfx1151/copy/f32_f16/generic_wg256.json
        routes/gfx1151/copy/f32/strided_flatten_wg256.json
        routes/gfx1151/concat/f32/dim0_strided_tiled_wg256.json
        routes/gfx1151/concat/f32/dim1_contiguous_flat_tiled_wg256.json
        routes/gfx1151/set_rows/f32_f16/generic.json
        routes/gfx1151/swiglu/f32/wide_split.json
        routes/gfx1151/l2_norm/f32/strided_wide_wg256.json
        routes/gfx1151/soft_max/f32/nomask_wide_wg256.json
        routes/gfx1151/mul_mat/f32_f32/mul_mat_f32_f32_tiled.json
        routes/gfx1151/mul_mat/f32_f32/mul_mat_f32_f32_tiled_ncol.json
        routes/gfx1151/mul_mat/f32_f32/wide_mul_mat_f32_f32.json
        routes/gfx1151/mul_mat/q8_0_f32/mul_mat_q8_0_f32_packed_loop_k32_8192_c1_16_wg64.json
        routes/gfx1151/mul_mat/q8_0_f32/mul_mat_q8_0_f32_wmmai8.json
        routes/gfx1151/mul_mat/q8_0_f32/mul_mat_q8_0_f32_packed_decode_k2048_wg256_scfunroll2.json
        routes/gfx1151/mul_mat/q8_0_f32/mul_mat_q8_0_f32_packed_decode_k4096_wg256_scfunroll4.json
        routes/gfx1151/flash_attn_ext/f32_f16/wmma_f32.json
        routes/gfx1151/flash_attn_ext/f32_f16/wave256.json
        routes/gfx1151/ssm_conv/f32/wide_wg256.json
        routes/gfx1151/ssm_conv/f32/channels_first_wg256.json
        routes/gfx1151/gated_delta_net/f32/sv128_scalar_gate.json
        routes/gfx1151/rms_norm/f32/strided_wide_wg256.json
        routes/gfx1151/rope/f32/multi_section_wg256.json
        routes/gfx1151/mul_mat_id/q4_k_f32/mul_mat_id_q4_k_f32_mmq_gfx1151_wg256_pre_tbl_token.json
        routes/gfx1151/mul_mat_id/q4_k_f32/mul_mat_id_q4_k_f32_mmq_gfx1151_wg256_pre_tbl.json
        routes/gfx1151/mul_mat_id/q5_k_f32/mul_mat_id_q5_k_f32_mmq_gfx1151_wg256_pre_tbl_token.json
        routes/gfx1151/mul_mat_id/q5_k_f32/mul_mat_id_q5_k_f32_mmq_gfx1151_wg256_pre_tbl.json
        routes/gfx1151/mul_mat_id/q5_k_f32/mul_mat_id_q5_k_f32_mmq_gfx1151_wg256_tbl_down_group4.json
        routes/gfx1151/mul_mat_id/q6_k_f32/mul_mat_id_q6_k_f32_mmq_gfx1151_wg256_tbl.json
        routes/gfx1151/mul_mat_id/q6_k_f32/mul_mat_id_q6_k_f32_mmq_gfx1151_wg256_tbl_selected.json
        routes/generic/add/f32/contiguous.json
        routes/generic/add/f32/transient_demo.json
        routes/generic/argsort/f32_i32/contiguous_wg256.json
        routes/generic/sum_rows/f32/contiguous_4d.json
        routes/generic/rope/f32/neox.json
        routes/generic/rms_norm/f32/contiguous_4d.json
        routes/generic/rms_norm_mul/f32/contiguous_4d.json
        routes/generic/clamp/f32/contiguous_4d.json
        routes/generic/add/f16/contiguous.json
        routes/generic/add/f16/non_contiguous_4d.json
        routes/generic/add/f32/non_contiguous_4d.json
        routes/generic/div/f16/contiguous.json
        routes/generic/div/f16/non_contiguous_4d.json
        routes/generic/div/f32/contiguous.json
        routes/generic/div/f32/non_contiguous_4d.json
        routes/generic/mul/f16/contiguous.json
        routes/generic/mul/f16/non_contiguous_4d.json
        routes/generic/mul/f32/contiguous.json
        routes/generic/mul/f32/graph_transient_mul_add.json
        routes/generic/mul/f32/non_contiguous_4d.json
        routes/generic/sub/f16/contiguous.json
        routes/generic/sub/f16/non_contiguous_4d.json
        routes/generic/sub/f32/contiguous.json
        routes/generic/sub/f32/non_contiguous_4d.json
        routes/generic/copy/bf16_bf16/contiguous_1d.json
        routes/generic/copy/f16_f16/contiguous_1d.json
        routes/generic/copy/f32_f32/contiguous_1d.json
        routes/generic/copy/i32_i32/contiguous_1d.json
        routes/generic/copy/bf16_f16/contiguous_1d.json
        routes/generic/copy/bf16_f32/contiguous_1d.json
        routes/generic/copy/f16_bf16/contiguous_1d.json
        routes/generic/copy/f16_f32/contiguous_1d.json
        routes/generic/copy/f32_bf16/contiguous_1d.json
        routes/generic/copy/f32_f16/contiguous_1d.json
        routes/generic/get_rows/f32/contiguous_rows_4d.json
        routes/generic/get_rows/f32/outer_indexed_rows_4d.json
        routes/generic/get_rows/q4_k_f32/contiguous_rows_4d.json
        routes/generic/set_rows/f32_f16/cont.json
        routes/generic/swiglu/f32/split_strided_4d.json
        routes/gfx1100/swiglu/f32/qwen3_moe_routed_swiglu_f32.json
        routes/gfx1100/qwen_moe/qwen3_moe/router_projection_top8_fused_decode_f32.json
        routes/gfx1100/qwen_moe/qwen3_moe/router_projection_f32_four_row_wave32_bringup_workaround.json
        routes/gfx1100/qwen_moe/qwen3_moe/router_top8_f32_stride_bringup_workaround.json
        routes/gfx1100/qwen_moe/qwen3_moe/dense_linear_q4k_q8_1_x4_next_q8.json
        routes/gfx1100/qwen_moe/qwen3_moe/routed_down_q4k_q8_1_x4_next_q8.json
        routes/gfx1100/qwen_moe/qwen3_moe/routed_down_q6k_q8_1_x4_next_q8.json
        routes/gfx1100/qwen_moe/qwen3_moe/dense_linear_q4k_f16_wmma.json
        routes/gfx1100/qwen_moe/qwen3_moe/dense_linear_q6k_f16_wmma.json
        routes/gfx1100/qwen_moe/qwen3_moe/routed_gate_up_swiglu_q4k_q8_1_x4_next_q8.json
        routes/gfx1100/qwen_moe/qwen3_moe/routed_gate_up_swiglu_q4k_f16_wmma_prefill_512_partition_fused.json
        routes/gfx1100/qwen_moe/qwen3_moe/routed_gate_up_swiglu_q4k_f16_wmma.json
        routes/gfx1100/qwen_moe/qwen3_moe/routed_down_q4k_f16_wmma_next_rmsnorm.json
        routes/gfx1100/qwen_moe/qwen3_moe/routed_down_q6k_f16_wmma_next_rmsnorm.json
        routes/gfx1100/qwen_moe/qwen3_moe/routed_down_q4k_f16_wmma_residual.json
        routes/gfx1100/qwen_moe/qwen3_moe/routed_down_q6k_f16_wmma_residual.json
        routes/gfx1100/qwen_moe/qwen3_moe/attention_qkv_postprocess_fused_decode_q4.json
        routes/gfx1100/qwen_moe/qwen3_moe/attention_qkv_postprocess_fused_decode_q6.json
        routes/gfx1100/qwen_moe/qwen3_moe/attention_postprocess_f32_f16.json
        routes/gfx1100/qwen_moe/qwen3_moe/flash_attention_f32_f16_wmma.json
        routes/gfx1100/qwen_moe/qwen3_moe/flash_attention_decode_split_f32_f16_wmma_short.json
        routes/gfx1100/qwen_moe/qwen3_moe/flash_attention_decode_split_f32_f16_wmma_long.json
        routes/gfx1100/flash_attn_ext/f32_f16/tiled_decode_d128_h32_hkv4_kv512_40960_t1.json
        routes/gfx1100/flash_attn_ext/f32_f16/tiled_decode_d128_h32_hkv8_kv512_8192_t1.json
        routes/gfx1100/flash_attn_ext/f32_f16/fallback_decode_plain_mask_t1.json
        routes/generic/mul_mat/q8_0_f32/contiguous_c4_4d.json
        routes/generic/mul_mat/q8_0_f32/contiguous_c1_4d.json
        routes/generic/mul_mat/q8_0_f32/contiguous_4d.json
        routes/generic/mul_mat/q4_k_f32/skinny_4d.json
        routes/generic/mul_mat/q4_k_f32/contiguous_c4_4d.json
        routes/generic/mul_mat/q4_k_f32/contiguous_4d.json
        routes/generic/mul_mat/q5_k_f32/contiguous_4d.json
        routes/generic/mul_mat/q6_k_f32/skinny_4d.json
        routes/generic/mul_mat/q6_k_f32/contiguous_c4_4d.json
        routes/generic/mul_mat/q6_k_f32/contiguous_4d.json
        routes/generic/mul_mat/f32_f32/skinny_4d.json
        routes/gfx1100/mul_mat/f32_f32/large_tiled_8x8_gfx1100_4d.json
        routes/generic/mul_mat/f32_f32/large_tiled_4d.json
        routes/generic/mul_mat/f32_f32/contiguous_4d.json
        routes/generic/mul_mat/f16_f32/contiguous_4d.json
        routes/gfx1100/mul_mat/f16_f16/contiguous_m64n64_gfx1100_4d.json
        routes/gfx1100/mul_mat/f16_f16/contiguous_c1_gfx1100_4d.json
        routes/generic/mul_mat/f16_f16/contiguous_4d.json
        routes/generic/soft_max/f32/skinny_contiguous_4d.json
        routes/generic/soft_max/f32/contiguous_4d.json
        routes/generic/mul_mat_id/q4_k_f32/expert_planes_4d.json
        routes/generic/mul_mat_id/q4_k_f32/expert_planes_src1_broadcast_4d.json
        routes/generic/mul_mat_id/q6_k_f32/expert_planes_4d.json
        routes/generic/mul_mat_id/q6_k_f32/expert_planes_src1_broadcast_4d.json
        routes/gfx1100/flash_attn_ext/f32_f16/perm0213_decode_d128_h32_hkv4_kv40960.json
    )
    set(GGML_HRX_LOOM_ROUTE_OPS
        GGML_OP_ADD
        GGML_OP_ARGSORT
        GGML_OP_CLAMP
        GGML_OP_CONCAT
        GGML_OP_CONT
        GGML_OP_CPY
        GGML_OP_DIV
        GGML_OP_FLASH_ATTN_EXT
        GGML_OP_GATED_DELTA_NET
        GGML_OP_GET_ROWS
        GGML_OP_GLU
        GGML_OP_L2_NORM
        GGML_OP_MUL
        GGML_OP_MUL_MAT
        GGML_OP_MUL_MAT_ID
        GGML_OP_RMS_NORM
        GGML_OP_ROPE
        GGML_OP_SCALE
        GGML_OP_SET_ROWS
        GGML_OP_SOFT_MAX
        GGML_OP_SUB
        GGML_OP_SUM_ROWS
        GGML_OP_SSM_CONV
        GGML_OP_UNARY
    )
    file(GLOB_RECURSE GGML_HRX_LOOM_ROUTE_DEFS CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/loom-catalog/defs/*.json"
    )
    file(GLOB_RECURSE GGML_HRX_LOOM_SOURCE_FILES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/loom-catalog/sources/*.loom"
    )
    file(GLOB_RECURSE GGML_HRX_LOOM_STORAGE_FILES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/loom-catalog/storage-transforms/*.json"
    )
    set(GGML_HRX_LOOM_ROUTE_JSONS)
    set(GGML_HRX_LOOM_OP_ROUTER_CPPS)
    set(GGML_HRX_LOOM_ROUTE_ARGS)
    foreach(GGML_HRX_LOOM_ROUTE_FILE IN LISTS GGML_HRX_LOOM_ROUTE_FILES)
        list(APPEND GGML_HRX_LOOM_ROUTE_JSONS "${CMAKE_CURRENT_SOURCE_DIR}/loom-catalog/${GGML_HRX_LOOM_ROUTE_FILE}")
        list(APPEND GGML_HRX_LOOM_ROUTE_ARGS --route "${GGML_HRX_LOOM_ROUTE_FILE}")
    endforeach()
    foreach(GGML_HRX_LOOM_ROUTE_OP IN LISTS GGML_HRX_LOOM_ROUTE_OPS)
        string(REPLACE "GGML_OP_" "" GGML_HRX_LOOM_OP_NAME "${GGML_HRX_LOOM_ROUTE_OP}")
        string(TOLOWER "${GGML_HRX_LOOM_OP_NAME}" GGML_HRX_LOOM_OP_NAME_LOWER)
        set(GGML_HRX_LOOM_OP_ROUTER_CPP
            "${GGML_HRX_LOOM_GENERATED_DIR}/ggml-hrx-loom-router-${GGML_HRX_LOOM_OP_NAME_LOWER}.cpp")
        add_custom_command(
            OUTPUT
                "${GGML_HRX_LOOM_OP_ROUTER_CPP}"
            COMMAND
                "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_loom_route_impl.py"
                --source-root "${CMAKE_CURRENT_SOURCE_DIR}/loom-catalog"
                ${GGML_HRX_LOOM_ROUTE_ARGS}
                --out "${GGML_HRX_LOOM_OP_ROUTER_CPP}"
                --op-router
                --op "${GGML_HRX_LOOM_ROUTE_OP}"
            DEPENDS
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/utils/hrx_route_schema.py"
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/utils/hrx_route_emit.py"
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_loom_route_impl.py"
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/validate_loom_routes.py"
                "${CMAKE_CURRENT_SOURCE_DIR}/loom-catalog/metadata.json"
                ${GGML_HRX_LOOM_ROUTE_DEFS}
                ${GGML_HRX_LOOM_ROUTE_JSONS}
            VERBATIM
        )
        list(APPEND GGML_HRX_LOOM_OP_ROUTER_CPPS "${GGML_HRX_LOOM_OP_ROUTER_CPP}")
    endforeach()
    set(GGML_HRX_LOOM_TARGET_ARGS)
    if (GGML_HRX_LOOM_TARGETS)
        list(APPEND GGML_HRX_LOOM_TARGET_ARGS --targets "${GGML_HRX_LOOM_TARGETS}")
    endif()

    add_custom_command(
        OUTPUT
            "${GGML_HRX_LOOM_GENERATED_CPP}"
            "${GGML_HRX_LOOM_GENERATED_H}"
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_loom_catalog.py"
            --source-root "${CMAKE_CURRENT_SOURCE_DIR}/loom-catalog"
            --out-cpp "${GGML_HRX_LOOM_GENERATED_CPP}"
            --out-h "${GGML_HRX_LOOM_GENERATED_H}"
            ${GGML_HRX_LOOM_TARGET_ARGS}
        DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/utils/hrx_catalog_emit.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/utils/hrx_route_schema.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_loom_catalog.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/validate_loom_routes.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/loom-catalog/metadata.json"
            ${GGML_HRX_LOOM_ROUTE_DEFS}
            ${GGML_HRX_LOOM_ROUTE_JSONS}
            ${GGML_HRX_LOOM_SOURCE_FILES}
            ${GGML_HRX_LOOM_STORAGE_FILES}
        VERBATIM
    )

    add_custom_target(ggml-hrx-loom-catalog
        DEPENDS
            "${GGML_HRX_LOOM_GENERATED_CPP}"
            "${GGML_HRX_LOOM_GENERATED_H}"
    )

    add_custom_command(
        OUTPUT
            "${GGML_HRX_LOOM_ROUTE_DISPATCH_CPP}"
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_loom_route_impl.py"
            --source-root "${CMAKE_CURRENT_SOURCE_DIR}/loom-catalog"
            ${GGML_HRX_LOOM_ROUTE_ARGS}
            --out "${GGML_HRX_LOOM_ROUTE_DISPATCH_CPP}"
            --dispatcher
        DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/utils/hrx_route_schema.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/utils/hrx_route_emit.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_loom_route_impl.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/validate_loom_routes.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/loom-catalog/metadata.json"
            ${GGML_HRX_LOOM_ROUTE_DEFS}
            ${GGML_HRX_LOOM_ROUTE_JSONS}
        VERBATIM
    )

    add_custom_target(ggml-hrx-loom-route-impls
        DEPENDS
            ${GGML_HRX_LOOM_OP_ROUTER_CPPS}
            "${GGML_HRX_LOOM_ROUTE_DISPATCH_CPP}"
    )
    if (GGML_HRX_GENERATED_ROUTER_COMPILE_OPTIONS)
        set_source_files_properties(
            ${GGML_HRX_LOOM_OP_ROUTER_CPPS}
            "${GGML_HRX_LOOM_ROUTE_DISPATCH_CPP}"
            PROPERTIES COMPILE_OPTIONS "${GGML_HRX_GENERATED_ROUTER_COMPILE_OPTIONS}"
        )
    endif()

    list(APPEND GGML_HRX_LOOM_SOURCES
        loom-catalog/ggml-hrx-loom-catalog-runtime.cpp
        loom-catalog/ggml-hrx-loom-catalog-runtime.h
        loom-catalog/ggml-hrx-loom-catalog-runtime-internal.h
        loom-catalog/ggml-hrx-loom-storage.cpp
        loom-catalog/ggml-hrx-loom-storage.h
        "${GGML_HRX_LOOM_GENERATED_CPP}"
        "${GGML_HRX_LOOM_GENERATED_H}"
        ${GGML_HRX_LOOM_OP_ROUTER_CPPS}
        "${GGML_HRX_LOOM_ROUTE_DISPATCH_CPP}"
    )
    list(APPEND GGML_HRX_LOOM_INCLUDE_DIRS
        "${GGML_HRX_LOOM_GENERATED_DIR}"
    )

ggml_add_backend_library(ggml-hrx
                         ggml-hrx.cpp
                         ggml-hrx-runtime-util.h
                         ${GGML_HRX_LOOM_SOURCES}
                         ../../include/ggml-hrx.h
                        )

add_dependencies(ggml-hrx ggml-hrx-loom-catalog)
add_dependencies(ggml-hrx ggml-hrx-loom-route-impls)

target_link_libraries(ggml-hrx PRIVATE
    hrx::hrx
    loomc::loomc
)
target_include_directories(ggml-hrx PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../vendor
    "${CMAKE_CURRENT_SOURCE_DIR}"
    ${GGML_HRX_LOOM_INCLUDE_DIRS}
)

if (GGML_HRX_EMBED_ROCM_LIBS)
    if (NOT CMAKE_SYSTEM_NAME MATCHES "Linux")
        message(FATAL_ERROR "GGML_HRX_EMBED_ROCM_LIBS is currently implemented for Linux only")
    endif()

    if (GGML_BACKEND_DL)
        if (GGML_BACKEND_DIR)
            set(GGML_HRX_EMBED_INSTALL_DIR "${GGML_BACKEND_DIR}")
        else()
            set(GGML_HRX_EMBED_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}")
        endif()
    else()
        set(GGML_HRX_EMBED_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}")
    endif()

    set(GGML_HRX_EMBED_SEARCH_DIRS ${GGML_HRX_EMBED_LIBRARY_DIRS})
    if (GGML_HRX_ROCM_PATH)
        list(APPEND GGML_HRX_EMBED_SEARCH_DIRS "${GGML_HRX_ROCM_PATH}/lib")
    endif()
    if (GGML_HRX_EMBED_SEARCH_DIRS)
        list(REMOVE_DUPLICATES GGML_HRX_EMBED_SEARCH_DIRS)
    endif()

    function(ggml_hrx_collect_bundle_paths OUT_VAR REQUIRED)
        set(GGML_HRX_COLLECTED_PATHS)
        set(GGML_HRX_MISSING_PATTERNS)
        foreach(GGML_HRX_PATTERN IN LISTS ARGN)
            set(GGML_HRX_PATTERN_MATCHES)
            foreach(GGML_HRX_SEARCH_DIR IN LISTS GGML_HRX_EMBED_SEARCH_DIRS)
                file(GLOB GGML_HRX_DIR_MATCHES "${GGML_HRX_SEARCH_DIR}/${GGML_HRX_PATTERN}")
                list(APPEND GGML_HRX_PATTERN_MATCHES ${GGML_HRX_DIR_MATCHES})
            endforeach()
            if (GGML_HRX_PATTERN_MATCHES)
                list(APPEND GGML_HRX_COLLECTED_PATHS ${GGML_HRX_PATTERN_MATCHES})
            elseif(REQUIRED)
                list(APPEND GGML_HRX_MISSING_PATTERNS "${GGML_HRX_PATTERN}")
            endif()
        endforeach()
        if (GGML_HRX_MISSING_PATTERNS)
            message(FATAL_ERROR
                "GGML_HRX_EMBED_ROCM_LIBS could not find required libraries matching: "
                "${GGML_HRX_MISSING_PATTERNS}. Searched: ${GGML_HRX_EMBED_SEARCH_DIRS}")
        endif()
        if (GGML_HRX_COLLECTED_PATHS)
            list(REMOVE_DUPLICATES GGML_HRX_COLLECTED_PATHS)
        endif()
        set(${OUT_VAR} "${GGML_HRX_COLLECTED_PATHS}" PARENT_SCOPE)
    endfunction()

    ggml_hrx_collect_bundle_paths(GGML_HRX_EMBED_MAIN_LIBS TRUE
        "libhrx.so*"
        "libloomc.so*"
        "libhsa-runtime64.so*"
        "libhsa-amd-aqlprofile64.so*"
        "librocprofiler-register.so*"
        "libomp.so*"
    )
    ggml_hrx_collect_bundle_paths(GGML_HRX_EMBED_SYSDEP_LIBS FALSE
        "rocm_sysdeps/lib/*.so*"
    )

    set_target_properties(ggml-hrx PROPERTIES
        BUILD_RPATH "$ORIGIN;$ORIGIN/rocm_sysdeps/lib"
        INSTALL_RPATH "$ORIGIN;$ORIGIN/rocm_sysdeps/lib"
    )

    add_custom_command(TARGET ggml-hrx POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E make_directory "$<TARGET_FILE_DIR:ggml-hrx>"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            ${GGML_HRX_EMBED_MAIN_LIBS}
            "$<TARGET_FILE_DIR:ggml-hrx>"
        VERBATIM
    )
    install(FILES ${GGML_HRX_EMBED_MAIN_LIBS}
        DESTINATION "${GGML_HRX_EMBED_INSTALL_DIR}")

    if (GGML_HRX_EMBED_SYSDEP_LIBS)
        add_custom_command(TARGET ggml-hrx POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E make_directory "$<TARGET_FILE_DIR:ggml-hrx>/rocm_sysdeps/lib"
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                ${GGML_HRX_EMBED_SYSDEP_LIBS}
                "$<TARGET_FILE_DIR:ggml-hrx>/rocm_sysdeps/lib"
            VERBATIM
        )
        install(FILES ${GGML_HRX_EMBED_SYSDEP_LIBS}
            DESTINATION "${GGML_HRX_EMBED_INSTALL_DIR}/rocm_sysdeps/lib")
    endif()
endif()
