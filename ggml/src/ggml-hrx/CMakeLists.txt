message(STATUS "Using HRX backend")

find_package(hrx CONFIG REQUIRED)
find_package(loomc CONFIG REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

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
        routes/generic/add/f32/contiguous.json
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
        routes/gfx1100/flash_attn_ext/f32_f16/tiled_decode_d128_h32_hkv4_kv512_40960_t1.json
        routes/gfx1100/flash_attn_ext/f32_f16/tiled_decode_d128_h32_hkv8_kv512_8192_t1.json
        routes/gfx1100/flash_attn_ext/f32_f16/tiled_prefill_d128_h32_hkv4_cap_kv40960_t512.json
        routes/gfx1100/flash_attn_ext/f32_f16/tiled_prefill_d128_h32_hkv8_cap_kv8192_t512.json
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
        routes/gfx1100/flash_attn_ext/f32_f16/perm0213_prefill_d128_h32_hkv4_kv40960_t512.json
    )
    set(GGML_HRX_LOOM_ROUTE_OPS
        GGML_OP_ADD
        GGML_OP_ARGSORT
        GGML_OP_CLAMP
        GGML_OP_CPY
        GGML_OP_DIV
        GGML_OP_FLASH_ATTN_EXT
        GGML_OP_GET_ROWS
        GGML_OP_GLU
        GGML_OP_MUL
        GGML_OP_MUL_MAT
        GGML_OP_MUL_MAT_ID
        GGML_OP_RMS_NORM
        GGML_OP_ROPE
        GGML_OP_SET_ROWS
        GGML_OP_SOFT_MAX
        GGML_OP_SUB
        GGML_OP_SUM_ROWS
    )
    file(GLOB_RECURSE GGML_HRX_LOOM_ROUTE_DEFS CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/loom-catalog/defs/*.json"
    )
    file(GLOB_RECURSE GGML_HRX_LOOM_SOURCE_FILES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/loom-catalog/sources/*.loom"
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
