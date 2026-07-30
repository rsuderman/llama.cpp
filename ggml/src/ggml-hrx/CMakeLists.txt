message(STATUS "Using HRX backend")

find_package(hrx CONFIG REQUIRED)
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
option(GGML_HRX_HSACO_ROUTER "Enable the HRX HSACO route catalog" ON)
option(GGML_HRX_LOOM "Generate and compile the HRX Loom catalog" OFF)
set(GGML_HRX_EMBED_LIBRARY_DIRS "" CACHE STRING "Library directories to scan when GGML_HRX_EMBED_ROCM_LIBS=ON")
set(GGML_HRX_HSACO_TARGETS "gfx1100" CACHE STRING "AMDGPU targets used for the v0 HRX HSACO catalog")
set(GGML_HRX_LOOM_TARGETS "" CACHE STRING "Targets used for the v0 HRX Loom catalog; empty uses catalog metadata")

if (NOT GGML_HRX_ROCM_PATH AND DEFINED ENV{ROCM_PATH})
    set(GGML_HRX_ROCM_PATH "$ENV{ROCM_PATH}")
endif()

get_filename_component(GGML_HRX_COMPILER_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
find_program(GGML_HRX_HIPCC_EXECUTABLE
    NAMES hipcc
    HINTS
        "${GGML_HRX_ROCM_PATH}/bin"
        "${GGML_HRX_COMPILER_BIN_DIR}/../../bin"
    REQUIRED
)
find_program(GGML_HRX_CLANG_OFFLOAD_BUNDLER_EXECUTABLE
    NAMES clang-offload-bundler
    HINTS
        "${GGML_HRX_ROCM_PATH}/lib/llvm/bin"
        "${GGML_HRX_COMPILER_BIN_DIR}"
    REQUIRED
)

set(GGML_HRX_HSACO_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/hsaco-catalog")
set(GGML_HRX_HSACO_GENERATED_CPP "${GGML_HRX_HSACO_GENERATED_DIR}/ggml-hrx-hsaco-catalog.cpp")
set(GGML_HRX_HSACO_GENERATED_H "${GGML_HRX_HSACO_GENERATED_DIR}/ggml-hrx-hsaco-catalog.h")
set(GGML_HRX_HSACO_ROUTE_DISPATCH_CPP "${GGML_HRX_HSACO_GENERATED_DIR}/ggml-hrx-hsaco-route-dispatch.cpp")
set(GGML_HRX_HSACO_ROUTE_FILES
    routes/scale_f32.json
    routes/clamp_f32.json
    routes/add_f32.json
    routes/mul_f32.json
    routes/div_f32.json
    routes/sum_rows_f32.json
    routes/soft_max_f32.json
    routes/argsort_f32_i32.json
)
set(GGML_HRX_HSACO_ROUTE_OPS
    GGML_OP_SCALE
    GGML_OP_CLAMP
    GGML_OP_ADD
    GGML_OP_MUL
    GGML_OP_DIV
    GGML_OP_SUM_ROWS
    GGML_OP_SOFT_MAX
    GGML_OP_ARGSORT
)
set(GGML_HRX_HSACO_ROUTE_DEFS
    "${CMAKE_CURRENT_SOURCE_DIR}/hsaco-catalog/defs/add_f32.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/hsaco-catalog/defs/argsort_f32_i32.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/hsaco-catalog/defs/clamp_f32.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/hsaco-catalog/defs/div_f32.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/hsaco-catalog/defs/mul_f32.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/hsaco-catalog/defs/scale_f32.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/hsaco-catalog/defs/soft_max_f32.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/hsaco-catalog/defs/sum_rows_f32.json"
)
set(GGML_HRX_HSACO_ROUTE_JSONS)
set(GGML_HRX_HSACO_OP_ROUTER_CPPS)
set(GGML_HRX_HSACO_ROUTE_ARGS)
foreach(GGML_HRX_HSACO_ROUTE_FILE IN LISTS GGML_HRX_HSACO_ROUTE_FILES)
    list(APPEND GGML_HRX_HSACO_ROUTE_JSONS "${CMAKE_CURRENT_SOURCE_DIR}/hsaco-catalog/${GGML_HRX_HSACO_ROUTE_FILE}")
    list(APPEND GGML_HRX_HSACO_ROUTE_ARGS --route "${GGML_HRX_HSACO_ROUTE_FILE}")
endforeach()
foreach(GGML_HRX_HSACO_ROUTE_OP IN LISTS GGML_HRX_HSACO_ROUTE_OPS)
    string(REPLACE "GGML_OP_" "" GGML_HRX_HSACO_OP_NAME "${GGML_HRX_HSACO_ROUTE_OP}")
    string(TOLOWER "${GGML_HRX_HSACO_OP_NAME}" GGML_HRX_HSACO_OP_NAME_LOWER)
    set(GGML_HRX_HSACO_OP_ROUTER_CPP
        "${GGML_HRX_HSACO_GENERATED_DIR}/ggml-hrx-hsaco-router-${GGML_HRX_HSACO_OP_NAME_LOWER}.cpp")
    add_custom_command(
        OUTPUT
            "${GGML_HRX_HSACO_OP_ROUTER_CPP}"
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_hsaco_route_impl.py"
            --source-root "${CMAKE_CURRENT_SOURCE_DIR}/hsaco-catalog"
            ${GGML_HRX_HSACO_ROUTE_ARGS}
            --out "${GGML_HRX_HSACO_OP_ROUTER_CPP}"
            --op-router
            --op "${GGML_HRX_HSACO_ROUTE_OP}"
        DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/utils/hrx_route_schema.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/utils/hrx_route_emit.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_hsaco_route_impl.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/validate_hsaco_routes.py"
            ${GGML_HRX_HSACO_ROUTE_DEFS}
            ${GGML_HRX_HSACO_ROUTE_JSONS}
        VERBATIM
    )
    list(APPEND GGML_HRX_HSACO_OP_ROUTER_CPPS "${GGML_HRX_HSACO_OP_ROUTER_CPP}")
endforeach()

add_custom_command(
    OUTPUT
        "${GGML_HRX_HSACO_GENERATED_CPP}"
        "${GGML_HRX_HSACO_GENERATED_H}"
    COMMAND
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_hsaco_catalog.py"
        --source-root "${CMAKE_CURRENT_SOURCE_DIR}/hsaco-catalog"
        --build-root "${GGML_HRX_HSACO_GENERATED_DIR}"
        --out-cpp "${GGML_HRX_HSACO_GENERATED_CPP}"
        --out-h "${GGML_HRX_HSACO_GENERATED_H}"
        --hipcc "${GGML_HRX_HIPCC_EXECUTABLE}"
        --bundler "${GGML_HRX_CLANG_OFFLOAD_BUNDLER_EXECUTABLE}"
        --targets "${GGML_HRX_HSACO_TARGETS}"
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/utils/hrx_catalog_emit.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_hsaco_catalog.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/hsaco-catalog/metadata.json"
        "${CMAKE_CURRENT_SOURCE_DIR}/hsaco-catalog/defs/add_f32.json"
        "${CMAKE_CURRENT_SOURCE_DIR}/hsaco-catalog/defs/argsort_f32_i32.json"
        "${CMAKE_CURRENT_SOURCE_DIR}/hsaco-catalog/defs/clamp_f32.json"
        "${CMAKE_CURRENT_SOURCE_DIR}/hsaco-catalog/defs/div_f32.json"
        "${CMAKE_CURRENT_SOURCE_DIR}/hsaco-catalog/defs/mul_f32.json"
        "${CMAKE_CURRENT_SOURCE_DIR}/hsaco-catalog/defs/scale_f32.json"
        "${CMAKE_CURRENT_SOURCE_DIR}/hsaco-catalog/defs/soft_max_f32.json"
        "${CMAKE_CURRENT_SOURCE_DIR}/hsaco-catalog/defs/sum_rows_f32.json"
        "${CMAKE_CURRENT_SOURCE_DIR}/hsaco-catalog/routes/add_f32.json"
        "${CMAKE_CURRENT_SOURCE_DIR}/hsaco-catalog/routes/argsort_f32_i32.json"
        "${CMAKE_CURRENT_SOURCE_DIR}/hsaco-catalog/routes/clamp_f32.json"
        "${CMAKE_CURRENT_SOURCE_DIR}/hsaco-catalog/routes/div_f32.json"
        "${CMAKE_CURRENT_SOURCE_DIR}/hsaco-catalog/routes/mul_f32.json"
        "${CMAKE_CURRENT_SOURCE_DIR}/hsaco-catalog/routes/scale_f32.json"
        "${CMAKE_CURRENT_SOURCE_DIR}/hsaco-catalog/routes/soft_max_f32.json"
        "${CMAKE_CURRENT_SOURCE_DIR}/hsaco-catalog/routes/sum_rows_f32.json"
        "${CMAKE_CURRENT_SOURCE_DIR}/hsaco-catalog/sources/add_f32.hip"
        "${CMAKE_CURRENT_SOURCE_DIR}/hsaco-catalog/sources/argsort_f32_i32.hip"
        "${CMAKE_CURRENT_SOURCE_DIR}/hsaco-catalog/sources/clamp_f32.hip"
        "${CMAKE_CURRENT_SOURCE_DIR}/hsaco-catalog/sources/div_f32.hip"
        "${CMAKE_CURRENT_SOURCE_DIR}/hsaco-catalog/sources/mul_f32.hip"
        "${CMAKE_CURRENT_SOURCE_DIR}/hsaco-catalog/sources/scale_f32.hip"
        "${CMAKE_CURRENT_SOURCE_DIR}/hsaco-catalog/sources/soft_max_f32.hip"
        "${CMAKE_CURRENT_SOURCE_DIR}/hsaco-catalog/sources/sum_rows_f32.hip"
    VERBATIM
)

add_custom_target(ggml-hrx-hsaco-catalog
    DEPENDS
        "${GGML_HRX_HSACO_GENERATED_CPP}"
        "${GGML_HRX_HSACO_GENERATED_H}"
)

add_custom_command(
    OUTPUT
        "${GGML_HRX_HSACO_ROUTE_DISPATCH_CPP}"
    COMMAND
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_hsaco_route_impl.py"
        --source-root "${CMAKE_CURRENT_SOURCE_DIR}/hsaco-catalog"
        ${GGML_HRX_HSACO_ROUTE_ARGS}
        --out "${GGML_HRX_HSACO_ROUTE_DISPATCH_CPP}"
        --dispatcher
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/utils/hrx_route_schema.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/utils/hrx_route_emit.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_hsaco_route_impl.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/validate_hsaco_routes.py"
        ${GGML_HRX_HSACO_ROUTE_DEFS}
        ${GGML_HRX_HSACO_ROUTE_JSONS}
    VERBATIM
)

add_custom_target(ggml-hrx-hsaco-route-impls
    DEPENDS
        ${GGML_HRX_HSACO_OP_ROUTER_CPPS}
        "${GGML_HRX_HSACO_ROUTE_DISPATCH_CPP}"
)

set(GGML_HRX_LOOM_SOURCES)
set(GGML_HRX_LOOM_INCLUDE_DIRS)
if (GGML_HRX_LOOM)
    find_package(loomc CONFIG REQUIRED)

    set(GGML_HRX_LOOM_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/loom-catalog")
    set(GGML_HRX_LOOM_GENERATED_CPP "${GGML_HRX_LOOM_GENERATED_DIR}/ggml-hrx-loom-catalog.cpp")
    set(GGML_HRX_LOOM_GENERATED_H "${GGML_HRX_LOOM_GENERATED_DIR}/ggml-hrx-loom-catalog.h")
    set(GGML_HRX_LOOM_ROUTE_DISPATCH_CPP "${GGML_HRX_LOOM_GENERATED_DIR}/ggml-hrx-loom-route-dispatch.cpp")
    set(GGML_HRX_LOOM_ROUTE_FILES
        routes/add/f32/contiguous.json
        routes/argsort/f32_i32/contiguous_desc_wg256.json
    )
    set(GGML_HRX_LOOM_ROUTE_OPS
        GGML_OP_ADD
        GGML_OP_ARGSORT
    )
    set(GGML_HRX_LOOM_ROUTE_DEFS
        "${CMAKE_CURRENT_SOURCE_DIR}/loom-catalog/defs/add/f32/contiguous.json"
        "${CMAKE_CURRENT_SOURCE_DIR}/loom-catalog/defs/argsort/f32_i32/contiguous_desc_wg256.json"
    )
    set(GGML_HRX_LOOM_SOURCE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/loom-catalog/sources/add/f32/contiguous.loom"
        "${CMAKE_CURRENT_SOURCE_DIR}/loom-catalog/sources/argsort/f32_i32/contiguous_desc_wg256.loom"
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
endif()

ggml_add_backend_library(ggml-hrx
                         ggml-hrx.cpp
                         ggml-hrx-runtime-util.h
                         hsaco-catalog/ggml-hrx-hsaco-catalog-runtime.cpp
                         hsaco-catalog/ggml-hrx-hsaco-catalog-runtime.h
                         hsaco-catalog/ggml-hrx-hsaco-catalog-runtime-internal.h
                         "${GGML_HRX_HSACO_GENERATED_CPP}"
                         "${GGML_HRX_HSACO_GENERATED_H}"
                         ${GGML_HRX_HSACO_OP_ROUTER_CPPS}
                         "${GGML_HRX_HSACO_ROUTE_DISPATCH_CPP}"
                         ${GGML_HRX_LOOM_SOURCES}
                         ../../include/ggml-hrx.h
                        )

add_dependencies(ggml-hrx ggml-hrx-hsaco-catalog)
add_dependencies(ggml-hrx ggml-hrx-hsaco-route-impls)
if (GGML_HRX_LOOM)
    add_dependencies(ggml-hrx ggml-hrx-loom-catalog)
    add_dependencies(ggml-hrx ggml-hrx-loom-route-impls)
endif()

target_link_libraries(ggml-hrx PRIVATE
    hrx::hrx
)
if (GGML_HRX_LOOM)
    target_link_libraries(ggml-hrx PRIVATE
        loomc::loomc
    )
endif()
target_include_directories(ggml-hrx PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../vendor
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${GGML_HRX_HSACO_GENERATED_DIR}"
    ${GGML_HRX_LOOM_INCLUDE_DIRS}
)

if (GGML_HRX_LOOM)
    target_compile_definitions(ggml-hrx PRIVATE GGML_HRX_USE_LOOM)
endif()
if (NOT GGML_HRX_HSACO_ROUTER)
    target_compile_definitions(ggml-hrx PRIVATE GGML_HRX_DISABLE_HSACO_ROUTER)
endif()

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
