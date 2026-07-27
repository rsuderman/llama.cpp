message(STATUS "Using HRX backend")

find_package(hrx CONFIG REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

option(GGML_HRX_EMBED_ROCM_LIBS "Bundle HRX/ROCm runtime libraries next to the HRX backend" OFF)
set(GGML_HRX_EMBED_LIBRARY_DIRS "" CACHE STRING "Library directories to scan when GGML_HRX_EMBED_ROCM_LIBS=ON")
set(GGML_HRX_HSACO_TARGETS "gfx1100" CACHE STRING "AMDGPU targets used for the v0 HRX HSACO catalog")

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

ggml_add_backend_library(ggml-hrx
                         ggml-hrx.cpp
                         hsaco-catalog/ggml-hrx-hsaco-catalog-runtime.cpp
                         hsaco-catalog/ggml-hrx-hsaco-catalog-runtime.h
                         "${GGML_HRX_HSACO_GENERATED_CPP}"
                         "${GGML_HRX_HSACO_GENERATED_H}"
                         ../../include/ggml-hrx.h
                        )

add_dependencies(ggml-hrx ggml-hrx-hsaco-catalog)

target_link_libraries(ggml-hrx PRIVATE
    hrx::hrx
)
target_include_directories(ggml-hrx PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../vendor
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${GGML_HRX_HSACO_GENERATED_DIR}"
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
