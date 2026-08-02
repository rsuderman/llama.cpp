find_package(hrx CONFIG REQUIRED)
find_package(loomc CONFIG REQUIRED)

add_library(ggml-hrx-graph STATIC
    graph/command-program.cpp
    graph/command-program.h
    graph/graph-ir.cpp
    graph/graph-ir.h
    graph/matcher.cpp
    graph/matcher.h
    graph/optimizer.cpp
    graph/optimizer.h
    graph/qwen-rules.cpp
    graph/qwen-rules.h
    graph/qwen-program.cpp
    graph/qwen-program.h
    graph/reactive-plan.cpp
    graph/reactive-plan.h
    graph/schedule.cpp
    graph/schedule.h
)
target_include_directories(ggml-hrx-graph PUBLIC graph PRIVATE .. ../../include ../../../vendor)
target_compile_features(ggml-hrx-graph PRIVATE cxx_std_17)
target_link_libraries(ggml-hrx-graph PRIVATE ggml-base)
set_target_properties(ggml-hrx-graph PROPERTIES POSITION_INDEPENDENT_CODE ON)

ggml_add_backend_library(ggml-hrx
    ggml-hrx.cpp
    loom-jit/ggml-hrx-loom-jit.cpp
    loom-jit/ggml-hrx-loom-jit.h
)
target_link_libraries(ggml-hrx PRIVATE hrx::hrx loomc::loomc)
target_link_libraries(ggml-hrx PRIVATE ggml-hrx-graph)
target_compile_definitions(ggml-hrx PRIVATE GGML_USE_HRX)
target_compile_definitions(ggml-hrx PRIVATE
    GGML_HRX_QWEN_CORPUS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kernels/qwen_moe")

add_executable(ggml-hrx-compile-kernel
    tools/compile-kernel.cpp
    loom-jit/ggml-hrx-loom-jit.cpp
)
target_link_libraries(ggml-hrx-compile-kernel PRIVATE hrx::hrx loomc::loomc)
target_include_directories(ggml-hrx-compile-kernel PRIVATE loom-jit)
target_compile_features(ggml-hrx-compile-kernel PRIVATE cxx_std_17)

add_executable(ggml-hrx-record-command-buffer-smoke tools/record-command-buffer-smoke.cpp)
target_link_libraries(ggml-hrx-record-command-buffer-smoke PRIVATE hrx::hrx)
target_compile_features(ggml-hrx-record-command-buffer-smoke PRIVATE cxx_std_17)

add_executable(ggml-hrx-dump-program tools/dump-program.cpp)
target_link_libraries(ggml-hrx-dump-program PRIVATE ggml-hrx-graph)
target_include_directories(ggml-hrx-dump-program PRIVATE graph .. ../../include ../../../vendor)
target_compile_features(ggml-hrx-dump-program PRIVATE cxx_std_17)
