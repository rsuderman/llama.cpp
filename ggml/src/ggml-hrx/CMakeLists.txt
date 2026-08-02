find_package(hrx CONFIG REQUIRED)

add_library(ggml-hrx-graph STATIC
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

ggml_add_backend_library(ggml-hrx ggml-hrx.cpp)
target_link_libraries(ggml-hrx PRIVATE hrx::hrx)
target_link_libraries(ggml-hrx PRIVATE ggml-hrx-graph)
target_compile_definitions(ggml-hrx PRIVATE GGML_USE_HRX)
