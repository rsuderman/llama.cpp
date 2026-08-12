#pragma once

#include <cstdint>

namespace ggml::hrx {

constexpr bool is_qwen_supported_query_length(int64_t query_length) {
    return query_length >= 1 && query_length <= 2048;
}

constexpr bool is_qwen_decode_query_length(int64_t query_length) {
    return query_length == 1;
}

constexpr bool is_qwen_prefill_query_length(int64_t query_length) {
    return query_length > 1 && query_length <= 2048;
}

constexpr bool is_qwen_prefill_512_query_length(int64_t query_length) {
    return query_length == 512;
}

}  // namespace ggml::hrx
