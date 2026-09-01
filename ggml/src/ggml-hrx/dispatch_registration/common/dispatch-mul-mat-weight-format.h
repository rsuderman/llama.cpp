#pragma once

#include "ggml.h"

#include <cstdint>

namespace ggml::hrx {

enum class CommonMulMatWeightFormat {
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    IQ4_NL,
    IQ4_XS,
    Q8_0,
    Q8_1,
    F16,
    BF16,
    F32,
};

inline bool common_mul_mat_format_for_type(ggml_type type, CommonMulMatWeightFormat & format) {
    switch (type) {
        case GGML_TYPE_Q3_K:
            format = CommonMulMatWeightFormat::Q3K;
            return true;
        case GGML_TYPE_Q4_K:
            format = CommonMulMatWeightFormat::Q4K;
            return true;
        case GGML_TYPE_Q5_K:
            format = CommonMulMatWeightFormat::Q5K;
            return true;
        case GGML_TYPE_Q6_K:
            format = CommonMulMatWeightFormat::Q6K;
            return true;
        case GGML_TYPE_IQ4_NL:
            format = CommonMulMatWeightFormat::IQ4_NL;
            return true;
        case GGML_TYPE_IQ4_XS:
            format = CommonMulMatWeightFormat::IQ4_XS;
            return true;
        case GGML_TYPE_Q8_0:
            format = CommonMulMatWeightFormat::Q8_0;
            return true;
        case GGML_TYPE_Q8_1:
            format = CommonMulMatWeightFormat::Q8_1;
            return true;
        case GGML_TYPE_F16:
            format = CommonMulMatWeightFormat::F16;
            return true;
        case GGML_TYPE_BF16:
            format = CommonMulMatWeightFormat::BF16;
            return true;
        case GGML_TYPE_F32:
            format = CommonMulMatWeightFormat::F32;
            return true;
        default:
            return false;
    }
}

inline int64_t common_mul_mat_format_config_value(CommonMulMatWeightFormat format) {
    switch (format) {
        case CommonMulMatWeightFormat::Q3K:
            return 11;
        case CommonMulMatWeightFormat::Q4K:
            return 4;
        case CommonMulMatWeightFormat::Q5K:
            return 5;
        case CommonMulMatWeightFormat::Q6K:
            return 6;
        case CommonMulMatWeightFormat::IQ4_NL:
            return 20;
        case CommonMulMatWeightFormat::IQ4_XS:
            return 23;
        case CommonMulMatWeightFormat::Q8_0:
            return 80;
        case CommonMulMatWeightFormat::Q8_1:
            return 81;
        case CommonMulMatWeightFormat::F16:
            return 16;
        case CommonMulMatWeightFormat::BF16:
            return 30;
        case CommonMulMatWeightFormat::F32:
            return 32;
    }
    return 0;
}

inline bool common_mul_mat_dense_float_format(CommonMulMatWeightFormat format) {
    return format == CommonMulMatWeightFormat::F16 || format == CommonMulMatWeightFormat::BF16 ||
           format == CommonMulMatWeightFormat::F32;
}

inline bool common_mul_mat_alternate_format_for_type(ggml_type type, CommonMulMatWeightFormat & format) {
    switch (type) {
        case GGML_TYPE_Q8_1:
        case GGML_TYPE_F16:
        case GGML_TYPE_F32:
            return common_mul_mat_format_for_type(type, format);
        default:
            return false;
    }
}

}  // namespace ggml::hrx
