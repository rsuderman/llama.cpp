#pragma once

#include "ggml.h"

#include <variant>

struct ggml_tensor;

namespace ggml::hrx {

struct RmsNormParams {
    float eps = 0.0f;
};

using OpParams = std::variant<std::monostate, RmsNormParams>;

template <typename T> const T * op_params_as(const OpParams & params) {
    return std::get_if<T>(&params);
}

OpParams import_op_params(const ggml_tensor & tensor);
bool     op_params_equivalent(ggml_op op, const OpParams & lhs, const OpParams & rhs);
bool     op_params_equivalent(ggml_op op, const OpParams & lhs, const ggml_tensor & rhs);

}  // namespace ggml::hrx
