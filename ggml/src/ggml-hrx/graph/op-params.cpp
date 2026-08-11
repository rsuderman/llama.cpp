#include "op-params.h"

#include "ggml-impl.h"

#include <cmath>

namespace ggml::hrx {
namespace {

static bool nearly_equal(float lhs, float rhs) {
    return std::fabs(lhs - rhs) <= 1.0e-12f;
}

static bool rms_norm_params_equivalent(const OpParams & lhs, const OpParams & rhs) {
    const RmsNormParams * lhs_params = op_params_as<RmsNormParams>(lhs);
    const RmsNormParams * rhs_params = op_params_as<RmsNormParams>(rhs);
    return lhs_params != nullptr && rhs_params != nullptr && nearly_equal(lhs_params->eps, rhs_params->eps);
}

static bool flash_attn_ext_params_equivalent(const OpParams & lhs, const OpParams & rhs) {
    const FlashAttnExtParams * lhs_params = op_params_as<FlashAttnExtParams>(lhs);
    const FlashAttnExtParams * rhs_params = op_params_as<FlashAttnExtParams>(rhs);
    return lhs_params != nullptr && rhs_params != nullptr && nearly_equal(lhs_params->scale, rhs_params->scale) &&
           nearly_equal(lhs_params->max_bias, rhs_params->max_bias) &&
           nearly_equal(lhs_params->logit_softcap, rhs_params->logit_softcap) && lhs_params->prec == rhs_params->prec;
}

}  // namespace

OpParams import_op_params(const ggml_tensor & tensor) {
    switch (tensor.op) {
        case GGML_OP_RMS_NORM:
            return RmsNormParams{ ggml_get_op_params_f32(&tensor, 0) };
        case GGML_OP_FLASH_ATTN_EXT:
            return FlashAttnExtParams{
                ggml_get_op_params_f32(&tensor, 0),
                ggml_get_op_params_f32(&tensor, 1),
                ggml_get_op_params_f32(&tensor, 2),
                ggml_flash_attn_ext_get_prec(&tensor),
            };
        default:
            return std::monostate{};
    }
}

bool op_params_equivalent(ggml_op op, const OpParams & lhs, const OpParams & rhs) {
    switch (op) {
        case GGML_OP_RMS_NORM:
            return rms_norm_params_equivalent(lhs, rhs);
        case GGML_OP_FLASH_ATTN_EXT:
            return flash_attn_ext_params_equivalent(lhs, rhs);
        default:
            return lhs.index() == rhs.index();
    }
}

bool op_params_equivalent(ggml_op op, const OpParams & lhs, const ggml_tensor & rhs) {
    return op_params_equivalent(op, lhs, import_op_params(rhs));
}

}  // namespace ggml::hrx
