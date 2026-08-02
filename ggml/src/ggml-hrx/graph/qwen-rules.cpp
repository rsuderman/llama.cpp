#include "qwen-rules.h"

#include <utility>

namespace ggml::hrx {
namespace {

using Kind = Transition::Kind;

static MatchState state(const char * name, enum ggml_op op, std::vector<Transition> transitions = {},
                        bool capture = true, std::vector<int64_t> shape = {}) {
    return { name, OpConstraint({ op }, GGML_TYPE_COUNT, -1, std::move(shape)), std::move(transitions), capture };
}

static MatchAutomaton attention_projection(bool capture_projections, bool capture_postprocess) {
    MatchAutomaton result;
    result.name = capture_projections ? "qwen.attention.qkv" : "qwen.attention.postprocess";
    result.require_internal_single_use = false;
    result.states = {
        state("prepared_activation", GGML_OP_MUL, {
            { Kind::OutputConsumer, 0, 1 }, { Kind::OutputConsumer, 0, 6 }, { Kind::OutputConsumer, 0, 13 },
        }, false),
        state("query_projection", GGML_OP_MUL_MAT, { { Kind::OutputConsumer, 0, 2 } }, capture_projections, { 4096 }),
        state("query_reshape", GGML_OP_RESHAPE, { { Kind::OutputConsumer, 0, 3 } }, capture_postprocess, { 128, 32 }),
        state("query_norm", GGML_OP_RMS_NORM, { { Kind::OutputConsumer, 0, 4 } }, capture_postprocess),
        state("query_scale", GGML_OP_MUL, { { Kind::OutputConsumer, 0, 5 } }, capture_postprocess),
        state("query_rope", GGML_OP_ROPE, {}, capture_postprocess),
        state("key_projection", GGML_OP_MUL_MAT, { { Kind::OutputConsumer, 0, 7 } }, capture_projections, { 512 }),
        state("key_reshape", GGML_OP_RESHAPE, { { Kind::OutputConsumer, 0, 8 } }, capture_postprocess, { 128, 4 }),
        state("key_norm", GGML_OP_RMS_NORM, { { Kind::OutputConsumer, 0, 9 } }, capture_postprocess),
        state("key_scale", GGML_OP_MUL, { { Kind::OutputConsumer, 0, 10 } }, capture_postprocess),
        state("key_rope", GGML_OP_ROPE, { { Kind::OutputConsumer, 0, 11 } }, capture_postprocess),
        state("key_cache_input", GGML_OP_VIEW, { { Kind::OutputConsumer, 0, 12 } }, capture_postprocess),
        state("key_cache_write", GGML_OP_SET_ROWS, {}, capture_postprocess),
        state("value_projection", GGML_OP_MUL_MAT, { { Kind::OutputConsumer, 0, 14 } }, capture_projections, { 512 }),
        state("value_reshape", GGML_OP_RESHAPE, { { Kind::OutputConsumer, 0, 15 } }, capture_postprocess, { 128, 4 }),
        state("value_cache_input", GGML_OP_VIEW, { { Kind::OutputConsumer, 0, 16 } }, capture_postprocess),
        state("value_cache_write", GGML_OP_SET_ROWS, {}, capture_postprocess),
    };
    return result;
}

static MatchAutomaton flash_attention() {
    MatchAutomaton result;
    result.name = "qwen.attention.flash";
    result.require_internal_single_use = false;
    result.states = {
        state("flash", GGML_OP_FLASH_ATTN_EXT, {
            { Kind::InputProducer, 0, 1 }, { Kind::InputProducer, 1, 3 }, { Kind::InputProducer, 2, 6 },
            { Kind::OutputConsumer, 0, 9 },
        }),
        state("query_permute", GGML_OP_PERMUTE, { { Kind::InputProducer, 0, 2 } }),
        state("query_view", GGML_OP_VIEW),
        state("key_permute", GGML_OP_PERMUTE, { { Kind::InputProducer, 0, 4 } }),
        state("key_view", GGML_OP_VIEW, { { Kind::StorageWriter, 0, 5 } }),
        state("key_cache_write", GGML_OP_SET_ROWS, {}, false),
        state("value_permute", GGML_OP_PERMUTE, { { Kind::InputProducer, 0, 7 } }),
        state("value_view", GGML_OP_VIEW, { { Kind::StorageWriter, 0, 8 } }),
        state("value_cache_write", GGML_OP_SET_ROWS, {}, false),
        state("result_reshape", GGML_OP_RESHAPE),
    };
    return result;
}

static MatchAutomaton router_top8() {
    MatchAutomaton result;
    result.name = "qwen.moe.router_top8";
    result.require_internal_single_use = false;
    result.states = {
        state("normalized_weights", GGML_OP_RESHAPE, { { Kind::InputProducer, 0, 1 } }),
        state("divide", GGML_OP_DIV, { { Kind::InputProducer, 0, 2 }, { Kind::InputProducer, 1, 6 } }),
        state("selected_weights", GGML_OP_RESHAPE, { { Kind::InputProducer, 0, 3 } }),
        state("gather", GGML_OP_GET_ROWS, { { Kind::InputProducer, 0, 4 }, { Kind::InputProducer, 1, 8 } }),
        state("probabilities_1d", GGML_OP_RESHAPE, { { Kind::InputProducer, 0, 5 } }),
        state("softmax", GGML_OP_SOFT_MAX),
        state("clamped_sum", GGML_OP_CLAMP, { { Kind::InputProducer, 0, 7 } }),
        state("weight_sum", GGML_OP_SUM_ROWS, { { Kind::InputProducer, 0, 2 } }),
        state("topk_view", GGML_OP_VIEW, { { Kind::InputProducer, 0, 9 } }),
        state("argsort", GGML_OP_ARGSORT, { { Kind::InputProducer, 0, 5 } }),
    };
    return result;
}

static MatchAutomaton routed_gate_up() {
    MatchAutomaton result;
    result.name = "qwen.moe.gate_up_swiglu";
    result.require_internal_single_use = false;
    result.states = {
        state("swiglu", GGML_OP_GLU, { { Kind::InputProducer, 0, 1 }, { Kind::InputProducer, 1, 2 } }),
        state("gate", GGML_OP_MUL_MAT_ID, { { Kind::InputProducer, 1, 3 } }),
        state("up", GGML_OP_MUL_MAT_ID, { { Kind::InputProducer, 1, 3 } }),
        state("activation_reshape", GGML_OP_RESHAPE),
    };
    return result;
}

static MatchAutomaton routed_down() {
    MatchAutomaton result;
    result.name = "qwen.moe.routed_down";
    result.require_internal_single_use = false;
    for (int i = 0; i < 7; ++i) {
        std::vector<Transition> transitions;
        if (i < 6) {
            transitions = { { Kind::InputProducer, 0, static_cast<size_t>(i + 1) },
                            { Kind::InputProducer, 1, static_cast<size_t>(14 - i) } };
        } else {
            transitions = { { Kind::InputProducer, 0, 7 }, { Kind::InputProducer, 1, 8 } };
        }
        result.states.push_back(state(("sum" + std::to_string(i)).c_str(), GGML_OP_ADD, std::move(transitions)));
    }
    for (int i = 0; i < 8; ++i) {
        result.states.push_back(state(("route" + std::to_string(i)).c_str(), GGML_OP_VIEW,
                                      { { Kind::InputProducer, 0, 15 } }));
    }
    result.states.push_back(state("weighted_down", GGML_OP_MUL, { { Kind::InputProducer, 0, 16 } }));
    result.states.push_back(state("down_projection", GGML_OP_MUL_MAT_ID, { { Kind::InputProducer, 1, 17 } }));
    result.states.push_back(state("swiglu_context", GGML_OP_GLU, {}, false));
    return result;
}

} // namespace

std::vector<FusionRule> canonical_qwen3_moe_rules() {
    MatchAutomaton prepare;
    prepare.name = "qwen.attention.prepare";
    prepare.states = {
        state("scale", GGML_OP_MUL, { { Kind::InputProducer, 0, 1 } }),
        state("norm", GGML_OP_RMS_NORM),
    };
    MatchAutomaton router_projection;
    router_projection.name = "qwen.moe.router_projection";
    router_projection.states = {
        { "projection", OpConstraint({ GGML_OP_MUL_MAT }, GGML_TYPE_F32, -1, { 128 },
              { { 0, { GGML_TYPE_F32 } } }), {} },
    };
    MatchAutomaton dense_output;
    dense_output.name = "qwen.attention.output_projection";
    dense_output.states = {
        { "projection", OpConstraint({ GGML_OP_MUL_MAT }, GGML_TYPE_F32, -1, { 2048 },
              { { 0, { GGML_TYPE_Q4_K } } }), {} },
    };
    MatchAutomaton dense_output_q6;
    dense_output_q6.name = "qwen.attention.output_projection_q6";
    dense_output_q6.states = {
        { "projection", OpConstraint({ GGML_OP_MUL_MAT }, GGML_TYPE_F32, -1, { 2048 },
              { { 0, { GGML_TYPE_Q6_K } } }), {} },
    };
    return {
        { attention_projection(false, true), { "qwen3_moe", "attention_postprocess_f32_f16", {} }, 900 },
        { attention_projection(true, false), { "qwen3_moe", "attention_qkv_quantized", {} }, 850 },
        { flash_attention(), { "qwen3_moe", "flash_attention_f32_f16_wmma", {} }, 800 },
        { prepare, { "qwen3_moe", "attention_prepare_quantized", {} }, 750 },
        { router_top8(), { "qwen3_moe", "router_top8_f32", {} }, 700 },
        { routed_gate_up(), { "qwen3_moe", "routed_gate_up_swiglu_q4k", {} }, 650 },
        { routed_down(), { "qwen3_moe", "routed_down_q6k", {} }, 600 },
        { router_projection, { "qwen3_moe", "router_projection_f32", {} }, 550 },
        { dense_output, { "qwen3_moe", "dense_linear_quantized_f16_wmma", {} }, 500 },
        { dense_output_q6, { "qwen3_moe", "dense_linear_q6k_f16_wmma", {} }, 500 },
    };
}

} // namespace ggml::hrx
