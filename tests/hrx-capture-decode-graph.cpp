#include "llama.h"
#include "llama-batch.h"
#include "llama-context.h"
#include "llama-kv-cache.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s MODEL prefill-512|decode-513\n", argv[0]);
        return 2;
    }
    const std::string workload = argv[2];
    const bool prefill = workload == "prefill-512";
    const bool decode = workload == "decode-513";
    if (!prefill && !decode) {
        std::fprintf(stderr, "workload must be prefill-512 or decode-513\n");
        return 2;
    }
    const llama_pos position = decode ? 512 : 0;
    const int32_t token_count = prefill ? 512 : 1;

    ggml_backend_load_all();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(argv[1], model_params);
    if (model == nullptr) {
        std::fprintf(stderr, "failed to load model\n");
        return 1;
    }
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = std::max<uint32_t>(1024, position + 1);
    context_params.n_batch = token_count;
    context_params.n_ubatch = token_count;
    context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context * context = llama_init_from_model(model, context_params);
    if (context == nullptr) {
        llama_model_free(model);
        std::fprintf(stderr, "failed to create context\n");
        return 1;
    }

    if (position > 0) {
        auto * cache = dynamic_cast<llama_kv_cache *>(context->get_memory());
        if (cache == nullptr) {
            llama_free(context);
            llama_model_free(model);
            std::fprintf(stderr, "model does not use a directly seedable KV cache\n");
            return 1;
        }
        llama_batch_allocr allocator(1);
        llama_ubatch seed = allocator.ubatch_reserve(position, 1);
        for (llama_pos pos = 0; pos < position; ++pos) {
            seed.pos[pos] = pos;
            seed.n_seq_id[pos] = 1;
            seed.seq_id[pos] = &seed.seq_id_unq[0];
        }
        const llama_kv_cache::slot_info slot = cache->find_slot(seed, false);
        if (slot.empty()) {
            llama_free(context);
            llama_model_free(model);
            std::fprintf(stderr, "failed to reserve synthetic KV metadata\n");
            return 1;
        }
        cache->apply_ubatch(slot, seed);
    }

    llama_batch batch = llama_batch_init(token_count, 0, 1);
    batch.n_tokens = token_count;
    for (int32_t i = 0; i < token_count; ++i) {
        batch.token[i] = llama_vocab_bos(llama_model_get_vocab(model));
        batch.pos[i] = position + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = i == token_count - 1;
    }
    const int result = llama_decode(context, batch);
    llama_batch_free(batch);
    llama_free(context);
    llama_model_free(model);
    if (result == 0) {
        std::fprintf(stderr, "capture-only backend unexpectedly executed the graph\n");
        return 1;
    }
    std::fprintf(stderr, "captured the %s graph and observed the expected planning-only execution failure\n", workload.c_str());
    return 0;
}
