#define GGML_COMMON_IMPL_CPP
#include "iq-codebook-resource.h"

#include "ggml-common.h"

#include <cstddef>
#include <cstdlib>

namespace ggml::hrx {

namespace {

const std::vector<uint8_t> & iq2_s_grid_bytes() {
    static_assert(sizeof(iq2s_grid) == 8192);
    static const std::vector<uint8_t> data(reinterpret_cast<const uint8_t *>(iq2s_grid),
                                           reinterpret_cast<const uint8_t *>(iq2s_grid) + sizeof(iq2s_grid));
    return data;
}

const std::vector<uint8_t> & iq3_s_grid_bytes() {
    static_assert(sizeof(iq3s_grid) == 2048);
    static const std::vector<uint8_t> data(reinterpret_cast<const uint8_t *>(iq3s_grid),
                                           reinterpret_cast<const uint8_t *>(iq3s_grid) + sizeof(iq3s_grid));
    return data;
}

}  // namespace

const IQCodebookResourceSpec & iq_codebook_resource_spec(IQCodebookResource resource) {
    static const IQCodebookResourceSpec iq2_s = { "common.iq2_s.grid.v1", iq2_s_grid_bytes(), 256 };
    static const IQCodebookResourceSpec iq3_s = { "common.iq3_s.grid.v1", iq3_s_grid_bytes(), 256 };
    switch (resource) {
        case IQCodebookResource::IQ2S:
            return iq2_s;
        case IQCodebookResource::IQ3S:
            return iq3_s;
    }
    std::abort();
}

}  // namespace ggml::hrx
