#include <hip/hip_runtime.h>

#include <stdint.h>

extern "C" __global__ void hrx_scale_f32(
        const float * src,
        float * dst,
        const float scale,
        const float bias,
        const int64_t nelements) {
    const int64_t tid = (int64_t) blockIdx.x * (int64_t) blockDim.x + (int64_t) threadIdx.x;
    const int64_t stride = (int64_t) blockDim.x * (int64_t) gridDim.x;

    for (int64_t i = tid; i < nelements; i += stride) {
        dst[i] = scale * src[i] + bias;
    }
}
