#include <hip/hip_runtime.h>

#include <stdint.h>

extern "C" __global__ void hrx_clamp_f32(
        const float * src,
        float * dst,
        const float minimum,
        const float maximum,
        const int32_t nelements) {
    const int32_t i = (int32_t) blockIdx.x * (int32_t) blockDim.x + (int32_t) threadIdx.x;

    if (i >= nelements) {
        return;
    }

    dst[i] = fminf(fmaxf(src[i], minimum), maximum);
}
