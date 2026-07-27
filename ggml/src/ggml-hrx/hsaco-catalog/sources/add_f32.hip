#include <hip/hip_runtime.h>

#include <stdint.h>

extern "C" __global__ void hrx_add_f32(
        const float * src0,
        const float * src1,
        float * dst,
        const int64_t nelements) {
    const int64_t i = (int64_t) blockIdx.x * (int64_t) blockDim.x + (int64_t) threadIdx.x;
    if (i >= nelements) {
        return;
    }

    dst[i] = src0[i] + src1[i];
}
