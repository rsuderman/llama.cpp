#include <hip/hip_runtime.h>
#include <stdint.h>

extern "C" __global__ void hrx_div_f32(const float * src0,
                                       const float * src1,
                                       float *       dst,
                                       const int64_t nelements,
                                       const int64_t ncols,
                                       const int64_t src1_nelements) {
    const int64_t i = (int64_t) blockIdx.x * (int64_t) blockDim.x + (int64_t) threadIdx.x;
    if (i >= nelements) {
        return;
    }

    const int64_t i1 = src1_nelements == nelements ? i : (src1_nelements == 1 ? 0 : i % ncols);
    dst[i]           = src0[i] / src1[i1];
}
