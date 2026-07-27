#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

extern "C" __global__ void hrx_soft_max_f32(const float * src, float * dst, const int64_t ncols, const int64_t nrows) {
    const int64_t row = (int64_t) blockIdx.x;
    const int     tid = threadIdx.x;

    __shared__ float scratch[256];
    float            value = -INFINITY;
    if (row < nrows && tid < ncols) {
        value = src[row * ncols + tid];
    }
    scratch[tid] = value;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        }
        __syncthreads();
    }
    const float maximum = scratch[0];

    value = 0.0f;
    if (row < nrows && tid < ncols) {
        value = expf(src[row * ncols + tid] - maximum);
    }
    scratch[tid] = value;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    const float sum = scratch[0];

    if (row < nrows && tid < ncols) {
        dst[row * ncols + tid] = value / sum;
    }
}
