#include <hip/hip_runtime.h>
#include <stdint.h>

extern "C" __global__ void hrx_sum_rows_f32(const float * src, float * dst, const int64_t ncols, const int64_t nrows) {
    const int64_t row = (int64_t) blockIdx.x;
    const int     tid = threadIdx.x;

    __shared__ float partial[256];
    float            sum = 0.0f;
    if (row < nrows) {
        const float * src_row = src + row * ncols;
        for (int64_t col = tid; col < ncols; col += blockDim.x) {
            sum += src_row[col];
        }
    }
    partial[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0 && row < nrows) {
        dst[row] = partial[0];
    }
}
