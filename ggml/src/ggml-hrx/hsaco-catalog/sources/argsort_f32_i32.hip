#include <hip/hip_runtime.h>
#include <stdint.h>

extern "C" __global__ void hrx_argsort_f32_i32(const float * src,
                                               int32_t *     dst,
                                               const int64_t ncols,
                                               const int64_t nrows,
                                               const int64_t ncols_pad,
                                               const int32_t order) {
    const int64_t row = (int64_t) blockIdx.x;
    const int     tid = threadIdx.x;

    __shared__ int32_t index[256];
    if (tid < ncols_pad) {
        index[tid] = tid;
    }
    __syncthreads();

    const float * src_row = src + row * ncols;
    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j >>= 1) {
            if (tid < ncols_pad) {
                const int ixj = tid ^ j;
                if (ixj > tid) {
                    const int32_t lhs       = index[tid];
                    const int32_t rhs       = index[ixj];
                    const bool    lhs_valid = lhs < ncols;
                    const bool    rhs_valid = rhs < ncols;
                    const bool    ascending = order == 0;
                    const bool    swap_low =
                        !lhs_valid ||
                        (rhs_valid && (ascending ? src_row[lhs] > src_row[rhs] : src_row[lhs] < src_row[rhs]));
                    const bool swap_high =
                        !rhs_valid ||
                        (lhs_valid && (ascending ? src_row[lhs] < src_row[rhs] : src_row[lhs] > src_row[rhs]));

                    if (((tid & k) == 0 && swap_low) || ((tid & k) != 0 && swap_high)) {
                        index[tid] = rhs;
                        index[ixj] = lhs;
                    }
                }
            }
            __syncthreads();
        }
    }

    if (row < nrows && tid < ncols) {
        dst[row * ncols + tid] = index[tid];
    }
}
