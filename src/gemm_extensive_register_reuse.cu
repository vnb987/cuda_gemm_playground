#include "cuda_gemm_handler.hpp"
#include "device_matrix.hpp"
#include "functors.cuh"
#include "explicit_instantiations.hpp"
#include <cmath>

namespace gpu {

template <typename T, int tile_m, int tile_n, int tile_k, int tile_mm,
          int tile_kk, typename Op>
__global__ void gemm_kernel(const T *A, const T *B, const int m, const int n,
                            const int k, const int num_tile_m,
                            const int num_tile_k, T *C, Op op) {
  // Even if we do not use shared memory, basic cache blocking is used
  // gridDim.x : k / tile_k
  // gridDim.y : m / tile_m
  // T A_[tile_mm];
  T C_[tile_mm];
  // int m_tile_index = blockIdx.y;
  // int k_tile_index = blockIdx.x;
  for (int m_tile_index = blockIdx.y; m_tile_index < num_tile_m; m_tile_index += gridDim.y) {
    for (int k_tile_index = blockIdx.x; k_tile_index < num_tile_k; k_tile_index += gridDim.x) {
      T *output_ptr = C + k * (tile_m * m_tile_index + tile_mm * threadIdx.y) + tile_k * k_tile_index + threadIdx.x;
      if(tile_k * k_tile_index + threadIdx.x > k) break;
      if(tile_m * m_tile_index + tile_mm * threadIdx.y > m) break;
      const T *A_ptr = A + n * (tile_m * m_tile_index + tile_mm * threadIdx.y);
      const T *B_ptr = B + tile_k * k_tile_index+ threadIdx.x;
      #pragma parallel for
      for (int i = 0; i < tile_mm; i++){
        C_[i] = 0;
      }
      for (int i = 0; i < n; i++) {
        T B_ = B_ptr[i*k];
        for (int jj = 0; jj < tile_mm; jj++) {
          int m_index = m_tile_index * tile_m + tile_mm * threadIdx.y + jj;
          if (m_index >= m)
            break;
          C_[jj] = op(A_ptr[jj * n + i], B_, C_[jj]);
        }
      }
      for (int i = 0; i < tile_mm; i++)
        output_ptr[i * k] = C_[i];
    }
  }
}

template <typename T, int tile_m, int tile_n, int tile_k, int tile_mm, int tile_kk,
          template <typename> class Op>
void cudaGemmHandler<T, tile_m, tile_n, tile_k, tile_mm, tile_kk, Op>::gemmNaive(
    const device_matrix<T> &A, const device_matrix<T> &B, device_matrix<T> &C) {
  assert(tile_m % tile_mm == 0);
  int num_tile_mm = std::ceil(float(tile_m) / tile_mm);
  int num_tile_m = std::ceil(float(A.rows()) / tile_m);
  int num_tile_k = std::ceil(float(B.cols()) / tile_k);
  dim3 grid_dim_(num_tile_k, num_tile_m);
  dim3 block_dim_(tile_k, num_tile_mm);
  // we have already checked size mismatch
  gemm_kernel<T, tile_m, tile_n, tile_k, tile_mm><<<grid_dim_, block_dim_>>>(
      A.data(), B.data(), A.rows(), A.cols(), B.cols(), num_tile_m, num_tile_k,
      C.data(), op_);
  cudaDeviceSynchronize();
}

INSTANTIATE_GEMM_HANDLERS();
} // namespace gpu
