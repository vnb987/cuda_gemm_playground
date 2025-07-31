#pragma once
#include "device_matrix.hpp"

namespace gpu {

enum class gemmType {
    naive,
    shared,
    shared_overlapping
};

template <typename T, int tile_m, int tile_n, int tile_k, int tile_mm,
          template <typename> class Op>
class cudaGemmHandler {
public:
  cudaGemmHandler(dim3 block_dim, dim3 grid_dim)
      : impl_type_(gemmType::naive), block_dim_(block_dim),
        grid_dim_(grid_dim), op_{} {};
  void setGemmType(gemmType impl_type) { impl_type_ = impl_type; }
  void compute(const device_matrix<T> &A, const device_matrix<T> &B,
               device_matrix<T> &C);

private:
    void gemmNaive(const device_matrix<T>& A, const device_matrix<T>& B, device_matrix<T>& C);
    gemmType impl_type_;
    dim3 block_dim_;
    dim3 grid_dim_;
    Op<T> op_;

    // No copy/move
    cudaGemmHandler(const cudaGemmHandler&) = delete;
    cudaGemmHandler& operator=(const cudaGemmHandler&) = delete;
};
}
