#include "cuda_gemm_handler.hpp"
#include "functors.cuh"
#include <stdexcept>

namespace gpu {

template <typename T, int tile_m, int tile_n, int tile_k, int tile_mm,
          template <typename> class Op>
void cudaGemmHandler<T, tile_m, tile_n, tile_k, tile_mm, Op>::compute(
    const device_matrix<T> &A, const device_matrix<T> &B, device_matrix<T> &C) {
  switch (impl_type_) {
  case gemmType::naive:
    gemmNaive(A, B, C);
    break;
  default:
    throw std::logic_error("Other types are not implemented yet");
  }
}
template class cudaGemmHandler<float, 32, 32, 32, 16, MultiplyAddOutplace>;
template class cudaGemmHandler<double, 64, 64, 32, 16, MultiplyAddOutplace>;
} // namespace gpu