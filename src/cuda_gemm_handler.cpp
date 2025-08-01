#include "cuda_gemm_handler.hpp"
#include "explicit_instantiations.hpp"
#include "functors.cuh"
#include <stdexcept>

namespace gpu {

template <typename T, int tile_m, int tile_n, int tile_k, int tile_mm,
          int tile_kk, template <typename> class Op>
void cudaGemmHandler<T, tile_m, tile_n, tile_k, tile_mm, tile_kk, Op>::compute(
    const device_matrix<T> &A, const device_matrix<T> &B, device_matrix<T> &C) {
  switch (impl_type_) {
  case gemmType::naive:
    gemmNaive(A, B, C);
    break;
  default:
    throw std::logic_error("Other types are not implemented yet");
  }
}
INSTANTIATE_GEMM_HANDLERS();
} // namespace gpu