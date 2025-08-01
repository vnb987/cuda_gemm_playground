#pragma once

#include "cuda_gemm_handler.hpp"
#include "functors.cuh"
namespace gpu {
#define INSTANTIATE_GEMM_HANDLERS()                                            \
  template class gpu::cudaGemmHandler<float, 32, 32, 32, 16,                   \
                                      MultiplyAddOutplace>;                    \
  template class gpu::cudaGemmHandler<double, 64, 64, 32, 16,                  \
                                      MultiplyAddOutplace>;
} // namespace gpu