#pragma once
#include "device_matrix.hpp"

namespace gpu {

enum class gemmType {
    naive,
    shared,
    shared_overlapping
};

template <typename T, int tile_m, int tile_n, int tile_k, int tile_mm,
          int tile_kk, template <typename> class Op>
class cudaGemmHandler {
public:
  cudaGemmHandler() : impl_type_(gemmType::naive), op_{} {};
  void setGemmType(gemmType impl_type) { impl_type_ = impl_type; }
  void compute(const device_matrix<T> &A, const device_matrix<T> &B,
               device_matrix<T> &C);
  void transpose(const device_matrix<T> &A, device_matrix<T>& C); 

private:
    // Roadmap 
    void gemmNaive(const device_matrix<T>& A, const device_matrix<T>& B, device_matrix<T>& C);
    void gemmExtensiveRegisterReuse(const device_matrix<T>, const device_matrix<T>& B, device_matrix<T>& C);
    // void gemmSharedMemReuse(const device_matrix<T>, const device_matrix<T>& B, device_matrix<T>& C);
    // void gemmSharedMemPipeline(const device_matrix<T>, const device_matrix<T>& B, device_matrix<T>& C);
    // void gemmSharedMemTMA(const device_matrix<T>, const device_matrix<T>& B, device_matrix<T>& C);
    gemmType impl_type_;
    Op<T> op_;

    // No copy/move
    cudaGemmHandler(const cudaGemmHandler&) = delete;
    cudaGemmHandler& operator=(const cudaGemmHandler&) = delete;
};
}
