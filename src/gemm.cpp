#include <cassert>
#include <cmath>
#include <gemm.hpp>
#include <vector>
#include <omp.h>

namespace cpu {
template<typename T>
void gemm(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C, layout t){
    assert(A.size() == t.m * t.n);
    assert(B.size() == t.n * t.k);
    C.resize(t.m * t.k);
    // output tile slicing
    int num_tile_m = std::ceil(float(t.m) / t.tile_m);
    int num_tile_k = std::ceil(float(t.k) / t.tile_k);
    int num_tile_n = std::ceil(float(t.n) / t.tile_n);
    #pragma omp parallel for
    for(int i = 0; i < num_tile_m * num_tile_k; i++){
        auto m_tile_index = i % num_tile_m;
        auto k_tile_index = i / num_tile_m;
        auto out_tile_length_m = std::min(t.tile_m, int(t.m - m_tile_index * t.tile_m));
        auto out_tile_length_k = std::min(t.tile_k, int(t.m - k_tile_index * t.tile_k));
        T* output_ptr = C.data() + t.k * t.tile_m * m_tile_index + t.tile_k * k_tile_index;
        const T* A_ptr = A.data() + t.n * t.tile_m * m_tile_index;
        const T* B_ptr = B.data() + t.tile_k * k_tile_index;
        for(int j = 0; j < num_tile_n; j++){
            int elem = std::min(t.tile_n, t.n - t.tile_n * j);
            for(int mm = 0; mm < out_tile_length_m; mm++){
                for(int kk = 0; kk < out_tile_length_k; kk++){
                    T accum = j == 0? 0 : output_ptr[mm * t.k + kk];
                    for(int jj = 0; jj < elem; jj++){
                        accum += A_ptr[mm * t.n + (j * t.tile_n + jj)] * B_ptr[(j * t.tile_n + jj) * t.k + kk];
                    }
                    output_ptr[mm * t.k + kk] = accum;
                }
            }

        }
    } 
}

template void gemm<float>(
    const std::vector<float>&, const std::vector<float>&, std::vector<float>&, layout);

template void gemm<double>(
    const std::vector<double>&, const std::vector<double>&, std::vector<double>&, layout);

template void gemm<long double>(
    const std::vector<long double>&, const std::vector<long double>&, std::vector<long double>&, layout);

template void gemm<int>(
    const std::vector<int>&, const std::vector<int>&, std::vector<int>&, layout);

template void gemm<unsigned int>(
    const std::vector<unsigned int>&, const std::vector<unsigned int>&, std::vector<unsigned int>&, layout);

template void gemm<long>(
    const std::vector<long>&, const std::vector<long>&, std::vector<long>&, layout);

template void gemm<unsigned long>(
    const std::vector<unsigned long>&, const std::vector<unsigned long>&, std::vector<unsigned long>&, layout);

template void gemm<long long>(
    const std::vector<long long>&, const std::vector<long long>&, std::vector<long long>&, layout);

template void gemm<unsigned long long>(
    const std::vector<unsigned long long>&, const std::vector<unsigned long long>&, std::vector<unsigned long long>&, layout);

template void gemm<short>(
    const std::vector<short>&, const std::vector<short>&, std::vector<short>&, layout);

template void gemm<unsigned short>(
    const std::vector<unsigned short>&, const std::vector<unsigned short>&, std::vector<unsigned short>&, layout);

}
