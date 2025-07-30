#pragma once
#include <vector>

namespace cpu {
struct layout{
    int m;
    int n;
    int k;
    int tile_m;
    int tile_n;
    int tile_k;
};

// reference code for sanity check
// This function basically assume flattened vector of 2D matrix which placed in row major order.
template<typename T>
void gemm(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C, layout);

}
