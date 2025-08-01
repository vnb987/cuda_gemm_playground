#include "gemm.hpp"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <vector>
#include <iostream>
#include <random>
#include "device_matrix.hpp"
#include "functors.cuh"
#include "cuda_gemm_handler.hpp"

using namespace cpu;
using namespace gpu;

template<typename T>
void printVector(std::vector<T>& A){
    int num_row = std::ceil(float(A.size()) / 16);
    for(int i = 0; i < num_row; i++){
        auto elem = std::min(16, int(A.size() - 16 * i));
        std::cout << "[";
        for (int j = 0; j < elem; j++){
            std::cout << A[i * 16 + j];
            if(j != elem-1)
                std::cout << ",";
        }
        std::cout << "]" << std::endl;
    }
    std::cout <<"end" << std::endl;
}

template<typename T>
void printMatrix(std::vector<T>& A, int num_row, int num_col){
    for(int i = 0; i < num_row; i++){
        std::cout << "[";
        for (int j = 0; j < num_col; j++){
            std::cout << A[i * num_col + j];
            if(j != num_col-1)
                std::cout << ",";
        }
        std::cout << "]" << std::endl;
    }
}

template<typename T>
std::vector<T> makeRandomVector(size_t size, T min = T(0), T max = T(1)) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<T> vec(size);
    if constexpr (std::is_integral<T>::value) {
        std::uniform_int_distribution<T> dist(min, max);
        for (auto& v : vec) {
            v = dist(gen);
        }
    } else {
        std::uniform_real_distribution<T> dist(min, max);
        for (auto& v : vec) {
            v = dist(gen);
        }
    }
    return vec;
}

TEST(GemmTestCpu, basic) {
    auto A = makeRandomVector<float>(64, 1, 5);
    auto B = makeRandomVector<float>(64, 1, 5);
    std::vector<float> C(64, 0.0f);
    layout t{8, 8, 8, 4, 4, 4};
    std::cout << "[A matrix]---------------------------------------------" << std::endl;
    printMatrix(A, 8, 8);
    std::cout << "[B matrix]---------------------------------------------"<< std::endl;
    printMatrix(B, 8, 8);
    gemm(A, B, C ,t);
    std::cout << "[C matrix]---------------------------------------------" << std::endl;
    printMatrix(C, 8, 8);
}


TEST(GemmTestGpu, naiveImplTest){
    // object generation
    int m = 1024;
    int n = 1024;
    int k = 1024;
    auto A = makeRandomVector<float>(m * n, 1, 5);
    auto B = makeRandomVector<float>(n * k, 1, 5);
    auto C_cpu = makeRandomVector<float>(m * k, 1, 1);
    auto C_gpu = makeRandomVector<float>(m * k, 1, 1);
    gpu::device_matrix<float> A_d(m,n);
    gpu::device_matrix<float> B_d(n, k);
    gpu::device_matrix<float> C_d(m, k);
    A_d.copy_from_host(A.data());
    B_d.copy_from_host(B.data());
    // define layout for CPU impl.
    layout t{m, n, k, 16, 16, 16};
    gemm(A, B, C_cpu, t);
    gpu::cudaGemmHandler<float, 32, 32, 32, 16, MultiplyAddOutplace> handler;
    handler.compute(A_d, B_d, C_d);
    C_d.copy_to_host(C_gpu.data());
    for(int i = 0; i < 128; i++){
        for(int j = 0; j < 128; j++){
          EXPECT_FLOAT_EQ(C_cpu[128 * i + j], C_gpu[128 * i + j]);
        }
    }
    // printMatrix(C_gpu, C_d.rows(), C_d.cols());
}
