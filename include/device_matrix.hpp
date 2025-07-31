#pragma once

#include "device_vector.hpp"
#include <stdexcept>
#include <cassert>
#define CUDA_TRY(call)                                                     \
  do {                                                                     \
    cudaError_t const status = (call);                                     \
    if (status != cudaSuccess) {                                           \
      throw std::runtime_error(std::string{"CUDA error: "} +               \
                               cudaGetErrorString(status));                \
    }                                                                      \
  } while (0)
namespace gpu{
template <typename T>
class device_matrix : public device_vector<T> {
public:
    using device_vector<T>::data;
    using device_vector<T>::size;

    device_matrix(std::size_t rows, std::size_t cols, rmm::cuda_stream_view stream = rmm::cuda_stream_default)
        : device_vector<T>(rows * cols), rows_(rows), cols_(cols) {}

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }

    // 2D access: (row, col) -> 1D offset
    __host__ __device__
    T* at(std::size_t row, std::size_t col) {
        assert(row < rows_ && col < cols_);
        return data() + row * cols_ + col;
    }

    __host__ __device__
    const T* at(std::size_t row, std::size_t col) const {
        assert(row < rows_ && col < cols_);
        return data() + row * cols_ + col;
    }

    // Host copy interface
    void copy_from_host(const T* host_ptr, cudaStream_t stream = 0) {
        if (size() > 0)
            CUDA_TRY(cudaMemcpyAsync(data(), host_ptr, sizeof(T) * size(), cudaMemcpyHostToDevice, stream));
    }

    void copy_to_host(T* host_ptr, cudaStream_t stream = 0) const {
        if (size() > 0)
            CUDA_TRY(cudaMemcpyAsync(host_ptr, data(), sizeof(T) * size(), cudaMemcpyDeviceToHost, stream));
    }

private:
    std::size_t rows_;
    std::size_t cols_;
};
}