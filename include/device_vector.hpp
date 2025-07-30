#pragma once

#include <rmm/device_uvector.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <vector>

namespace rmm_wrapped {

template <typename T>
class device_vector {
public:
    using value_type = T;
    using iterator = typename rmm::device_uvector<T>::iterator;
    using const_iterator = typename rmm::device_uvector<T>::const_iterator;

    // 기본 allocator: memory pool 기반
    static rmm::mr::device_memory_resource* get_default_resource() {
        static rmm::mr::cuda_memory_resource cuda_mr;
        static rmm::mr::pool_memory_resource pool_mr{&cuda_mr};
        return &pool_mr;
    }

    // 생성자: 크기만 지정
    explicit device_vector(size_t size)
        : vec_(size, rmm::cuda_stream_view{}, get_default_resource()) {}

    // 생성자: 크기 + 초기값
    device_vector(size_t size, const T& value)
        : vec_(size, rmm::cuda_stream_view{}, get_default_resource()) {
        thrust::fill(begin(), end(), value);
        cudaStreamSynchronize(0);  // 기본 stream
    }

    // 크기 반환
    size_t size() const { return vec_.size(); }

    // 데이터 접근
    T* data() { return vec_.data(); }
    const T* data() const { return vec_.data(); }

    iterator begin() { return vec_.begin(); }
    iterator end() { return vec_.end(); }
    const_iterator begin() const { return vec_.begin(); }
    const_iterator end() const { return vec_.end(); }

    // 리사이즈
    void resize(size_t new_size) {
        vec_.resize(new_size, rmm::cuda_stream_view{});
    }

    // operator[]는 포인터 반환
    T* operator[](size_t i) { return vec_.data() + i; }
    const T* operator[](size_t i) const { return vec_.data() + i; }

    // host → device 복사
    void copy_from_host(const std::vector<T>& host_vec) {
        resize(host_vec.size());
        thrust::copy(thrust::host, host_vec.begin(), host_vec.end(), begin());
    }

    // device → host 복사
    std::vector<T> copy_to_host() const {
        std::vector<T> host_vec(size());
        thrust::copy(begin(), end(), host_vec.begin());
        return host_vec;
    }

    // 내부 uvect 접근
    rmm::device_uvector<T>& raw() { return vec_; }
    const rmm::device_uvector<T>& raw() const { return vec_; }

private:
    rmm::device_uvector<T> vec_;
};

} // namespace rmm_wrapped
