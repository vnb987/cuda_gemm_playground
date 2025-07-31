#pragma once 

template<typename T>
struct MultiplyAddInplace{
    __host__ __device__ void operator()(const T& a, const T& b, T& accum)const{
        accum += a * b;
    }
};

template<typename T>
struct MultiplyAddOutplace{
    __host__ __device__ 
    T operator()(T a, T b, T accum)const{
        return accum + a * b;
    }
};

template<typename T>
struct Add{
    __host__ __device__ 
    T operator()(T a, T b, T accum)const{
        return a + b;
    }
};



