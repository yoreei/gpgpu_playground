#pragma once
#include <tuple>
#include <stdexcept>
#include <iterator>
#include <iostream>
#include <source_location>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

const int SHMEM_SIZE = 256 * 4; // todo unhardcode

std::tuple<size_t, size_t> query_best_kernel_dims(size_t elem_cnt)
{
        size_t threads_cnt = 256;
        size_t blocks_cnt = (elem_cnt + threads_cnt - 1) / threads_cnt;
        return {blocks_cnt, threads_cnt};
}

void chck() {
        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) {
                std::string err = cudaGetErrorString(status);
                throw std::runtime_error(err);
        }
        cudaDeviceSynchronize();
        status = cudaGetLastError();
        if (status != cudaSuccess) {
                std::string err = cudaGetErrorString(status);
                throw std::runtime_error(err);
        }
}

inline void log(const std::string_view message,
         const std::source_location location =
               std::source_location::current())
{
        std::clog << location.line() << ':'
                << message << '\n';
}

template<typename Container>
void print2d(const Container& v, size_t max_x, const std::string& label)
{
        std::cout << label << std::endl;
        size_t nl_cnt = 0;
        for (auto it = std::begin(v); it != std::end(v); ++it) {
                if (nl_cnt == max_x) {
                        std::cout << std::endl;
                        nl_cnt = 0;
                }
                std::cout << *it;
                ++nl_cnt;
        }
        std::cout << std::endl;
}
