// BEGIN utils
#include <string>
#include <stdexcept>
#include <iterator>
#include <iostream>
#include <source_location>

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
// END utils

namespace model {

const int SHMEM_SIZE = 256 * 4; // todo unhardcode

// BEGIN cuda_array
#include <stdexcept>
#include <cassert>
#include <memory>

/*
POD wrapper around device & host memory representing the same object.
*/
template<typename T>
struct cuda_array {
public:
        T* d_ptr = nullptr;
        T* h_ptr = nullptr;

private:
        size_t _size = 0;

public:
        __device__ __host__ size_t size() const { return _size; }

        struct deleter {
        void operator()(cuda_array* obj) const
        {
                if (obj) {
                cudaFree(obj->d_ptr);
                chck();
                cudaFreeHost(obj->h_ptr);
                chck();
                }
        }
        };

        using ptr_type = std::unique_ptr<cuda_array<T>, cuda_array<T>::deleter>;

        template<typename... Args>
        static ptr_type make_ptr(Args&&... args) {
                auto* ptr = new cuda_array<T>(std::forward<Args>(args)...);
                return ptr_type(ptr, cuda_array<T>::deleter{});
        }

        static ptr_type make_ptr(std::initializer_list<T> init) {
                auto* ptr = new cuda_array<T>(init);
                return ptr_type(ptr, cuda_array<T>::deleter{});
        }

        cuda_array(size_t num_elements, T fill) : cuda_array(num_elements) {
                std::fill_n(h_ptr, num_elements, fill);
                cudaMemcpy(d_ptr, h_ptr, size_bytes(), ::cudaMemcpyHostToDevice);
                chck();
        }

        cuda_array(std::initializer_list<T> init) : cuda_array(init.size()) {
                std::copy(init.begin(), init.end(), h_ptr);
                cudaMemcpy(d_ptr, h_ptr, size_bytes(), ::cudaMemcpyHostToDevice);
                chck();
        }

        explicit cuda_array(size_t num_elements) : _size(num_elements) {
                cudaError_t status = cudaMalloc(&d_ptr, size_bytes());
                chck();
                status = cudaMallocHost(&h_ptr, size_bytes());
                chck();
        }

        /* TODO: Initialize cuda_array with a C array, taking ownership */

        using iterator = T*;
        using const_iterator = const T*;

        iterator begin() noexcept {
                return h_ptr;
        }

        iterator end() noexcept {
                return h_ptr + _size;
        }

        const_iterator begin() const noexcept {
                return h_ptr;
        }

        const_iterator end() const noexcept {
                return h_ptr + _size;
        }

        const_iterator cbegin() const noexcept {
                return h_ptr;
        }

        const_iterator cend() const noexcept {
                return h_ptr + _size;
        }

        void cudaMemcpyHostToDevice()
        {
                cudaMemcpy(d_ptr, h_ptr, size_bytes(), ::cudaMemcpyHostToDevice);
                chck();
        }

        void cudaMemcpyDeviceToHost()
        {
                cudaMemcpy(h_ptr, d_ptr, size_bytes(), ::cudaMemcpyDeviceToHost);
                chck();
        }

private:
        size_t size_bytes() const { return _size * sizeof(T); }
};
// END cuda_array

struct data { // model::data
        cuda_array<int>::ptr_type tower_radius;
        cuda_array<int>::ptr_type tower_x;
        cuda_array<int>::ptr_type tower_y;
        cuda_array<int>::ptr_type poi_x;
        cuda_array<int>::ptr_type poi_y;
        int max_x ;
        int max_y;
};


// Sum function credit: Nick from CoffeeBeforeArch
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/*
 * Optimization: loop unrolling
 */
__device__ void warp_reduce(int* shmem_ptr, int t)
{
    shmem_ptr[t] += shmem_ptr[t + 32];
        __syncwarp(); // Important: Prevent caching of old values
    shmem_ptr[t] += shmem_ptr[t + 16];
        __syncwarp();
    shmem_ptr[t] += shmem_ptr[t + 8];
         __syncwarp();
    shmem_ptr[t] += shmem_ptr[t + 4];
         __syncwarp();
    shmem_ptr[t] += shmem_ptr[t + 2];
        __syncwarp();
    shmem_ptr[t] += shmem_ptr[t + 1];
}

/* 
 * When max(blockIdx.x) > 1 : v_r[i % blockDim.x == 0] will store blockwise sum
 * When max(blockIdx.x) = 1 : v_r[0] will store the sum of all elements in v
 */
__global__ void signal_reduce(const model::cuda_array<int> v,
                                model::cuda_array<int> v_r, int N)
{
        __shared__ int partial_sum[SHMEM_SIZE];
        int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (tid >= N) {
                printf("stopped thread %d\n", tid);
                return;
        }
        // Optimization: Load elements AND do first add of reduction
        int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
        partial_sum[threadIdx.x] = v.d_ptr[i] + v.d_ptr[i + blockDim.x];
        __syncthreads();

        // Optimization: Stop early (finish off with warp_reduce)
        for (int s = blockDim.x / 2; s > 32; s >>= 1) {
                if (threadIdx.x < s) {
                    partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
                }
                __syncthreads();
        }

        if (threadIdx.x < 32) {
                warp_reduce(partial_sum, threadIdx.x);
        }

        // if 1st pass <num_blocks, num_threads>: store partial sum for each block
        // if 2nd pass <1, num_threads>: store final sum
        if (threadIdx.x == 0) {
                v_r.d_ptr[blockIdx.x] = partial_sum[0];
        }
}

#define NO_SIGNAL 0
#define HAS_SIGNAL 1
#define OUT_OF_POI 2

__global__ void signal_map (
        const model::cuda_array<int> tower_radius, const model::cuda_array<int> tower_x,
        const model::cuda_array<int> tower_y, const model::cuda_array<int> poi_x, 
        const model::cuda_array<int> poi_y, const int max_x, const int max_y,
        model::cuda_array<int> buffer2d
  )
 {
        int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (tid >= max_x * max_y) return;
        int x = tid % max_x;
        int y = tid / max_x;
        buffer2d.d_ptr[tid] = NO_SIGNAL; 
        for (size_t i = 0; i < tower_radius.size(); ++i) {
                printf("x: %d, y: %d, tower_x: %d, tower_y: %d, tower_radius: %d\n",
                        x, y, tower_x.d_ptr[i], tower_y.d_ptr[i], tower_radius.d_ptr[i]);
                int dx = x - tower_x.d_ptr[i];
                int dy = y - tower_y.d_ptr[i];
                int d2 = dx * dx + dy * dy; 
                int r2 = tower_radius.d_ptr[i] * tower_radius.d_ptr[i];
                if (d2 <= r2) {
                printf("INTERSECT: x: %d, y: %d, tower_x: %d, tower_y: %d, tower_radius: %d\n",
                        x, y, tower_x.d_ptr[i], tower_y.d_ptr[i], tower_radius.d_ptr[i]);
                buffer2d.d_ptr[tid] = HAS_SIGNAL;
                break; 
                }
        }
}

std::tuple<size_t, size_t> query_best_kernel_dims(size_t elem_cnt)
{
        size_t threads_cnt = 256;
        size_t blocks_cnt = (elem_cnt + threads_cnt - 1) / threads_cnt;
        return {blocks_cnt, threads_cnt};
}

cuda_array<int>::ptr_type compute_map(model::data& in)
{
        uint32_t max_x = *std::max_element(in.poi_x->begin(), in.poi_x->end());
        uint32_t max_y = *std::max_element(in.poi_y->begin(), in.poi_y->end());
        size_t N = max_x * max_y;
        auto [blocks_cnt, threads_cnt] = query_best_kernel_dims(N);
        auto buffer2d = cuda_array<int>::make_ptr(N, 0); // fill 0
        printf("compute_map<<<%d, %d>>> processing %d elements",
                blocks_cnt, threads_cnt, N); // TODO use fmt log

        signal_map<<<blocks_cnt, threads_cnt>>>(
        *in.tower_radius, *in.tower_x, *in.tower_y, *in.poi_x,
        *in.poi_y, max_x, max_y, *buffer2d);
        chck();

        buffer2d->cudaMemcpyDeviceToHost();
        return buffer2d;
}

int compute_reduce(cuda_array<int>::ptr_type& in)
{
        auto [blocks_cnt, threads_cnt] = query_best_kernel_dims(in->size());
        // Optimization: sum reduction requires 2 times less blocks
        blocks_cnt = std::max<size_t>(1, blocks_cnt / 2);
        auto out = cuda_array<int>::make_ptr(threads_cnt, 0);
        printf ("sum_reduction<<<%d, %d>>> processing %d elements",
                blocks_cnt, threads_cnt, in->size()); // TODO use fmt log

        signal_reduce<<<blocks_cnt, threads_cnt>>> (
        *in, *out, in->size()); // TODO refactor max_x*max_y
        chck();
        signal_reduce<<<1, threads_cnt>>> (*out, *out, in->size());
        chck();

        out->cudaMemcpyDeviceToHost();
        return out->h_ptr[0];
}

} // namespace model
 

#include <numeric>  // std::reduce
#include <algorithm>
#include <cassert>
namespace view {

struct data {
        model::cuda_array<int>::ptr_type buffer;
        int max_x;
        int sum;
};

void render(view::data& vd)
{
        print2d(*vd.buffer, vd.max_x, "buffer2d");
        printf("GPU sum: %d \n", vd.sum);
        printf("CPU sum: %d \n", std::reduce(vd.buffer->begin(), vd.buffer->end()));
}
    
} // namespace view

void test_cuda_array()
{
        auto arr = model::cuda_array<int>::make_ptr({1, 2, 3});
        assert(arr->size() == 3);
        assert(arr->h_ptr[0] == 1);
        assert(arr->h_ptr[1] == 2);
        assert(arr->h_ptr[2] == 3);
        arr->h_ptr[0] = 4;
        arr->cudaMemcpyHostToDevice();
        arr->cudaMemcpyDeviceToHost();
        assert(arr->h_ptr[0] == 4);
}

void test_compute_map(){

}

void test_compute_reduce(){

}

model::data read_data(){
        // TODO read JSON
        model::data md = {
            .tower_radius = model::cuda_array<int>::make_ptr({1, 2}),
            .tower_x = model::cuda_array<int>::make_ptr({0, 4}),
            .tower_y = model::cuda_array<int>::make_ptr({0, 4}),
            .poi_x = model::cuda_array<int>::make_ptr({0, 5, 0, 5}),
            .poi_y = model::cuda_array<int>::make_ptr({0, 5, 5, 0}),
            .max_x = 5,
            .max_y = 5
        };
        return md;
}

int main() {
        bool cfg_run_tests = true;
        printf("BEGINNING EXECUTION\n"); // TODO fmt log
        if (cfg_run_tests) {
                test_cuda_array();
                test_compute_map();
                test_compute_reduce();
        }

        { // device scope
                model::data md = read_data();
                view::data vd {};
                vd.max_x = md.max_x;
                vd.buffer = model::compute_map(md);
                vd.sum = model::compute_reduce(vd.buffer);
                view::render(vd);

        } // device scope

        cudaDeviceReset();
        return 0;
}

// void getBoundingSq(
//     const std::vector<int>& poi_x,
//     const std::vector<int>& poi_y,
//     uint32_t& minX, uint32_t& minY,
//     uint32_t& max_x, uint32_t& max_y){
//         max_x=max_y=0;
//         minX=minY=UINT32_MAX;
//         for(size_t i = 0; i < poi_x.size(); ++i){
//             if (poi_x[i] < minX) minX = poi_x[i];
//             if (poi_x[i] > max_x) max_x = poi_x[i];
//             if (poi_y[i] < minY) minY = poi_y[i];
//             if (poi_y[i] > max_y) max_y = poi_y[i];
//         }
// }

///* Dynamically calculate max x and y grid span
//*/
//void getBoundingSq(
//    const std::vector<int>& poi_x,
//    const std::vector<int>& poi_y,
//    uint32_t& max_x, uint32_t& max_y){
//        max_x=max_y=0;
//        for(size_t i = 0; i < poi_x.size(); ++i){
//            if (poi_x.at(i) > max_x) max_x = poi_x.at(i);
//            if (poi_y.at(i) > max_y) max_y = poi_y.at(i);
//        }
//}



