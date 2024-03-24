// BEGIN utils
#include <string>
#include <stdexcept>
#include <iterator>
#include <iostream>
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


template<typename Container>
void print2d(const Container& v, size_t maxX, const std::string& label) {
    std::cout << label << std::endl;

    size_t nl_cnt = 0;
    for (auto it = std::begin(v); it != std::end(v); ++it) {
        if(nl_cnt == maxX){
            std::cout << std::endl;
            nl_cnt = 0;
        }
        std::cout << *it;
        ++nl_cnt;
    }
    std::cout << std::endl;
}
// END utils

// BEGIN CUDAArray
#include <stdexcept>
#include <cassert>
#include <memory>

/*
POD wrapper around device & host memory representing the same object.
*/
template<typename T>
struct CUDAArray {
public:
    T* d_ptr = nullptr;
    T* h_ptr = nullptr;

private:
    size_t _size = 0;

public:
    __device__ __host__ size_t size() const { return _size; }

    struct Deleter {
      void operator()(CUDAArray* obj) const {
        if(obj){
            cudaFree(obj->d_ptr);
            chck();
            cudaFreeHost(obj->h_ptr);
            chck();
        }
      }
    };

    using CUDAArrayPtr = std::unique_ptr<CUDAArray<T>, CUDAArray<T>::Deleter>;

    template<typename... Args>
    static CUDAArrayPtr make_ptr(Args&&... args){
       auto* ptr = new CUDAArray<T>(std::forward<Args>(args)...);
       return CUDAArrayPtr(ptr, CUDAArray<T>::Deleter{});
    }

    static CUDAArrayPtr make_ptr(std::initializer_list<T> init) {
        auto* ptr = new CUDAArray<T>(init);
        return CUDAArrayPtr(ptr, CUDAArray<T>::Deleter{});
    }


    CUDAArray(size_t num_elements, T fill) : CUDAArray(num_elements) {
        std::fill_n(h_ptr, num_elements, fill);
        cudaMemcpy(d_ptr, h_ptr, size_bytes(), ::cudaMemcpyHostToDevice);
        chck();
    }

    CUDAArray(std::initializer_list<T> init) : CUDAArray(init.size()) {
        std::copy(init.begin(), init.end(), h_ptr);
        cudaMemcpy(d_ptr, h_ptr, size_bytes(), ::cudaMemcpyHostToDevice);
        chck();
    }

    explicit CUDAArray(size_t num_elements) : _size(num_elements) {
        cudaError_t status = cudaMalloc(&d_ptr, size_bytes());
        chck();
        status = cudaMallocHost(&h_ptr, size_bytes());
        chck();
    }

    /* TODO: Initialize CUDAArray with a C array, taking ownership */

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

    // not working?
//    __host__ __device__
//    T& _at(int idx) const {
//#ifdef __CUDA_ARCH__
//        if(idx >= _size) {
//          printf("Out of bounds access at %d\n", idx);
//        }
//        return d_ptr[idx];
//#else
//        assert(idx < _size);
//        return h_ptr[idx];
//#endif
//    }
//    __host__ __device__ T& at(size_t idx){ return _at(idx); }
//    __host__ __device__ const T& at(size_t idx) const { return _at(idx); }

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
// END CUDAArray

#include <numeric>  // std::reduce
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

// START SUM REDUCTION
// Sum function credit: Nick from CoffeeBeforeArch
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#define SIZE 256
#define SHMEM_SIZE 256 * 4

// Optimization: loop unrolling
__device__ void warpReduce(int* shmem_ptr, int t) {
	shmem_ptr[t] += shmem_ptr[t + 32];
  __syncwarp(); // to prevent caching of old values. Ensures correct result.
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

/* when max(blockIdx.x) > 1 : v_r[i % blockDim.x == 0] will store blockwise sum
*  when max(blockIdx.x) = 1: v_r[0] will store the sum of all elements in v
*/
__global__ void sum_reduction(const CUDAArray<int> v, CUDAArray<int> v_r, int N) {
	__shared__ int partial_sum[SHMEM_SIZE];
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  //printf("starting thread %d\n", tid);
  if (tid >= N) {
    printf("stopped thread %d\n", tid);
    return;
  }
	// Optimization: Load elements AND do first add of reduction
	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	partial_sum[threadIdx.x] = v.d_ptr[i] + v.d_ptr[i + blockDim.x];
	__syncthreads();

	// Optimization: Stop early (finish off with warpReduce)
	for (int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (threadIdx.x < s) {
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}

	if (threadIdx.x < 32) {
		warpReduce(partial_sum, threadIdx.x);
	}

  // if 1st pass <num_blocks, num_threads>: store partial sum for each block
	// if 2nd pass <1, num_threads>: store final sum
	if (threadIdx.x == 0) {
		v_r.d_ptr[blockIdx.x] = partial_sum[0];
	}
}
// END SUM REDUCTION

#define NO_SIGNAL 0
#define HAS_SIGNAL 1
#define OUT_OF_POI 2

__global__ void calculateCoverage(
  const CUDAArray<int> radii, const CUDAArray<int> originX, const CUDAArray<int> originY,
  const CUDAArray<int> PoIVertX, const CUDAArray<int> PoIVertY,
  const int maxX, const int maxY, CUDAArray<int> Grid2D
  )
 {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= maxX * maxY) return;
    int X = tid % maxX;
    int Y = tid / maxX;
    Grid2D.d_ptr[tid] = NO_SIGNAL; 
    for (size_t i = 0; i < radii.size(); ++i) {
        printf("X: %d, Y: %d, originX: %d, originY: %d, radii: %d\n", X, Y, originX.d_ptr[i], originY.d_ptr[i], radii.d_ptr[i]);
        int dx = X - originX.d_ptr[i];
        int dy = Y - originY.d_ptr[i];
        int d2 = dx * dx + dy * dy; 
        int r2 = radii.d_ptr[i] * radii.d_ptr[i];
        if (d2 <= r2) {
            printf("INTERSECT: X: %d, Y: %d, originX: %d, originY: %d, radii: %d\n", X, Y, originX.d_ptr[i], originY.d_ptr[i], radii.d_ptr[i]);
            Grid2D.d_ptr[tid] = HAS_SIGNAL;
            break; 
        }
    }
}

int main() {
      std::cout<< "BEGINNING EXECUTION\n";
      // struct of arrays data layout
      { // Device memory scope
      auto radii = CUDAArray<int>::make_ptr({1, 2});
      auto originX = CUDAArray<int>::make_ptr({0, 4});
      auto originY = CUDAArray<int>::make_ptr({0, 4});
      auto PoIVertX = CUDAArray<int>::make_ptr({0, 5, 0, 5});
      auto PoIVertY = CUDAArray<int>::make_ptr({0, 5, 5, 0});
      uint32_t maxX = *std::max_element(PoIVertX->begin(), PoIVertX->end());
      uint32_t maxY = *std::max_element(PoIVertY->begin(), PoIVertY->end());
      int numThreads = SIZE;
      int numBlocks = (maxX * maxY + numThreads - 1) / numThreads;
      auto Grid2D = CUDAArray<int>::make_ptr(maxX * maxY, 0); // fill with 0
      auto sum = CUDAArray<int>::make_ptr(numThreads, 0);
  
      std::cout<< "calculateCoverage<<<" << numBlocks << ", " << numThreads << ">>> to process " << maxX*maxY << " elements\n";
      calculateCoverage<<<numBlocks, numThreads>>>(*radii, *originX, *originY, *PoIVertX, *PoIVertY, maxX, maxY, *Grid2D);
      chck();
      // Optimization: sum_reduction requires 2 times less blocks
      std::cout<< "sum_reduction<<<" << std::max(1, numBlocks / 2) << ", " << numThreads << ">>> to process " << maxX*maxY << " elements\n";
      sum_reduction << <std::max(1, numBlocks / 2), numThreads >> > (*Grid2D, *sum, maxX * maxY); // TODO remove maxX*maxY param
      sum_reduction << <1, numThreads >> > (*sum, *sum, maxX * maxY); // TODO remove maxX*maxY param

      Grid2D->cudaMemcpyDeviceToHost();
      sum->cudaMemcpyDeviceToHost();
  
      print2d(*Grid2D, maxX, "Grid2D");
      printf("GPU sum: %d \n", sum->h_ptr[0]);
      printf("CPU sum: %d \n", std::reduce(Grid2D->begin(), Grid2D->end()));

      } // Device memory scope
      cudaDeviceReset();
      return 0;
}

// void getBoundingSq(
//     const std::vector<int>& PoIVertX,
//     const std::vector<int>& PoIVertY,
//     uint32_t& minX, uint32_t& minY,
//     uint32_t& maxX, uint32_t& maxY){
//         maxX=maxY=0;
//         minX=minY=UINT32_MAX;
//         for(size_t i = 0; i < PoIVertX.size(); ++i){
//             if(PoIVertX[i] < minX) minX = PoIVertX[i];
//             if(PoIVertX[i] > maxX) maxX = PoIVertX[i];
//             if(PoIVertY[i] < minY) minY = PoIVertY[i];
//             if(PoIVertY[i] > maxY) maxY = PoIVertY[i];
//         }
// }

///* Dynamically calculate max X and Y grid span
//*/
//void getBoundingSq(
//    const std::vector<int>& PoIVertX,
//    const std::vector<int>& PoIVertY,
//    uint32_t& maxX, uint32_t& maxY){
//        maxX=maxY=0;
//        for(size_t i = 0; i < PoIVertX.size(); ++i){
//            if(PoIVertX.at(i) > maxX) maxX = PoIVertX.at(i);
//            if(PoIVertY.at(i) > maxY) maxY = PoIVertY.at(i);
//        }
//}



