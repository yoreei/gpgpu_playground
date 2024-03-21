  // d_Grid2D[tid] = tid;
    // if (tid == 0) {
    //     // Pretty print radii and their corresponding origins
    //     printf("Antennae Radii and Origins:\n");
    //     for (int i = 0; i < num_antennae; ++i) {
    //         printf("Radii[%d] = %d, OriginX[%d] = %d, OriginY[%d] = %d\n", i, d_radii[i], i, d_originX[i], i, d_originY[i]);
    //     }
    //     // Pretty print Points of Interest Vertices
    //     printf("Points of Interest Vertices:\n");
    //     for (int i = 0; i < num_PoIVert; ++i) {
    //         printf("PoIVertX[%d] = %d, PoIVertY[%d] = %d\n", i, d_PoIVertX[i], i, d_PoIVertY[i]);
    //     }
    //     // Print maxX
    //     printf("maxX = %d\n", maxX);
    //     // Optionally, print the starting part of the grid, if needed
    //     // Limited number for demonstration as grids can be very large
    //     int printLimit = min(maxX * maxX, 10); // Arbitrary limit for demonstration
    //     printf("Grid2D (First %d Elements):\n", printLimit);
    //     for (int i = 0; i < printLimit; ++i) {
    //         printf("Grid2D[%d] = %d\n", i, d_Grid2D[i]);
    //     }
    // }

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

/* if max(blockIdx.x) > 1 : v_r[i % blockDim.x == 0] will store blockwise sum
*  if max(blockIdx.x) == 1: v_r[0] will store the sum of all elements in v
*/
__global__ void sum_reduction(int *v, int *v_r, int N) {
	__shared__ int partial_sum[SHMEM_SIZE];
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  //printf("starting thread %d\n", tid);
  if (tid >= N) {
    printf("stopped thread %d\n", tid);
    return;
  }
	// Optimization: Load elements AND do first add of reduction
	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];
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
		v_r[blockIdx.x] = partial_sum[0];
	}
}
// END SUM REDUCTION


#define NO_SIGNAL 0
#define HAS_SIGNAL 1
#define OUT_OF_POI 2

__global__ void calculateCoverage(const int *d_radii, int num_antennae,
                                  const int *d_originX, const int *d_originY,
                                  const int *d_PoIVertX, const int *d_PoIVertY, int num_PoIVert,
                                  const int maxX, const int maxY, int *d_Grid2D)
 {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= maxX * maxY) return;
    int X = tid % maxX;
    int Y = tid / maxX;
    d_Grid2D[tid] = NO_SIGNAL; 
    for (size_t i = 0; i < num_antennae; ++i) {
        int dx = X - d_originX[i];
        int dy = Y - d_originY[i];
        int d2 = dx * dx + dy * dy; 
        int r2 = d_radii[i] * d_radii[i];
        if (d2 <= r2) {
            d_Grid2D[tid] = HAS_SIGNAL;
            break; 
        }
    }
}

void display_2d(std::vector<int> Grid2D, size_t maxX, std::string label) {
  std::cout<<label<<std::endl;
  for (int i = 0; i < Grid2D.size() / maxX; ++i) {
    for( int j = 0; j < maxX; ++j){
        std::cout<<Grid2D[i*maxX + j]<<" ";
    }
    std::cout<<std::endl;
  }
}

/* Dynamically calculate max X and Y grid span
*/
void getBoundingSq(
    const std::vector<int>& PoIVertX,
    const std::vector<int>& PoIVertY,
    uint32_t& maxX, uint32_t& maxY){
        maxX=maxY=0;
        for(size_t i = 0; i < PoIVertX.size(); ++i){
            if(PoIVertX[i] > maxX) maxX = PoIVertX[i];
            if(PoIVertY[i] > maxY) maxY = PoIVertY[i];
        }
}

int main() {
  // struct of arrays data layout
  std::vector<int> radii = {10, 10};
  std::vector<int> originX = {0, 10};
  std::vector<int> originY = {0, 10};
  std::vector<int> PoIVertX = {0, 10, 0, 10};
  std::vector<int> PoIVertY = {0, 10, 0, 10};
  uint32_t maxX = 0, maxY = 0;
  getBoundingSq(PoIVertX, PoIVertY, maxX, maxY);
  int numThreads = SIZE;
  int numBlocks = (maxX * maxY + numThreads - 1) / numThreads;
  std::vector<int> Grid2D;
  Grid2D.resize(maxX * maxY);
  std::vector<int> h_sum_r;
  h_sum_r.resize(numThreads);

  int *d_radii, *d_originX, *d_originY, *d_PoIVertX, *d_PoIVertY, *d_Grid2D, *d_sum_r;
  size_t size_radii = sizeof(decltype(radii)::value_type) * radii.size();
  size_t size_originX = sizeof(decltype(originX)::value_type) * originX.size();
  size_t size_originY = sizeof(decltype(originY)::value_type) * originY.size();
  size_t size_PoIVertX = sizeof(decltype(PoIVertX)::value_type) * PoIVertX.size();
  size_t size_PoIVertY = sizeof(decltype(PoIVertY)::value_type) * PoIVertY.size();
  size_t size_Grid2D = sizeof(decltype(Grid2D)::value_type) * Grid2D.size();
  size_t size_sum_r = sizeof(decltype(Grid2D)::value_type) * Grid2D.size();

  cudaMalloc(&d_radii, size_radii);
  cudaMalloc(&d_originX, size_originX);
  cudaMalloc(&d_originY, size_originY);
  cudaMalloc(&d_PoIVertX, size_PoIVertX);
  cudaMalloc(&d_PoIVertY, size_PoIVertY);
  cudaMalloc(&d_Grid2D, size_Grid2D);
  cudaMalloc(&d_sum_r, size_sum_r);

  cudaMemcpy(d_radii, radii.data(), size_radii, cudaMemcpyHostToDevice);
  cudaMemcpy(d_originX, originX.data(), size_originX, cudaMemcpyHostToDevice);
  cudaMemcpy(d_originY, originY.data(), size_originY, cudaMemcpyHostToDevice);
  cudaMemcpy(d_PoIVertX, PoIVertX.data(), size_PoIVertX, cudaMemcpyHostToDevice);
  cudaMemcpy(d_PoIVertY, PoIVertY.data(), size_PoIVertY, cudaMemcpyHostToDevice);
  
  std::cout<< "calculateCoverage<<<" << numBlocks << ", " << numThreads << ">>> to process " << maxX*maxY << " elements\n";
  calculateCoverage<<<numBlocks, numThreads>>>(d_radii, radii.size(), d_originX, d_originY, d_PoIVertX, d_PoIVertY, PoIVertX.size(), maxX, maxY, d_Grid2D);
	// Optimization: sum_reduction requires 2 times less blocks
    std::cout<< "sum_reduction<<<" << std::max(1, numBlocks / 2) << ", " << numThreads << ">>> to process " << maxX*maxY << " elements\n";
  sum_reduction << <std::max(1, numBlocks / 2), numThreads >> > (d_Grid2D, d_sum_r, maxX * maxY);
	sum_reduction << <1, numThreads >> > (d_sum_r, d_sum_r, maxX * maxY);

  cudaMemcpy(Grid2D.data(), d_Grid2D, size_Grid2D, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_sum_r.data(), d_sum_r, size_sum_r, cudaMemcpyDeviceToHost);
  
	cudaDeviceSynchronize(); // todo do we need?
  display_2d(Grid2D, maxX, "Grid2D");
	printf("GPU sum: %d \n", h_sum_r[0]);
	printf("CPU sum: %d \n", std::reduce(Grid2D.begin(), Grid2D.end()));

  cudaFree(d_radii); cudaFree(d_originX); cudaFree(d_originY);
  cudaFree(d_PoIVertX); cudaFree(d_PoIVertY); cudaFree(d_Grid2D);
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



