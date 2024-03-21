#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <numeric>

using std::accumulate;
using std::generate;
using std::cout;
using std::vector;

template <typename T>
void print_v(vector<T> v){
	for(const auto& e : v){
		cout << e << " ";
	}
	cout<<std::endl;
}

#define SHMEM_SIZE 256

__global__ void sumReduction(int *v, int *v_r, int N) {
	// Allocate shared memory
	__shared__ int partial_sum[SHMEM_SIZE];

	// Calculate thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) { return; }

	// Load elements into shared memory
	partial_sum[threadIdx.x] = v[tid];
	__syncthreads();

	// Iterate of log base 2 the block dimension
	for (int s = 1; s < blockDim.x; s *= 2) {
		// Reduce the threads performing work by half previous the previous
		// iteration each cycle
		if (threadIdx.x % (2 * s) == 0) {
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}

	// Let the thread 0 for this block write its result to main memory
	// Result is inexed by this block
	if (threadIdx.x == 0) {
		v_r[blockIdx.x] = partial_sum[0];
	}
}

int test(int N, int TB_SIZE) {
	// Vector size
	vector<int> h_v(N);

	int GRID_SIZE = (N + TB_SIZE -1) / TB_SIZE;
	vector<int> h_v_r(GRID_SIZE);

	size_t bytesIn = N * sizeof(int);
	size_t bytesOut = GRID_SIZE * sizeof(int);

    // Initialize the input data
    generate(begin(h_v), end(h_v), [](){ return 1; });

	// Allocate device memory
	int *d_v, *d_v_r;
	cudaMalloc(&d_v, bytesIn);
	cudaMalloc(&d_v_r, bytesOut);
	
	// Copy to device
	cudaMemcpy(d_v, h_v.data(), bytesIn, cudaMemcpyHostToDevice);

	// Call kernels
	cout << "input data: "; print_v(h_v); cout << "N: " << N << " GRID_SIZE: " << GRID_SIZE << " TB_SIZE: " << TB_SIZE << std::endl;
	sumReduction<<<GRID_SIZE, TB_SIZE>>>(d_v, d_v_r, N);
	cudaMemcpy(h_v_r.data(), d_v_r, bytesOut, cudaMemcpyDeviceToHost);
    cout << "device 1st pass: "; print_v(h_v_r);

	sumReduction<<<1, TB_SIZE>>>(d_v_r, d_v_r, N);
	cudaMemcpy(h_v_r.data(), d_v_r, bytesOut, cudaMemcpyDeviceToHost);
    cout << "device 2nd pass: "; print_v(h_v_r);

    int sum_h = std::accumulate(begin(h_v), end(h_v), 0);
	cout << "host: " << sum_h << std::endl;

	cudaFree(d_v);
	cudaFree(d_v_r);

	return 0;
}

int main(){
	test(1 << 16, 256);
}