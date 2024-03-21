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

void errCheck(){
	cudaError_t launchError = cudaGetLastError();
	if (launchError != cudaSuccess) {
		printf("Kernel launch failed: %s\n", cudaGetErrorString(launchError));
	}

	// Synchronize device
	cudaError_t syncError = cudaDeviceSynchronize();
	if (syncError != cudaSuccess) {
		printf("Kernel execution failed: %s\n", cudaGetErrorString(syncError));
	}
}
__global__ void sumReduction(int *v, int N) {
	__shared__ int partial_sum[5];

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > N) { return; }

	partial_sum[threadIdx.x] = v[tid];
	__syncthreads();

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
		v[blockIdx.x] = partial_sum[0];
	}
}

template <typename T>
void print_v(vector<T> v){
	for(const auto& e : v){
		cout << e << " ";
	}
	cout<<std::endl;
}

int test(int N, int TB_SIZE) {
	// Vector size
	vector<int> h_v(N);

	int GRID_SIZE = (N + TB_SIZE -1) / TB_SIZE;

	size_t bytes_i = N * sizeof(int);

    // Initialize the input data
    generate(begin(h_v), end(h_v), [](){ return 1; });
	int sum_h = std::accumulate(begin(h_v), end(h_v), 0);
	cout << "host: " << sum_h << std::endl;

	// Allocate device memory
	int *d_v;
	cudaMalloc(&d_v, bytes_i);
	
	// Copy to device
	cudaMemcpy(d_v, h_v.data(), bytes_i, cudaMemcpyHostToDevice);

	// Call kernels
	cout << "input data: "; print_v(h_v); cout << "N: " << N << " GRID_SIZE: " << GRID_SIZE << " TB_SIZE: " << TB_SIZE << std::endl;
	sumReduction<<<GRID_SIZE, TB_SIZE>>>(d_v, N);
	errCheck();
	cudaMemcpy(h_v.data(), d_v, bytes_i, cudaMemcpyDeviceToHost);
    cout << "device 1st pass: "; print_v(h_v);

	sumReduction<<<1, TB_SIZE>>> (d_v, N);
	errCheck();

	cudaMemcpy(h_v.data(), d_v, bytes_i, cudaMemcpyDeviceToHost);
    cout << "device 2nd pass: "; print_v(h_v);

	cudaFree(d_v);
	return 0;
}

int main(){
	test(5, 4);
}