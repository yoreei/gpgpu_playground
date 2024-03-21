#include <iostream>
#include <vector>

__device__ int d_result;
int h_result;

__global__ void oob() {
	__shared__ int partial_sum[2];

	partial_sum[4]=	partial_sum[0] + partial_sum[3];
	d_result = 5;
}

int main(){
		oob<<<1, 12>>>();
		cudaMemcpyFromSymbol(&h_result, d_result, sizeof(int));
		std::cout << "Result: " << h_result << std::endl;
	
}