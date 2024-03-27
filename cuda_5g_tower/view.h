#pragma once
#include <numeric>
#include "cuda_array.cuh"
#include "common.cuh"

struct view_data {
        cuda_array<int>::ptr_type buffer;
        int max_x;
        int sum;
};

void view_render(view_data& vd)
{
        print2d(*vd.buffer, vd.max_x, "buffer2d");
        printf("GPU sum: %d \n", vd.sum);
        printf("CPU sum: %d \n", std::reduce(vd.buffer->begin(), vd.buffer->end()));
}
