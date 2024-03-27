#include "view.h"
#include "cuda_array.cuh"
#include "cuda_map.cuh"
#include "cuda_reduce.cuh"

map_data read_data(){
        // TODO read JSON
        map_data md = {
            .tower_radius = cuda_array<int>::make_ptr({1, 2}),
            .tower_x = cuda_array<int>::make_ptr({0, 4}),
            .tower_y = cuda_array<int>::make_ptr({0, 4}),
            .poi_x = cuda_array<int>::make_ptr({0, 5, 0, 5}),
            .poi_y = cuda_array<int>::make_ptr({0, 5, 5, 0}),
            .max_x = 5,
            .max_y = 5
        };
        return md;
}

int main() {
        printf("BEGINNING EXECUTION\n"); // TODO fmt log

        { // device scope
                map_data md = read_data();
                view_data vd {};
                vd.max_x = md.max_x;
                vd.buffer = cuda_map(md);
                vd.sum = cuda_reduce(vd.buffer);
                view_render(vd);

        } // device scope

        cudaDeviceReset();
        return 0;
}