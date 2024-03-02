//#include <cooperative_groups.h>
//
//namespace histogram {
//    // Distributed Shared memory histogram kernel
//    __global__ void clusterHist_kernel(int *bins, const int nbins, const int bins_per_block, const int *__restrict__ input,
//                                       size_t array_size)
//    {
//      extern __shared__ int smem[];
//      namespace cg = cooperative_groups;
//      int tid = cg::this_grid().thread_rank();
//
//      // Cluster initialization, size and calculating local bin offsets.
//      cg::cluster_group cluster = cg::this_cluster();
//      unsigned int clusterBlockRank = cluster.block_rank();
//      int cluster_size = cluster.dim_blocks().x;
//
//      for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
//      {
//        smem[i] = 0; //Initialize shared memory histogram to zeros
//      }
//
//      // cluster synchronization ensures that shared memory is initialized to zero in
//      // all thread blocks in the cluster. It also ensures that all thread blocks
//      // have started executing and they exist concurrently.
//      cluster.sync();
//
//      for (int i = tid; i < array_size; i += blockDim.x * gridDim.x)
//      {
//        int ldata = input[i];
//
//        //Find the right histogram bin.
//        int binid = ldata;
//        if (ldata < 0)
//          binid = 0;
//        else if (ldata >= nbins)
//          binid = nbins - 1;
//
//        //Find destination block rank and offset for computing
//        //distributed shared memory histogram
//        int dst_block_rank = (int)(binid / bins_per_block);
//        int dst_offset = binid % bins_per_block;
//
//        //Pointer to target block shared memory
//        int *dst_smem = cluster.map_shared_rank(smem, dst_block_rank);
//
//        //Perform atomic update of the histogram bin
//        atomicAdd(dst_smem + dst_offset, 1);
//      }
//
//      // cluster synchronization is required to ensure all distributed shared
//      // memory operations are completed and no thread block exits while
//      // other thread blocks are still accessing distributed shared memory
//      cluster.sync();
//
//      // Perform global memory histogram, using the local distributed memory histogram
//      int *lbins = bins + cluster.block_rank() * bins_per_block;
//      for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
//      {
//        atomicAdd(&lbins[i], smem[i]);
//      }
//    }
//
//    void main()
//    {
//        // Kernel configuration
//        dim3 block(256);
//        dim3 grid(32);
//        int bins_per_block = 256;
//        int nbins = 256;
//        size_t array_size = 1024 * 1024 * 32;
//    
//        // Allocate device memory
//        int *d_bins;
//        int *d_input;
//        cudaMalloc(&d_bins, nbins * sizeof(int));
//        cudaMalloc(&d_input, array_size * sizeof(int));
//    
//        // Launch kernel
//        clusterHist_kernel<<<grid, block, bins_per_block * sizeof(int)>>>(d_bins, nbins, bins_per_block, d_input, array_size);
//    
//        // Free device memory
//        cudaFree(d_bins);
//        cudaFree(d_input);
//        }
//} // namespace histogram