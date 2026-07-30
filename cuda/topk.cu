#include <cuda_runtime.h>
#include <cstdint>

#define WARP_SIZE 32

__inline__ __device__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float tmp = __shfl_down_sync(0xffffffff, val, offset);
        val = fmaxf(val, tmp);
    }
    return __shfl_sync(0xffffffff, val, 0);
}

__inline__ __device__ int warp_reduce_max_index(float val, int idx) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float tmp_val = __shfl_down_sync(0xffffffff, val, offset);
        int tmp_idx = __shfl_down_sync(0xffffffff, idx, offset);
        if (tmp_val > val) {
            val = tmp_val;
            idx = tmp_idx;
        }
    }
    return __shfl_sync(0xffffffff, idx, 0);
}

__inline__ __device__ void warp_bitonic_sort(float* values, int* indices, int k) {
    for (int stride = 1; stride < k; stride <<= 1) {
        for (int offset = stride; offset > 0; offset >>= 1) {
            int lane_id = threadIdx.x % WARP_SIZE;
            int partner = lane_id ^ offset;
            bool dir = (lane_id & stride) == 0;
            
            if (partner < k) {
                float my_val = values[lane_id];
                float partner_val = values[partner];
                int my_idx = indices[lane_id];
                int partner_idx = indices[partner];
                
                if ((my_val > partner_val) == dir) {
                    values[lane_id] = partner_val;
                    indices[lane_id] = partner_idx;
                }
            }
        }
    }
}

__global__ void topk_kernel(const float* __restrict__ similarities, 
                            int64_t num_vectors,
                            int64_t k,
                            float* __restrict__ topk_values,
                            int64_t* __restrict__ topk_indices) {
    int64_t query_idx = blockIdx.x;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int warps_per_block = blockDim.x / WARP_SIZE;
    
    if (query_idx >= gridDim.x) return;

    const float* sim = similarities + query_idx * num_vectors;
    
    float local_values[32];
    int local_indices[32];
    
    for (int i = 0; i < 32; ++i) {
        local_values[i] = -1e30f;
        local_indices[i] = -1;
    }
    
    for (int64_t i = lane_id + warp_id * WARP_SIZE; i < num_vectors; i += blockDim.x) {
        float val = sim[i];
        int idx = i;
        
        for (int j = 0; j < 32; ++j) {
            if (val > local_values[j]) {
                for (int m = 31; m > j; --m) {
                    local_values[m] = local_values[m-1];
                    local_indices[m] = local_indices[m-1];
                }
                local_values[j] = val;
                local_indices[j] = idx;
                break;
            }
        }
    }
    
    for (int j = 0; j < 32; ++j) {
        if (lane_id == j) {
            warp_bitonic_sort(local_values, local_indices, 32);
        }
    }
    
    if (lane_id < k) {
        int out_idx = query_idx * k + lane_id;
        topk_values[out_idx] = local_values[lane_id];
        topk_indices[out_idx] = local_indices[lane_id];
    }
}

__global__ void topk_kernel_block(const float* __restrict__ similarities, 
                                   int64_t num_vectors,
                                   int64_t k,
                                   float* __restrict__ topk_values,
                                   int64_t* __restrict__ topk_indices) {
    extern __shared__ float smem[];
    int64_t query_idx = blockIdx.x;
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int warps_per_block = blockDim.x / WARP_SIZE;
    
    if (query_idx >= gridDim.x) return;

    const float* sim = similarities + query_idx * num_vectors;
    float* warp_values = smem;
    int64_t* warp_indices = (int64_t*)(smem + warps_per_block * k);
    
    float local_values[32];
    int local_indices[32];
    
    for (int i = 0; i < 32; ++i) {
        local_values[i] = -1e30f;
        local_indices[i] = -1;
    }
    
    for (int64_t i = tid; i < num_vectors; i += blockDim.x) {
        float val = sim[i];
        int idx = i;
        
        for (int j = 0; j < 32; ++j) {
            if (val > local_values[j]) {
                for (int m = 31; m > j; --m) {
                    local_values[m] = local_values[m-1];
                    local_indices[m] = local_indices[m-1];
                }
                local_values[j] = val;
                local_indices[j] = idx;
                break;
            }
        }
    }
    
    for (int j = 0; j < 32; ++j) {
        if (lane_id == j) {
            warp_bitonic_sort(local_values, local_indices, 32);
        }
    }
    
    if (lane_id < k) {
        warp_values[warp_id * k + lane_id] = local_values[lane_id];
        warp_indices[warp_id * k + lane_id] = local_indices[lane_id];
    }
    __syncthreads();
    
    if (warp_id == 0) {
        float merge_values[32];
        int merge_indices[32];
        
        for (int i = 0; i < k; ++i) {
            merge_values[i] = -1e30f;
            merge_indices[i] = -1;
        }
        
        for (int w = 0; w < warps_per_block; ++w) {
            for (int j = 0; j < k; ++j) {
                float val = warp_values[w * k + j];
                int idx = warp_indices[w * k + j];
                
                for (int m = 0; m < k; ++m) {
                    if (val > merge_values[m]) {
                        for (int n = k - 1; n > m; --n) {
                            merge_values[n] = merge_values[n-1];
                            merge_indices[n] = merge_indices[n-1];
                        }
                        merge_values[m] = val;
                        merge_indices[m] = idx;
                        break;
                    }
                }
            }
        }
        
        if (lane_id < k) {
            topk_values[query_idx * k + lane_id] = merge_values[lane_id];
            topk_indices[query_idx * k + lane_id] = merge_indices[lane_id];
        }
    }
}

extern "C" void launch_topk(const float* similarities, int64_t num_vectors, int64_t k,
                            float* topk_values, int64_t* topk_indices, int num_queries) {
    int block_size = 256;
    int grid_size = num_queries;
    
    size_t shared_mem = (block_size / WARP_SIZE) * k * (sizeof(float) + sizeof(int64_t));
    topk_kernel_block<<<grid_size, block_size, shared_mem>>>(similarities, num_vectors, k, topk_values, topk_indices);
}

extern "C" void launch_topk_simple(const float* similarities, int64_t num_vectors, int64_t k,
                                    float* topk_values, int64_t* topk_indices, int num_queries) {
    int block_size = 256;
    int grid_size = num_queries;
    topk_kernel<<<grid_size, block_size>>>(similarities, num_vectors, k, topk_values, topk_indices);
}