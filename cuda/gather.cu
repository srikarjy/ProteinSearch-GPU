#include <cuda_runtime.h>
#include <cstdint>

__global__ void gather_kernel(const int64_t* __restrict__ indices,
                               const float* __restrict__ database,
                               float* __restrict__ output,
                               int64_t query_dim,
                               int64_t k,
                               int64_t num_queries) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_output = num_queries * k * query_dim;
    
    if (tid >= total_output) return;
    
    int64_t query_idx = tid / (k * query_dim);
    int64_t k_idx = (tid % (k * query_dim)) / query_dim;
    int64_t dim_idx = tid % query_dim;
    
    int64_t db_idx = indices[query_idx * k + k_idx];
    output[tid] = database[db_idx * query_dim + dim_idx];
}

__global__ void gather_scores_kernel(const float* __restrict__ topk_values,
                                      float* __restrict__ output,
                                      int64_t k,
                                      int64_t num_queries) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = num_queries * k;
    
    if (tid >= total) return;
    output[tid] = topk_values[tid];
}

__global__ void gather_kernel_coalesced(const int64_t* __restrict__ indices,
                                         const float* __restrict__ database,
                                         float* __restrict__ output,
                                         int64_t query_dim,
                                         int64_t k,
                                         int64_t num_queries) {
    int64_t query_idx = blockIdx.x;
    int64_t k_idx = blockIdx.y;
    int64_t dim_start = threadIdx.x * 4;
    
    if (query_idx >= num_queries || k_idx >= k) return;
    
    int64_t db_idx = indices[query_idx * k + k_idx];
    const float* db_vec = database + db_idx * query_dim;
    float* out_vec = output + (query_idx * k + k_idx) * query_dim;
    
    for (int64_t i = dim_start; i < query_dim; i += blockDim.x * 4) {
        if (i + 3 < query_dim) {
            float4 val = *reinterpret_cast<const float4*>(db_vec + i);
            *reinterpret_cast<float4*>(out_vec + i) = val;
        } else {
            for (int j = 0; j < 4 && (i + j) < query_dim; ++j) {
                out_vec[i + j] = db_vec[i + j];
            }
        }
    }
}

__global__ void gather_indices_kernel(const int64_t* __restrict__ topk_indices,
                                       int64_t* __restrict__ output_indices,
                                       int64_t k,
                                       int64_t num_queries) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = num_queries * k;
    
    if (tid >= total) return;
    output_indices[tid] = topk_indices[tid];
}

extern "C" void launch_gather(const int64_t* indices, const float* database,
                               float* output, int64_t query_dim, int64_t k,
                               int64_t num_queries) {
    int64_t total = num_queries * k * query_dim;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    gather_kernel<<<grid_size, block_size>>>(indices, database, output, query_dim, k, num_queries);
}

extern "C" void launch_gather_coalesced(const int64_t* indices, const float* database,
                                         float* output, int64_t query_dim, int64_t k,
                                         int64_t num_queries) {
    dim3 grid(num_queries, k);
    int block_size = 256;
    gather_kernel_coalesced<<<grid, block_size>>>(indices, database, output, query_dim, k, num_queries);
}

extern "C" void launch_gather_scores(const float* topk_values, float* output,
                                      int64_t k, int64_t num_queries) {
    int64_t total = num_queries * k;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    gather_scores_kernel<<<grid_size, block_size>>>(topk_values, output, k, num_queries);
}