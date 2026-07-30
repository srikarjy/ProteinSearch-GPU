#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_fp16.h>

#define TILE_SIZE 32

__global__ void cosine_similarity_kernel(const float* __restrict__ query,
                                          const float* __restrict__ database,
                                          float* __restrict__ output,
                                          int64_t query_dim,
                                          int64_t db_size) {
    int64_t db_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (db_idx >= db_size) return;

    const float* db_vec = database + db_idx * query_dim;
    float sum = 0.0f;

    for (int64_t i = 0; i < query_dim; ++i) {
        sum += query[i] * db_vec[i];
    }

    output[db_idx] = sum;
}

__global__ void cosine_similarity_batch_kernel(const float* __restrict__ queries,
                                                const float* __restrict__ database,
                                                float* __restrict__ output,
                                                int64_t query_dim,
                                                int64_t db_size,
                                                int64_t num_queries) {
    int64_t query_idx = blockIdx.z;
    int64_t db_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (query_idx >= num_queries || db_idx >= db_size) return;

    const float* query = queries + query_idx * query_dim;
    const float* db_vec = database + db_idx * query_dim;
    
    float sum = 0.0f;
    for (int64_t i = 0; i < query_dim; ++i) {
        sum += query[i] * db_vec[i];
    }
    
    output[query_idx * db_size + db_idx] = sum;
}

__global__ void cosine_similarity_tiled_kernel(const float* __restrict__ queries,
                                                const float* __restrict__ database,
                                                float* __restrict__ output,
                                                int64_t query_dim,
                                                int64_t db_size,
                                                int64_t num_queries) {
    extern __shared__ float smem[];
    float* query_tile = smem;
    float* db_tile = smem + TILE_SIZE * query_dim;

    int64_t query_idx = blockIdx.z;
    int64_t db_tile_start = blockIdx.x * TILE_SIZE;
    int64_t db_idx = db_tile_start + threadIdx.x;

    if (query_idx >= num_queries) return;

    const float* query = queries + query_idx * query_dim;
    float sum = 0.0f;

    for (int64_t dim_tile = 0; dim_tile < query_dim; dim_tile += TILE_SIZE) {
        int64_t tile_dim = min(TILE_SIZE, query_dim - dim_tile);

        if (threadIdx.x < tile_dim) {
            query_tile[threadIdx.x * query_dim + dim_tile] = query[dim_tile + threadIdx.x];
        }

        if (db_idx < db_size && threadIdx.x < tile_dim) {
            db_tile[threadIdx.x * query_dim + dim_tile] = database[db_idx * query_dim + dim_tile + threadIdx.x];
        }
        __syncthreads();

        if (db_idx < db_size) {
            for (int64_t i = 0; i < tile_dim; ++i) {
                sum += query_tile[threadIdx.x * query_dim + dim_tile + i] * 
                       db_tile[threadIdx.x * query_dim + dim_tile + i];
            }
        }
        __syncthreads();
    }

    if (db_idx < db_size) {
        output[query_idx * db_size + db_idx] = sum;
    }
}

__global__ void cosine_similarity_vectorized_kernel(const float* __restrict__ query,
                                                     const float* __restrict__ database,
                                                     float* __restrict__ output,
                                                     int64_t query_dim,
                                                     int64_t db_size) {
    int64_t db_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (db_idx >= db_size) return;

    const float* db_vec = database + db_idx * query_dim;
    float sum = 0.0f;

    int64_t dim4 = query_dim / 4;
    int64_t rem = query_dim % 4;

    for (int64_t i = 0; i < dim4; ++i) {
        float4 q = *reinterpret_cast<const float4*>(query + i * 4);
        float4 d = *reinterpret_cast<const float4*>(db_vec + i * 4);
        sum += q.x * d.x + q.y * d.y + q.z * d.z + q.w * d.w;
    }

    for (int64_t i = dim4 * 4; i < query_dim; ++i) {
        sum += query[i] * db_vec[i];
    }

    output[db_idx] = sum;
}

__global__ void cosine_similarity_batch_vectorized_kernel(const float* __restrict__ queries,
                                                           const float* __restrict__ database,
                                                           float* __restrict__ output,
                                                           int64_t query_dim,
                                                           int64_t db_size,
                                                           int64_t num_queries) {
    int64_t query_idx = blockIdx.z;
    int64_t db_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (query_idx >= num_queries || db_idx >= db_size) return;

    const float* query = queries + query_idx * query_dim;
    const float* db_vec = database + db_idx * query_dim;
    
    float sum = 0.0f;
    int64_t dim4 = query_dim / 4;
    int64_t rem = query_dim % 4;

    for (int64_t i = 0; i < dim4; ++i) {
        float4 q = *reinterpret_cast<const float4*>(query + i * 4);
        float4 d = *reinterpret_cast<const float4*>(db_vec + i * 4);
        sum += q.x * d.x + q.y * d.y + q.z * d.z + q.w * d.w;
    }

    for (int64_t i = dim4 * 4; i < query_dim; ++i) {
        sum += query[i] * db_vec[i];
    }
    
    output[query_idx * db_size + db_idx] = sum;
}

__global__ void cosine_similarity_fp16_kernel(const half* __restrict__ query,
                                               const half* __restrict__ database,
                                               float* __restrict__ output,
                                               int64_t query_dim,
                                               int64_t db_size) {
    int64_t db_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (db_idx >= db_size) return;

    const half* db_vec = database + db_idx * query_dim;
    float sum = 0.0f;

    for (int64_t i = 0; i < query_dim; ++i) {
        sum += __half2float(query[i]) * __half2float(db_vec[i]);
    }

    output[db_idx] = sum;
}

__global__ void cosine_similarity_batch_fp16_kernel(const half* __restrict__ queries,
                                                     const half* __restrict__ database,
                                                     float* __restrict__ output,
                                                     int64_t query_dim,
                                                     int64_t db_size,
                                                     int64_t num_queries) {
    int64_t query_idx = blockIdx.z;
    int64_t db_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (query_idx >= num_queries || db_idx >= db_size) return;

    const half* query = queries + query_idx * query_dim;
    const half* db_vec = database + db_idx * query_dim;
    
    float sum = 0.0f;
    for (int64_t i = 0; i < query_dim; ++i) {
        sum += __half2float(query[i]) * __half2float(db_vec[i]);
    }
    
    output[query_idx * db_size + db_idx] = sum;
}

extern "C" void launch_cosine_similarity_fp32(const float* query, const float* database,
                                               float* output, int64_t query_dim, int64_t db_size) {
    int block_size = 256;
    int grid_size = (db_size + block_size - 1) / block_size;
    cosine_similarity_vectorized_kernel<<<grid_size, block_size>>>(query, database, output, query_dim, db_size);
}

extern "C" void launch_cosine_similarity_batch_fp32(const float* queries, const float* database,
                                                     float* output, int64_t query_dim, 
                                                     int64_t db_size, int64_t num_queries) {
    int block_size = 256;
    dim3 grid((db_size + block_size - 1) / block_size, 1, num_queries);
    cosine_similarity_batch_vectorized_kernel<<<grid, block_size>>>(queries, database, output, query_dim, db_size, num_queries);
}

extern "C" void launch_cosine_similarity_tiled_fp32(const float* queries, const float* database,
                                                     float* output, int64_t query_dim, 
                                                     int64_t db_size, int64_t num_queries) {
    int block_size = TILE_SIZE;
    dim3 grid((db_size + block_size - 1) / block_size, 1, num_queries);
    size_t shared_mem = 2 * TILE_SIZE * query_dim * sizeof(float);
    cosine_similarity_tiled_kernel<<<grid, block_size, shared_mem>>>(queries, database, output, query_dim, db_size, num_queries);
}

extern "C" void launch_cosine_similarity_fp16(const half* query, const half* database,
                                               float* output, int64_t query_dim, int64_t db_size) {
    int block_size = 256;
    int grid_size = (db_size + block_size - 1) / block_size;
    cosine_similarity_fp16_kernel<<<grid_size, block_size>>>(query, database, output, query_dim, db_size);
}

extern "C" void launch_cosine_similarity_batch_fp16(const half* queries, const half* database,
                                                     float* output, int64_t query_dim, 
                                                     int64_t db_size, int64_t num_queries) {
    int block_size = 256;
    dim3 grid((db_size + block_size - 1) / block_size, 1, num_queries);
    cosine_similarity_batch_fp16_kernel<<<grid, block_size>>>(queries, database, output, query_dim, db_size, num_queries);
}