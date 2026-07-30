#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

#define WARP_SIZE 32

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return __shfl_sync(0xffffffff, val, 0);
}

__inline__ __device__ float block_reduce_sum(float val) {
    __shared__ float shared[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warp_reduce_sum(val);
    
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}

__global__ void l2_normalize_kernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int64_t num_vectors,
                                     int64_t dim) {
    int64_t vec_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (vec_idx >= num_vectors) return;
    
    const float* vec = input + vec_idx * dim;
    float* out_vec = output + vec_idx * dim;
    
    float sum_sq = 0.0f;
    for (int64_t i = tid; i < dim; i += blockDim.x) {
        float val = vec[i];
        sum_sq += val * val;
    }
    
    sum_sq = block_reduce_sum(sum_sq);
    
    if (tid == 0) {
        float norm = sqrtf(sum_sq) + 1e-12f;
        shared_mem[0] = norm;
    }
    __syncthreads();
    
    float norm = shared_mem[0];
    
    for (int64_t i = tid; i < dim; i += blockDim.x) {
        out_vec[i] = vec[i] / norm;
    }
}

__global__ void l2_normalize_kernel_vectorized(const float* __restrict__ input,
                                                float* __restrict__ output,
                                                int64_t num_vectors,
                                                int64_t dim) {
    int64_t vec_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (vec_idx >= num_vectors) return;
    
    const float* vec = input + vec_idx * dim;
    float* out_vec = output + vec_idx * dim;
    
    float sum_sq = 0.0f;
    int64_t vec_dim4 = dim / 4;
    int64_t rem = dim % 4;
    
    for (int64_t i = tid; i < vec_dim4; i += blockDim.x) {
        float4 vals = *reinterpret_cast<const float4*>(vec + i * 4);
        sum_sq += vals.x * vals.x + vals.y * vals.y + vals.z * vals.z + vals.w * vals.w;
    }
    
    if (tid < rem) {
        int64_t idx = vec_dim4 * 4 + tid;
        float val = vec[idx];
        sum_sq += val * val;
    }
    
    sum_sq = block_reduce_sum(sum_sq);
    
    __shared__ float s_norm;
    if (tid == 0) {
        s_norm = sqrtf(sum_sq) + 1e-12f;
    }
    __syncthreads();
    
    float norm = s_norm;
    
    for (int64_t i = tid; i < vec_dim4; i += blockDim.x) {
        float4 vals = *reinterpret_cast<const float4*>(vec + i * 4);
        float4 out_vals;
        out_vals.x = vals.x / norm;
        out_vals.y = vals.y / norm;
        out_vals.z = vals.z / norm;
        out_vals.w = vals.w / norm;
        *reinterpret_cast<float4*>(out_vec + i * 4) = out_vals;
    }
    
    if (tid < rem) {
        int64_t idx = vec_dim4 * 4 + tid;
        out_vec[idx] = vec[idx] / norm;
    }
}

__global__ void l2_normalize_batch_kernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           int64_t num_vectors,
                                           int64_t dim) {
    int64_t vec_idx = blockIdx.x * blockDim.y + threadIdx.y;
    int tid = threadIdx.x;
    
    if (vec_idx >= num_vectors) return;
    
    const float* vec = input + vec_idx * dim;
    float* out_vec = output + vec_idx * dim;
    
    float sum_sq = 0.0f;
    for (int64_t i = tid; i < dim; i += blockDim.x) {
        float val = vec[i];
        sum_sq += val * val;
    }
    
    sum_sq = block_reduce_sum(sum_sq);
    
    __shared__ float s_norm[32];
    if (tid == 0) {
        s_norm[threadIdx.y] = sqrtf(sum_sq) + 1e-12f;
    }
    __syncthreads();
    
    float norm = s_norm[threadIdx.y];
    
    for (int64_t i = tid; i < dim; i += blockDim.x) {
        out_vec[i] = vec[i] / norm;
    }
}

__global__ void l2_normalize_fp16_kernel(const half* __restrict__ input,
                                          half* __restrict__ output,
                                          int64_t num_vectors,
                                          int64_t dim) {
    int64_t vec_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (vec_idx >= num_vectors) return;
    
    const half* vec = input + vec_idx * dim;
    half* out_vec = output + vec_idx * dim;
    
    float sum_sq = 0.0f;
    for (int64_t i = tid; i < dim; i += blockDim.x) {
        float val = __half2float(vec[i]);
        sum_sq += val * val;
    }
    
    sum_sq = block_reduce_sum(sum_sq);
    
    __shared__ float s_norm;
    if (tid == 0) {
        s_norm = sqrtf(sum_sq) + 1e-12f;
    }
    __syncthreads();
    
    float norm = s_norm;
    
    for (int64_t i = tid; i < dim; i += blockDim.x) {
        out_vec[i] = __float2half_rn(__half2float(vec[i]) / norm);
    }
}

__global__ void l2_normalize_inplace_kernel(float* __restrict__ data,
                                             int64_t num_vectors,
                                             int64_t dim) {
    int64_t vec_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (vec_idx >= num_vectors) return;
    
    float* vec = data + vec_idx * dim;
    
    float sum_sq = 0.0f;
    for (int64_t i = tid; i < dim; i += blockDim.x) {
        float val = vec[i];
        sum_sq += val * val;
    }
    
    sum_sq = block_reduce_sum(sum_sq);
    
    __shared__ float s_norm;
    if (tid == 0) {
        s_norm = sqrtf(sum_sq) + 1e-12f;
    }
    __syncthreads();
    
    float norm = s_norm;
    
    for (int64_t i = tid; i < dim; i += blockDim.x) {
        vec[i] /= norm;
    }
}

extern "C" void launch_l2_normalize(const float* input, float* output,
                                     int64_t num_vectors, int64_t dim) {
    int block_size = 256;
    int grid_size = num_vectors;
    l2_normalize_kernel_vectorized<<<grid_size, block_size>>>(input, output, num_vectors, dim);
}

extern "C" void launch_l2_normalize_batch(const float* input, float* output,
                                           int64_t num_vectors, int64_t dim,
                                           int batch_size) {
    dim3 block(256, 4);
    dim3 grid((num_vectors + block.y - 1) / block.y);
    l2_normalize_batch_kernel<<<grid, block>>>(input, output, num_vectors, dim);
}

extern "C" void launch_l2_normalize_fp16(const half* input, half* output,
                                          int64_t num_vectors, int64_t dim) {
    int block_size = 256;
    int grid_size = num_vectors;
    l2_normalize_fp16_kernel<<<grid_size, block_size>>>(input, output, num_vectors, dim);
}

extern "C" void launch_l2_normalize_inplace(float* data, int64_t num_vectors, int64_t dim) {
    int block_size = 256;
    int grid_size = num_vectors;
    l2_normalize_inplace_kernel<<<grid_size, block_size>>>(data, num_vectors, dim);
}