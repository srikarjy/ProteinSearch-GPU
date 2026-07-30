#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>

#define MAX_SEQ_LEN 1024
#define WARP_SIZE 32

__constant__ int8_t BLOSUM62[24][24];

__device__ int8_t get_blosum62_score(char a, char b) {
    static const char aa_order[] = "ARNDCQEGHILKMFPSTWYVXUOJBZ";
    int idx_a = -1, idx_b = -1;
    
    for (int i = 0; i < 24; ++i) {
        if (aa_order[i] == a) idx_a = i;
        if (aa_order[i] == b) idx_b = i;
    }
    
    if (idx_a >= 0 && idx_b >= 0) {
        return BLOSUM62[idx_a][idx_b];
    }
    return -4;
}

__global__ void smith_waterman_kernel(const char* __restrict__ query_seqs,
                                       const char* __restrict__ db_seqs,
                                       const int64_t* __restrict__ query_lens,
                                       const int64_t* __restrict__ db_lens,
                                       float* __restrict__ scores,
                                       int64_t num_queries,
                                       int64_t num_db,
                                       int gap_open,
                                       int gap_extend) {
    int64_t query_idx = blockIdx.x;
    int64_t db_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (query_idx >= num_queries || db_idx >= num_db) return;
    
    int64_t qlen = query_lens[query_idx];
    int64_t dlen = db_lens[db_idx];
    
    if (qlen > MAX_SEQ_LEN || dlen > MAX_SEQ_LEN) {
        scores[query_idx * num_db + db_idx] = -1e9f;
        return;
    }
    
    const char* query = query_seqs + query_idx * MAX_SEQ_LEN;
    const char* db = db_seqs + db_idx * MAX_SEQ_LEN;
    
    __shared__ int H[(MAX_SEQ_LEN + 1) * (MAX_SEQ_LEN + 1)];
    __shared__ int E[(MAX_SEQ_LEN + 1) * (MAX_SEQ_LEN + 1)];
    __shared__ int F[(MAX_SEQ_LEN + 1) * (MAX_SEQ_LEN + 1)];
    
    int* H_ptr = H;
    int* E_ptr = E;
    int* F_ptr = F;
    
    for (int i = 0; i <= qlen; ++i) {
        H_ptr[i * (dlen + 1)] = 0;
        E_ptr[i * (dlen + 1)] = -10000;
        F_ptr[i * (dlen + 1)] = -10000;
    }
    for (int j = 0; j <= dlen; ++j) {
        H_ptr[j] = 0;
        E_ptr[j] = -10000;
        F_ptr[j] = -10000;
    }
    
    int max_score = 0;
    
    for (int i = 1; i <= qlen; ++i) {
        for (int j = 1; j <= dlen; ++j) {
            int idx = i * (dlen + 1) + j;
            
            int match = H_ptr[(i-1) * (dlen + 1) + (j-1)] + get_blosum62_score(query[i-1], db[j-1]);
            
            int e = max(E_ptr[(i-1) * (dlen + 1) + j] - gap_extend, 
                       H_ptr[(i-1) * (dlen + 1) + j] - gap_open);
            E_ptr[idx] = e;
            
            int f = max(F_ptr[i * (dlen + 1) + (j-1)] - gap_extend,
                       H_ptr[i * (dlen + 1) + (j-1)] - gap_open);
            F_ptr[idx] = f;
            
            int h = max({0, match, e, f});
            H_ptr[idx] = h;
            
            if (h > max_score) max_score = h;
        }
    }
    
    scores[query_idx * num_db + db_idx] = static_cast<float>(max_score);
}

__global__ void smith_waterman_striped_kernel(const char* __restrict__ query_seqs,
                                               const char* __restrict__ db_seqs,
                                               const int64_t* __restrict__ query_lens,
                                               const int64_t* __restrict__ db_lens,
                                               float* __restrict__ scores,
                                               int64_t num_queries,
                                               int64_t num_db,
                                               int gap_open,
                                               int gap_extend) {
    int64_t query_idx = blockIdx.x;
    int64_t db_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (query_idx >= num_queries || db_idx >= num_db) return;
    
    int64_t qlen = query_lens[query_idx];
    int64_t dlen = db_lens[db_idx];
    
    if (qlen > MAX_SEQ_LEN || dlen > MAX_SEQ_LEN) {
        scores[query_idx * num_db + db_idx] = -1e9f;
        return;
    }
    
    const char* query = query_seqs + query_idx * MAX_SEQ_LEN;
    const char* db = db_seqs + db_idx * MAX_SEQ_LEN;
    
    __shared__ int H_shared[(MAX_SEQ_LEN + 1) * (MAX_SEQ_LEN + 1)];
    __shared__ int E_shared[(MAX_SEQ_LEN + 1) * (MAX_SEQ_LEN + 1)];
    __shared__ int F_shared[(MAX_SEQ_LEN + 1) * (MAX_SEQ_LEN + 1)];
    
    int* H = H_shared;
    int* E = E_shared;
    int* F = F_shared;
    
    int max_score = 0;
    int prev_H = 0;
    
    for (int j = 0; j <= dlen; ++j) {
        H[j] = 0;
        E[j] = -10000;
        F[j] = -10000;
    }
    
    for (int i = 1; i <= qlen; ++i) {
        int idx = i * (dlen + 1);
        H[idx] = 0;
        E[idx] = -10000;
        F[idx] = -10000;
        prev_H = 0;
        
        for (int j = 1; j <= dlen; ++j) {
            int cur_idx = idx + j;
            int diag_idx = (i-1) * (dlen + 1) + (j-1);
            int up_idx = (i-1) * (dlen + 1) + j;
            int left_idx = idx + (j-1);
            
            int match = H[diag_idx] + get_blosum62_score(query[i-1], db[j-1]);
            
            int e = max(E[up_idx] - gap_extend, H[up_idx] - gap_open);
            E[cur_idx] = e;
            
            int f = max(F[left_idx] - gap_extend, H[left_idx] - gap_open);
            F[cur_idx] = f;
            
            int h = max({0, match, e, f});
            H[cur_idx] = h;
            
            if (h > max_score) max_score = h;
        }
    }
    
    scores[query_idx * num_db + db_idx] = static_cast<float>(max_score);
}

__global__ void smith_waterman_batch_kernel(const char* __restrict__ query_seqs,
                                             const char* __restrict__ db_seqs,
                                             const int64_t* __restrict__ query_lens,
                                             const int64_t* __restrict__ db_lens,
                                             float* __restrict__ scores,
                                             int64_t num_pairs,
                                             int gap_open,
                                             int gap_extend) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;
    
    int64_t query_idx = idx / gridDim.y;
    int64_t db_idx = idx % gridDim.y;
    
    int64_t qlen = query_lens[query_idx];
    int64_t dlen = db_lens[db_idx];
    
    if (qlen > MAX_SEQ_LEN || dlen > MAX_SEQ_LEN) {
        scores[idx] = -1e9f;
        return;
    }
    
    const char* query = query_seqs + query_idx * MAX_SEQ_LEN;
    const char* db = db_seqs + db_idx * MAX_SEQ_LEN;
    
    int H[(MAX_SEQ_LEN + 1) * (MAX_SEQ_LEN + 1)];
    int E[(MAX_SEQ_LEN + 1) * (MAX_SEQ_LEN + 1)];
    int F[(MAX_SEQ_LEN + 1) * (MAX_SEQ_LEN + 1)];
    
    int max_score = 0;
    
    for (int j = 0; j <= dlen; ++j) {
        H[j] = 0;
        E[j] = -10000;
        F[j] = -10000;
    }
    
    for (int i = 1; i <= qlen; ++i) {
        int idx_i = i * (dlen + 1);
        H[idx_i] = 0;
        E[idx_i] = -10000;
        F[idx_i] = -10000;
        
        for (int j = 1; j <= dlen; ++j) {
            int cur = idx_i + j;
            int diag = (i-1) * (dlen + 1) + (j-1);
            int up = (i-1) * (dlen + 1) + j;
            int left = idx_i + (j-1);
            
            int match = H[diag] + get_blosum62_score(query[i-1], db[j-1]);
            
            int e = max(E[up] - gap_extend, H[up] - gap_open);
            E[cur] = e;
            
            int f = max(F[left] - gap_extend, H[left] - gap_open);
            F[cur] = f;
            
            int h = max({0, match, e, f});
            H[cur] = h;
            
            if (h > max_score) max_score = h;
        }
    }
    
    scores[idx] = static_cast<float>(max_score);
}

__host__ void init_blosum62() {
    int8_t blosum62[24][24] = {
        { 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0,-2,-1, 0,-2},
        {-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3,-1, 0,-1,-2},
        {-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3, 3, 0,-1,-1},
        {-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3, 4, 1,-1,-1},
        { 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1,-3,-3,-2,-3},
        {-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2, 0, 3,-1,-1},
        {-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2, 1, 4,-1,-1},
        { 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3,-1,-2,-1,-2},
        {-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3, 0, 0,-1,-1},
        {-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3,-3,-3,-1,-1},
        {-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1,-4,-3,-1,-1},
        {-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2, 0, 1,-1,-1},
        {-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1,-3,-1,-1,-1},
        {-1,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1,-3,-3,-1,-1},
        {-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2,-2,-1,-2,-1},
        { 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2, 0, 0, 0,-1},
        { 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0,-1,-1, 0,-1},
        {-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3,-4,-3,-2,-2},
        {-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-2,-3,-2,-1,-2},
        { 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-2, 4,-3,-2,-1,-1},
        {-2,-1, 3, 4,-3, 0, 1,-1, 0,-3,-4, 0,-3,-1,-2, 0,-1,-4,-3,-3, 4, 1,-1,-1},
        {-1, 0, 0, 1,-3, 3, 4,-2, 0,-3,-3, 1,-1,-3,-1, 0,-1,-3,-2,-2, 1, 4,-1,-1},
        { 0,-1,-1,-1,-2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-2, 0, 0,-2,-1,-1,-1,-1,-1,-1},
        {-2,-2,-1,-1,-3,-1,-1,-2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1, 4}
    };
    
    cudaMemcpyToSymbol(BLOSUM62, blosum62, sizeof(blosum62));
}

extern "C" void launch_smith_waterman(const char* query_seqs, const char* db_seqs,
                                       const int64_t* query_lens, const int64_t* db_lens,
                                       float* scores, int64_t num_queries, int64_t num_db,
                                       int gap_open, int gap_extend) {
    init_blosum62();
    
    dim3 block(256);
    dim3 grid(num_queries, (num_db + block.x - 1) / block.x);
    smith_waterman_striped_kernel<<<grid, block>>>(query_seqs, db_seqs, query_lens, db_lens,
                                                    scores, num_queries, num_db,
                                                    gap_open, gap_extend);
}

extern "C" void launch_smith_waterman_batch(const char* query_seqs, const char* db_seqs,
                                             const int64_t* query_lens, const int64_t* db_lens,
                                             float* scores, int64_t num_pairs,
                                             int gap_open, int gap_extend) {
    init_blosum62();
    
    int block_size = 256;
    int grid_size = (num_pairs + block_size - 1) / block_size;
    smith_waterman_batch_kernel<<<grid_size, block_size>>>(query_seqs, db_seqs, query_lens, db_lens,
                                                            scores, num_pairs, gap_open, gap_extend);
}