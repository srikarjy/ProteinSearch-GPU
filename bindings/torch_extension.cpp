#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

extern "C" void launch_l2_normalize(const float* input, float* output, int64_t num_vectors, int64_t dim);
extern "C" void launch_l2_normalize_batch(const float* input, float* output, int64_t num_vectors, int64_t dim, int batch_size);
extern "C" void launch_l2_normalize_fp16(const half* input, half* output, int64_t num_vectors, int64_t dim);
extern "C" void launch_l2_normalize_inplace(float* data, int64_t num_vectors, int64_t dim);

extern "C" void launch_cosine_similarity_fp32(const float* query, const float* database, float* output, int64_t query_dim, int64_t db_size);
extern "C" void launch_cosine_similarity_batch_fp32(const float* queries, const float* database, float* output, int64_t query_dim, int64_t db_size, int64_t num_queries);
extern "C" void launch_cosine_similarity_tiled_fp32(const float* queries, const float* database, float* output, int64_t query_dim, int64_t db_size, int64_t num_queries);
extern "C" void launch_cosine_similarity_fp16(const half* query, const half* database, float* output, int64_t query_dim, int64_t db_size);
extern "C" void launch_cosine_similarity_batch_fp16(const half* queries, const half* database, float* output, int64_t query_dim, int64_t db_size, int64_t num_queries);

extern "C" void launch_topk(const float* similarities, int64_t num_vectors, int64_t k, float* topk_values, int64_t* topk_indices, int num_queries);
extern "C" void launch_topk_simple(const float* similarities, int64_t num_vectors, int64_t k, float* topk_values, int64_t* topk_indices, int num_queries);

extern "C" void launch_gather(const int64_t* indices, const float* database, float* output, int64_t query_dim, int64_t k, int64_t num_queries);
extern "C" void launch_gather_coalesced(const int64_t* indices, const float* database, float* output, int64_t query_dim, int64_t k, int64_t num_queries);
extern "C" void launch_gather_scores(const float* topk_values, float* output, int64_t k, int64_t num_queries);

extern "C" void launch_smith_waterman(const char* query_seqs, const char* db_seqs, const int64_t* query_lens, const int64_t* db_lens, float* scores, int64_t num_queries, int64_t num_db, int gap_open, int gap_extend);
extern "C" void launch_smith_waterman_batch(const char* query_seqs, const char* db_seqs, const int64_t* query_lens, const int64_t* db_lens, float* scores, int64_t num_pairs, int gap_open, int gap_extend);

torch::Tensor l2_normalize(torch::Tensor input) {
    TORCH_CHECK(input.dim() == 2, "Input must be 2D tensor [num_vectors, dim]");
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
    
    int64_t num_vectors = input.size(0);
    int64_t dim = input.size(1);
    
    torch::Tensor output = torch::empty_like(input);
    
    if (input.scalar_type() == torch::kFloat32) {
        launch_l2_normalize(input.data_ptr<float>(), output.data_ptr<float>(), num_vectors, dim);
    } else if (input.scalar_type() == torch::kFloat16) {
        launch_l2_normalize_fp16(input.data_ptr<half>(), output.data_ptr<half>(), num_vectors, dim);
    } else {
        TORCH_CHECK(false, "Unsupported dtype");
    }
    
    cudaDeviceSynchronize();
    return output;
}

torch::Tensor l2_normalize_inplace(torch::Tensor input) {
    TORCH_CHECK(input.dim() == 2, "Input must be 2D tensor [num_vectors, dim]");
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Only float32 supported for inplace");
    
    int64_t num_vectors = input.size(0);
    int64_t dim = input.size(1);
    
    launch_l2_normalize_inplace(input.data_ptr<float>(), num_vectors, dim);
    cudaDeviceSynchronize();
    return input;
}

torch::Tensor cosine_similarity(torch::Tensor query, torch::Tensor database) {
    TORCH_CHECK(query.dim() == 1 || query.dim() == 2, "Query must be 1D or 2D");
    TORCH_CHECK(database.dim() == 2, "Database must be 2D [db_size, dim]");
    TORCH_CHECK(query.is_cuda() && database.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(query.scalar_type() == database.scalar_type(), "Query and database must have same dtype");
    
    int64_t query_dim = query.size(-1);
    int64_t db_size = database.size(0);
    int64_t db_dim = database.size(1);
    TORCH_CHECK(query_dim == db_dim, "Query and database dimensions must match");
    
    if (query.dim() == 1) {
        torch::Tensor output = torch::empty({db_size}, query.options().dtype(torch::kFloat32));
        
        if (query.scalar_type() == torch::kFloat32) {
            launch_cosine_similarity_fp32(query.data_ptr<float>(), database.data_ptr<float>(), 
                                          output.data_ptr<float>(), query_dim, db_size);
        } else if (query.scalar_type() == torch::kFloat16) {
            launch_cosine_similarity_fp16(query.data_ptr<half>(), database.data_ptr<half>(), 
                                          output.data_ptr<float>(), query_dim, db_size);
        }
        
        cudaDeviceSynchronize();
        return output;
    } else {
        int64_t num_queries = query.size(0);
        torch::Tensor output = torch::empty({num_queries, db_size}, query.options().dtype(torch::kFloat32));
        
        if (query.scalar_type() == torch::kFloat32) {
            launch_cosine_similarity_batch_fp32(query.data_ptr<float>(), database.data_ptr<float>(), 
                                                output.data_ptr<float>(), query_dim, db_size, num_queries);
        } else if (query.scalar_type() == torch::kFloat16) {
            launch_cosine_similarity_batch_fp16(query.data_ptr<half>(), database.data_ptr<half>(), 
                                                output.data_ptr<float>(), query_dim, db_size, num_queries);
        }
        
        cudaDeviceSynchronize();
        return output;
    }
}

torch::Tensor cosine_similarity_tiled(torch::Tensor queries, torch::Tensor database) {
    TORCH_CHECK(queries.dim() == 2, "Queries must be 2D [num_queries, dim]");
    TORCH_CHECK(database.dim() == 2, "Database must be 2D [db_size, dim]");
    TORCH_CHECK(queries.is_cuda() && database.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(queries.scalar_type() == torch::kFloat32, "Only float32 supported for tiled kernel");
    TORCH_CHECK(database.scalar_type() == torch::kFloat32, "Only float32 supported for tiled kernel");
    
    int64_t num_queries = queries.size(0);
    int64_t query_dim = queries.size(1);
    int64_t db_size = database.size(0);
    int64_t db_dim = database.size(1);
    TORCH_CHECK(query_dim == db_dim, "Query and database dimensions must match");
    
    torch::Tensor output = torch::empty({num_queries, db_size}, queries.options().dtype(torch::kFloat32));
    
    launch_cosine_similarity_tiled_fp32(queries.data_ptr<float>(), database.data_ptr<float>(), 
                                         output.data_ptr<float>(), query_dim, db_size, num_queries);
    
    cudaDeviceSynchronize();
    return output;
}

std::vector<torch::Tensor> topk(torch::Tensor similarities, int64_t k) {
    TORCH_CHECK(similarities.dim() == 2, "Similarities must be 2D [num_queries, db_size]");
    TORCH_CHECK(similarities.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(similarities.scalar_type() == torch::kFloat32, "Only float32 supported");
    
    int64_t num_queries = similarities.size(0);
    int64_t db_size = similarities.size(1);
    TORCH_CHECK(k <= db_size, "k must be <= db_size");
    
    torch::Tensor topk_values = torch::empty({num_queries, k}, similarities.options().dtype(torch::kFloat32));
    torch::Tensor topk_indices = torch::empty({num_queries, k}, similarities.options().dtype(torch::kInt64));
    
    launch_topk(similarities.data_ptr<float>(), db_size, k, 
                topk_values.data_ptr<float>(), topk_indices.data_ptr<int64_t>(), num_queries);
    
    cudaDeviceSynchronize();
    return {topk_values, topk_indices};
}

torch::Tensor gather(torch::Tensor database, torch::Tensor indices) {
    TORCH_CHECK(database.dim() == 2, "Database must be 2D [db_size, dim]");
    TORCH_CHECK(indices.dim() == 2, "Indices must be 2D [num_queries, k]");
    TORCH_CHECK(database.is_cuda() && indices.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(database.scalar_type() == torch::kFloat32, "Database must be float32");
    TORCH_CHECK(indices.scalar_type() == torch::kInt64, "Indices must be int64");
    
    int64_t db_size = database.size(0);
    int64_t query_dim = database.size(1);
    int64_t num_queries = indices.size(0);
    int64_t k = indices.size(1);
    
    torch::Tensor output = torch::empty({num_queries, k, query_dim}, database.options());
    
    launch_gather_coalesced(indices.data_ptr<int64_t>(), database.data_ptr<float>(),
                            output.data_ptr<float>(), query_dim, k, num_queries);
    
    cudaDeviceSynchronize();
    return output;
}

torch::Tensor smith_waterman(torch::Tensor query_seqs, torch::Tensor db_seqs,
                              torch::Tensor query_lens, torch::Tensor db_lens,
                              int gap_open, int gap_extend) {
    TORCH_CHECK(query_seqs.dim() == 2, "query_seqs must be 2D [num_queries, max_len]");
    TORCH_CHECK(db_seqs.dim() == 2, "db_seqs must be 2D [num_db, max_len]");
    TORCH_CHECK(query_lens.dim() == 1 && db_lens.dim() == 1, "Lengths must be 1D");
    TORCH_CHECK(query_seqs.is_cuda() && db_seqs.is_cuda() && query_lens.is_cuda() && db_lens.is_cuda(), "All tensors must be on CUDA");
    TORCH_CHECK(query_seqs.scalar_type() == torch::kInt8, "query_seqs must be int8");
    TORCH_CHECK(db_seqs.scalar_type() == torch::kInt8, "db_seqs must be int8");
    TORCH_CHECK(query_lens.scalar_type() == torch::kInt64, "query_lens must be int64");
    TORCH_CHECK(db_lens.scalar_type() == torch::kInt64, "db_lens must be int64");
    
    int64_t num_queries = query_seqs.size(0);
    int64_t num_db = db_seqs.size(0);
    
    torch::Tensor scores = torch::empty({num_queries, num_db}, query_seqs.options().dtype(torch::kFloat32));
    
    launch_smith_waterman(query_seqs.data_ptr<int8_t>(), db_seqs.data_ptr<int8_t>(),
                          query_lens.data_ptr<int64_t>(), db_lens.data_ptr<int64_t>(),
                          scores.data_ptr<float>(), num_queries, num_db, gap_open, gap_extend);
    
    cudaDeviceSynchronize();
    return scores;
}

torch::Tensor smith_waterman_batch(torch::Tensor query_seqs, torch::Tensor db_seqs,
                                    torch::Tensor query_lens, torch::Tensor db_lens,
                                    torch::Tensor pair_indices,
                                    int gap_open, int gap_extend) {
    TORCH_CHECK(query_seqs.dim() == 2, "query_seqs must be 2D [num_queries, max_len]");
    TORCH_CHECK(db_seqs.dim() == 2, "db_seqs must be 2D [num_db, max_len]");
    TORCH_CHECK(query_lens.dim() == 1 && db_lens.dim() == 1, "Lengths must be 1D");
    TORCH_CHECK(pair_indices.dim() == 2 && pair_indices.size(1) == 2, "pair_indices must be [num_pairs, 2]");
    TORCH_CHECK(query_seqs.is_cuda() && db_seqs.is_cuda() && query_lens.is_cuda() && db_lens.is_cuda() && pair_indices.is_cuda(), "All tensors must be on CUDA");
    TORCH_CHECK(query_seqs.scalar_type() == torch::kInt8, "query_seqs must be int8");
    TORCH_CHECK(db_seqs.scalar_type() == torch::kInt8, "db_seqs must be int8");
    TORCH_CHECK(query_lens.scalar_type() == torch::kInt64, "query_lens must be int64");
    TORCH_CHECK(db_lens.scalar_type() == torch::kInt64, "db_lens must be int64");
    TORCH_CHECK(pair_indices.scalar_type() == torch::kInt64, "pair_indices must be int64");
    
    int64_t num_pairs = pair_indices.size(0);
    
    torch::Tensor scores = torch::empty({num_pairs}, query_seqs.options().dtype(torch::kFloat32));
    
    launch_smith_waterman_batch(query_seqs.data_ptr<int8_t>(), db_seqs.data_ptr<int8_t>(),
                                 query_lens.data_ptr<int64_t>(), db_lens.data_ptr<int64_t>(),
                                 scores.data_ptr<float>(), num_pairs, gap_open, gap_extend);
    
    cudaDeviceSynchronize();
    return scores;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("l2_normalize", &l2_normalize, "L2 normalize vectors (CUDA)");
    m.def("l2_normalize_inplace", &l2_normalize_inplace, "L2 normalize vectors inplace (CUDA)");
    m.def("cosine_similarity", &cosine_similarity, "Cosine similarity (CUDA)");
    m.def("cosine_similarity_tiled", &cosine_similarity_tiled, "Tiled cosine similarity (CUDA)");
    m.def("topk", &topk, "Top-K selection (CUDA)");
    m.def("gather", &gather, "Gather vectors by indices (CUDA)");
    m.def("smith_waterman", &smith_waterman, "Smith-Waterman alignment (CUDA)");
    m.def("smith_waterman_batch", &smith_waterman_batch, "Batch Smith-Waterman alignment (CUDA)");
}