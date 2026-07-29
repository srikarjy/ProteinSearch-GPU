protein-semsearch-gpu

### CUDA-Accelerated Semantic Protein Retrieval with Biological Re-ranking

> A high-performance semantic protein search engine that combines transformer embeddings, custom CUDA kernels, and biological sequence alignment to retrieve similar proteins from hundreds of thousands of sequences. Built to demonstrate GPU systems engineering, CUDA optimization, and AI infrastructure for computational biology.

---

## Motivation

Modern protein foundation models such as ESM-2 produce embeddings that capture structural and functional similarity beyond traditional sequence alignment. Searching these embeddings efficiently across hundreds of thousands or millions of proteins requires GPU-optimized retrieval pipelines rather than brute-force CPU computation.

ProteinSearch-GPU is an educational yet production-inspired implementation of an exact semantic protein search engine. Instead of relying solely on existing libraries such as cuBLAS or FAISS, this project implements the core retrieval primitives directly in CUDA to explore GPU architecture, memory optimization, and kernel design.

The pipeline performs:

* Transformer-based protein embedding generation
* Custom CUDA cosine similarity computation
* Warp-level Top-K candidate selection
* Biological re-ranking using Smith–Waterman alignment
* End-to-end benchmarking and GPU profiling

The goal is to understand how modern AI infrastructure systems are built rather than simply using existing frameworks.

---

# Architecture

```text
                     FASTA Query
                          │
                          ▼
              ESM-2 (PyTorch Inference)
                          │
                          ▼
              CUDA Kernel 1
                L2 Normalize
                          │
                          ▼
              CUDA Kernel 2
          Batched Cosine Similarity
                          │
                          ▼
              CUDA Kernel 3
            Warp-Level Top-K
                          │
                          ▼
              CUDA Kernel 4
        Candidate Compression
      (Top Indices + Similarities)
                          │
                          ▼
      Smith–Waterman Biological Re-ranking
                          │
                          ▼
                Ranked Protein Hits
```

---

# Features

* Custom CUDA kernels written from scratch
* PyTorch C++/CUDA extension
* Shared-memory optimized kernels
* Warp-level reductions using shuffle intrinsics
* Vectorized memory accesses
* Mixed precision experiments (FP16/BF16)
* Biological sequence re-ranking
* GPU profiling with NVIDIA Nsight Compute
* End-to-end retrieval benchmarks
* Reproducible evaluation pipeline

---

# CUDA Kernels

## Kernel 1 — L2 Normalize

Normalizes each protein embedding to unit length before similarity search.

### CUDA Concepts

* Parallel reductions
* Warp shuffle intrinsics
* Shared memory
* Coalesced memory access

---

## Kernel 2 — Tiled Cosine Similarity

Computes cosine similarity between a batch of query embeddings and a large embedding database.

### CUDA Concepts

* Shared-memory tiling
* Grid-stride loops
* Float4 vectorized loads
* Loop unrolling
* Occupancy optimization

---

## Kernel 3 — Warp-Level Top-K

Efficiently finds the highest scoring candidates without sorting the entire similarity matrix.

### CUDA Concepts

* Warp intrinsics
* Bitonic sorting
* Register-only computation
* Branch minimization

---

## Kernel 4 — Candidate Compression

Compacts only the Top-K indices and scores into contiguous buffers.

### CUDA Concepts

* Parallel gather
* Coalesced writes
* Prefix indexing

---

## Kernel 5 (Optional) — GPU Smith–Waterman

Wavefront dynamic programming for biological sequence alignment.

### CUDA Concepts

* Anti-diagonal DP
* Shared-memory banding
* Synchronization
* Dynamic programming optimization

---

# Technology Stack

## Languages

* CUDA C++
* C++17
* Python

## AI

* PyTorch
* Hugging Face Transformers
* ESM-2

## GPU

* CUDA
* Nsight Compute
* Nsight Systems

## Biology

* UniProt
* Swiss-Prot
* UniRef50
* Pfam
* Smith–Waterman
* Parasail

---

# Datasets

## Development

Swiss-Prot (Reviewed)

Purpose:

* Kernel debugging
* Initial benchmarking
* Functional validation

---

## Scaling

UniRef50 Subset

Purpose:

* Large-scale throughput testing
* Memory bandwidth analysis

---

## Biological Evaluation

Pfam Seed Alignments

Purpose:

* Recall@K
* Precision@K
* Family retrieval accuracy

---

# Pipeline

```text
FASTA

↓

ESM-2 Embedding

↓

L2 Normalization

↓

CUDA Similarity Search

↓

Warp Top-K

↓

Candidate Compression

↓

Smith–Waterman Re-ranking

↓

Final Results
```

---

# Repository Structure

```text
proteinsearch-gpu/

├── cuda/
│   ├── normalize.cu
│   ├── cosine.cu
│   ├── topk.cu
│   ├── gather.cu
│   └── smith_waterman.cu
│
├── bindings/
│   └── torch_extension.cpp
│
├── python/
│   ├── encoder.py
│   ├── benchmark.py
│   ├── evaluate.py
│   └── datasets.py
│
├── profiling/
│   ├── nsight_compute.md
│   ├── nsight_systems.md
│   └── reports/
│
├── benchmarks/
│
├── notebooks/
│
├── data/
│
└── README.md
```

---

# Benchmarks

## GPU Performance

The following metrics are collected using NVIDIA Nsight Compute.

| Metric                    | Description                          |
| ------------------------- | ------------------------------------ |
| Kernel latency            | Execution time per kernel            |
| Queries/sec               | End-to-end throughput                |
| Average query latency     | Time per query                       |
| GPU occupancy             | Percentage of active warps           |
| SM efficiency             | Streaming multiprocessor utilization |
| Warp execution efficiency | Active lanes per warp                |
| DRAM bandwidth            | Memory throughput                    |
| L2 cache hit rate         | Cache effectiveness                  |
| Global load efficiency    | Memory coalescing                    |
| Achieved FLOPS            | Floating-point throughput            |

---

## CUDA Metrics

Measured using:

```bash
ncu --set full python benchmark.py
```

Collected metrics include:

* sm__warps_active.avg.pct_of_peak_sustained_active
* dram__bytes.sum.per_second
* lts__t_sectors_hit_rate
* smsp__sass_average_branch_targets_threads_uniform
* gpu__time_duration
* sm__throughput.avg.pct_of_peak_sustained_elapsed

---

## Scalability

Benchmarks are performed across multiple database sizes.

| Database Size | Purpose                    |
| ------------- | -------------------------- |
| 10K proteins  | Correctness                |
| 100K proteins | Optimization               |
| 560K proteins | Production-scale benchmark |
| 1M+ proteins  | Scalability evaluation     |

---

## Batch Sizes

Experiments are repeated using

* Batch = 1
* Batch = 8
* Batch = 32
* Batch = 64
* Batch = 128
* Batch = 256
* Batch = 512

---

# Retrieval Evaluation

Beyond raw GPU performance, ProteinSearch-GPU measures biological retrieval quality.

Metrics include

* Recall@10
* Recall@50
* Precision@10
* Precision@50
* Mean Reciprocal Rank (MRR)
* nDCG@10
* Mean Average Precision (mAP)

---

# Biological Evaluation

The retrieved proteins are evaluated using curated annotations.

Experiments measure

* Same Pfam family retrieval
* Same UniProt annotation retrieval
* Improvement after Smith–Waterman re-ranking
* Exact match recovery
* Functional similarity

---

# Baselines

ProteinSearch-GPU is compared against

## GPU

* PyTorch + torch.matmul
* PyTorch + cosine_similarity
* FAISS (Flat Index)

## CPU

* NumPy
* SciPy
* BLASTP (retrieval quality reference)
* MMseqs2 (retrieval quality and speed reference)

---

# Mixed Precision Experiments

The project evaluates

* FP32
* FP16
* BF16

Metrics collected

* Throughput
* Latency
* Memory usage
* Retrieval quality
* Numerical error

---

# Profiling

## Nsight Compute

Kernel-level optimization

* Occupancy
* Memory bandwidth
* Warp efficiency
* Register pressure
* Shared-memory utilization
* Instruction mix

## Nsight Systems

Pipeline-level profiling

* CUDA kernel timeline
* CPU scheduling
* Memory copies
* Stream overlap
* Kernel launch overhead

---

# Future Work

* Approximate nearest-neighbor search (IVF, HNSW, Product Quantization)
* Multi-GPU database sharding with NCCL
* CUDA Graphs
* Persistent CUDA kernels
* Tensor Core optimization
* Triton kernel implementation
* FP8 inference experiments
* Distributed retrieval across multiple GPUs

---

# Results

Performance numbers below are intentionally left blank until benchmarking is completed.

## Throughput

| Database | Batch | Queries/sec | Avg Latency |
| -------- | ----- | ----------: | ----------: |
| 100K     | 1     |           — |           — |
| 100K     | 64    |           — |           — |
| 100K     | 512   |           — |           — |
| 560K     | 64    |           — |           — |
| 560K     | 512   |           — |           — |

---

## GPU Metrics

| Metric             | Result |
| ------------------ | -----: |
| Occupancy          |      — |
| DRAM Bandwidth     | — GB/s |
| L2 Cache Hit Rate  |    — % |
| Warp Efficiency    |    — % |
| Kernel Time        |   — ms |
| End-to-End Latency |   — ms |
| GPU Memory Usage   |   — GB |

---

## Retrieval Metrics

| Metric       | Result |
| ------------ | -----: |
| Recall@10    |      — |
| Recall@50    |      — |
| Precision@10 |      — |
| MRR          |      — |
| nDCG@10      |      — |
| mAP          |      — |

---

# Resume Highlights (After Benchmarking)

* Built a CUDA-accelerated semantic protein retrieval engine using ESM-2 embeddings and custom GPU kernels, enabling exact similarity search across hundreds of thousands of proteins.
* Implemented shared-memory tiled cosine similarity, warp-level Top-K selection, and candidate compression kernels optimized for high GPU occupancy and memory throughput.
* Profiled GPU performance using NVIDIA Nsight Compute, optimizing memory bandwidth, warp efficiency, and kernel latency through vectorized loads and shared-memory tiling.
* Integrated biological re-ranking with Smith–Waterman alignment to improve retrieval quality while limiting expensive sequence alignments to a small candidate set.
* Benchmarked end-to-end retrieval performance against PyTorch and FAISS baselines across varying database sizes, batch sizes, and mixed-precision configurations.

---

# License

MIT License

---

# Acknowledgments

* Meta AI for the ESM-2 protein language models.
* UniProt Consortium for Swiss-Prot and UniRef datasets.
* NVIDIA for CUDA, Nsight Compute, and GPU profiling tools.
* The Parasail and Striped Smith–Waterman projects for sequence alignment algorithms.
