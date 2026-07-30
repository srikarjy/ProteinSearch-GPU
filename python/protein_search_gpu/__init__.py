"""
ProteinSearch-GPU: CUDA-Accelerated Semantic Protein Retrieval
"""

from .encoder import (
    ESM2Encoder,
    encode_fasta,
    load_embeddings,
    load_esm2_model,
    get_model_info,
)

from .datasets import (
    parse_fasta,
    sequence_to_indices,
    indices_to_sequence,
    ProteinDataset,
    ProteinPairDataset,
    PfamDataset,
    get_dataloader,
    load_swissprot,
    load_uniref50,
    create_train_val_split,
    collate_protein_batch,
)

from .benchmark import (
    benchmark_l2_normalize,
    benchmark_cosine_similarity,
    benchmark_topk,
    benchmark_gather,
    benchmark_smith_waterman,
    benchmark_pytorch_cosine,
    benchmark_pytorch_topk,
    run_full_benchmark,
    profile_memory,
)

from .evaluate import (
    recall_at_k,
    precision_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    average_precision,
    mean_average_precision,
    compute_retrieval_metrics,
    evaluate_pfam_families,
    evaluate_uniprot_annotations,
    exact_match_recovery,
    print_metrics,
    save_metrics,
    load_metrics,
)

__version__ = "0.1.0"