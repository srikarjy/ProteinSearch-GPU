import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional, Iterator
import os
import gzip
import random
from collections import defaultdict


AA_TO_IDX = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
    'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
    'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
    'X': 20, 'U': 21, 'O': 22, 'B': 23, 'Z': 24, 'J': 25
}

IDX_TO_AA = {v: k for k, v in AA_TO_IDX.items()}


def parse_fasta(filepath: str) -> List[Tuple[str, str]]:
    sequences = []
    current_id = None
    current_seq = []
    
    open_func = gzip.open if filepath.endswith('.gz') else open
    
    with open_func(filepath, 'rt') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_id is not None:
                    sequences.append((current_id, ''.join(current_seq)))
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line.upper())
        
        if current_id is not None:
            sequences.append((current_id, ''.join(current_seq)))
    
    return sequences


def sequence_to_indices(seq: str, max_len: Optional[int] = None) -> torch.Tensor:
    indices = [AA_TO_IDX.get(aa, AA_TO_IDX['X']) for aa in seq]
    tensor = torch.tensor(indices, dtype=torch.int8)
    
    if max_len is not None:
        if len(tensor) > max_len:
            tensor = tensor[:max_len]
        else:
            padding = torch.full((max_len - len(tensor),), AA_TO_IDX['X'], dtype=torch.int8)
            tensor = torch.cat([tensor, padding])
    
    return tensor


def indices_to_sequence(indices: torch.Tensor) -> str:
    return ''.join(IDX_TO_AA.get(idx.item(), 'X') for idx in indices)


class ProteinDataset(Dataset):
    def __init__(
        self,
        fasta_path: str,
        max_len: int = 1024,
        return_ids: bool = False
    ):
        self.max_len = max_len
        self.return_ids = return_ids
        self.sequences = parse_fasta(fasta_path)
        
        self.valid_sequences = []
        for seq_id, seq in self.sequences:
            clean_seq = ''.join(c for c in seq if c in AA_TO_IDX)
            if clean_seq:
                self.valid_sequences.append((seq_id, clean_seq))
        
        print(f"Loaded {len(self.valid_sequences)} valid sequences from {fasta_path}")
    
    def __len__(self) -> int:
        return len(self.valid_sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        seq_id, seq = self.valid_sequences[idx]
        indices = sequence_to_indices(seq, self.max_len)
        
        if self.return_ids:
            return indices, seq_id
        return indices


class ProteinPairDataset(Dataset):
    def __init__(
        self,
        query_fasta: str,
        db_fasta: str,
        max_len: int = 1024,
        max_pairs: Optional[int] = None
    ):
        self.max_len = max_len
        self.query_seqs = parse_fasta(query_fasta)
        self.db_seqs = parse_fasta(db_fasta)
        
        self.query_valid = [(id_, ''.join(c for c in s if c in AA_TO_IDX)) 
                           for id_, s in self.query_seqs if s]
        self.db_valid = [(id_, ''.join(c for c in s if c in AA_TO_IDX)) 
                        for id_, s in self.db_seqs if s]
        
        self.pairs = []
        for i, (q_id, q_seq) in enumerate(self.query_valid):
            for j, (d_id, d_seq) in enumerate(self.db_valid):
                self.pairs.append((i, j, q_id, d_id, q_seq, d_seq))
                if max_pairs and len(self.pairs) >= max_pairs:
                    break
            if max_pairs and len(self.pairs) >= max_pairs:
                break
        
        random.shuffle(self.pairs)
        print(f"Created {len(self.pairs)} pairs")
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        q_idx, d_idx, q_id, d_id, q_seq, d_seq = self.pairs[idx]
        q_tensor = sequence_to_indices(q_seq, self.max_len)
        d_tensor = sequence_to_indices(d_seq, self.max_len)
        return q_tensor, d_tensor, q_id, d_id


class PfamDataset(Dataset):
    def __init__(
        self,
        pfam_dir: str,
        split: str = 'train',
        max_len: int = 1024
    ):
        self.max_len = max_len
        self.split = split
        self.families = defaultdict(list)
        
        for family in os.listdir(pfam_dir):
            family_dir = os.path.join(pfam_dir, family)
            if not os.path.isdir(family_dir):
                continue
                
            for split_file in ['train.fasta', 'test.fasta', 'dev.fasta']:
                filepath = os.path.join(family_dir, split_file)
                if os.path.exists(filepath):
                    seqs = parse_fasta(filepath)
                    for seq_id, seq in seqs:
                        clean = ''.join(c for c in seq if c in AA_TO_IDX)
                        if clean:
                            self.families[family].append((seq_id, clean))
        
        self.all_sequences = []
        self.labels = []
        for family_idx, (family, seqs) in enumerate(self.families.items()):
            for seq_id, seq in seqs:
                self.all_sequences.append(seq)
                self.labels.append(family_idx)
        
        print(f"Loaded {len(self.all_sequences)} sequences from {len(self.families)} families")
    
    def __len__(self) -> int:
        return len(self.all_sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        seq = self.all_sequences[idx]
        label = self.labels[idx]
        indices = sequence_to_indices(seq, self.max_len)
        return indices, label


def collate_protein_batch(batch: List[Tuple]) -> Tuple[torch.Tensor, ...]:
    if len(batch[0]) == 2:
        sequences, labels = zip(*batch)
        return torch.stack(sequences), torch.tensor(labels)
    elif len(batch[0]) == 4:
        q_seqs, d_seqs, q_ids, d_ids = zip(*batch)
        return torch.stack(q_seqs), torch.stack(d_seqs), list(q_ids), list(d_ids)
    else:
        return torch.stack(batch)


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_protein_batch
    )


def load_swissprot(
    filepath: str,
    max_len: int = 1024,
    max_seqs: Optional[int] = None
) -> Tuple[List[str], List[str]]:
    sequences = parse_fasta(filepath)
    
    ids = []
    seqs = []
    
    for seq_id, seq in sequences:
        clean = ''.join(c for c in seq if c in AA_TO_IDX)
        if clean:
            ids.append(seq_id)
            seqs.append(clean)
            if max_seqs and len(seqs) >= max_seqs:
                break
    
    return ids, seqs


def load_uniref50(
    filepath: str,
    max_len: int = 1024,
    max_seqs: Optional[int] = None
) -> Tuple[List[str], List[str]]:
    return load_swissprot(filepath, max_len, max_seqs)


def create_train_val_split(
    ids: List[str],
    seqs: List[str],
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str], List[str]]:
    random.seed(seed)
    indices = list(range(len(seqs)))
    random.shuffle(indices)
    
    val_size = int(len(seqs) * val_ratio)
    val_indices = set(indices[:val_size])
    
    train_ids = [ids[i] for i in indices if i not in val_indices]
    train_seqs = [seqs[i] for i in indices if i not in val_indices]
    val_ids = [ids[i] for i in val_indices]
    val_seqs = [seqs[i] for i in val_indices]
    
    return train_ids, train_seqs, val_ids, val_seqs


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test dataset loading")
    parser.add_argument("--fasta", type=str, required=True, help="FASTA file path")
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    
    args = parser.parse_args()
    
    dataset = ProteinDataset(args.fasta, max_len=args.max_len)
    loader = get_dataloader(dataset, batch_size=args.batch_size)
    
    for batch in loader:
        print(f"Batch shape: {batch.shape}")
        print(f"Sample: {indices_to_sequence(batch[0])[:50]}...")
        break