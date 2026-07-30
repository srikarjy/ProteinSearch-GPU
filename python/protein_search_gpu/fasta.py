import gzip
from typing import List, Tuple, Iterator


def read_fasta(filepath: str) -> Tuple[List[str], List[str]]:
    """Read FASTA file and return (ids, sequences)."""
    ids = []
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
                    ids.append(current_id)
                    sequences.append(''.join(current_seq))
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line.upper())
        
        if current_id is not None:
            ids.append(current_id)
            sequences.append(''.join(current_seq))
    
    return ids, sequences


def write_fasta(ids: List[str], sequences: List[str], filepath: str):
    """Write sequences to FASTA file."""
    with open(filepath, 'w') as f:
        for id_, seq in zip(ids, sequences):
            f.write(f'>{id_}\n')
            for i in range(0, len(seq), 80):
                f.write(f'{seq[i:i+80]}\n')


def parse_fasta_iter(filepath: str) -> Iterator[Tuple[str, str]]:
    """Iterate over FASTA records."""
    open_func = gzip.open if filepath.endswith('.gz') else open
    
    with open_func(filepath, 'rt') as f:
        current_id = None
        current_seq = []
        
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_id is not None:
                    yield current_id, ''.join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line.upper())
        
        if current_id is not None:
            yield current_id, ''.join(current_seq)


def filter_fasta(
    input_path: str,
    output_path: str,
    min_len: int = 0,
    max_len: int = float('inf'),
    valid_chars: str = 'ARNDCQEGHILKMFPSTWYVXUOBZJ'
):
    """Filter FASTA by length and valid characters."""
    valid_set = set(valid_chars)
    
    with open(output_path, 'w') as out:
        for id_, seq in parse_fasta_iter(input_path):
            if min_len <= len(seq) <= max_len:
                if all(c in valid_set for c in seq):
                    out.write(f'>{id_}\n')
                    for i in range(0, len(seq), 80):
                        out.write(f'{seq[i:i+80]}\n')


def split_fasta(
    input_path: str,
    train_path: str,
    val_path: str,
    test_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
):
    """Split FASTA into train/val/test."""
    import random
    random.seed(seed)
    
    records = list(parse_fasta_iter(input_path))
    random.shuffle(records)
    
    n = len(records)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    splits = {
        'train': records[:train_end],
        'val': records[train_end:val_end],
        'test': records[val_end:],
    }
    
    for name, path in [('train', train_path), ('val', val_path), ('test', test_path)]:
        with open(path, 'w') as f:
            for id_, seq in splits[name]:
                f.write(f'>{id_}\n')
                for i in range(0, len(seq), 80):
                    f.write(f'{seq[i:i+80]}\n')
        print(f"{name}: {len(splits[name])} sequences")


def get_fasta_stats(filepath: str) -> dict:
    """Get statistics about FASTA file."""
    lengths = []
    for _, seq in parse_fasta_iter(filepath):
        lengths.append(len(seq))
    
    if not lengths:
        return {'count': 0}
    
    return {
        'count': len(lengths),
        'min_len': min(lengths),
        'max_len': max(lengths),
        'mean_len': sum(lengths) / len(lengths),
        'total_aa': sum(lengths),
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='FASTA utilities')
    parser.add_argument('command', choices=['stats', 'filter', 'split'])
    parser.add_argument('input')
    parser.add_argument('--output', '-o')
    parser.add_argument('--min-len', type=int, default=0)
    parser.add_argument('--max-len', type=int, default=10000)
    parser.add_argument('--train', help='Train output path')
    parser.add_argument('--val', help='Val output path')
    parser.add_argument('--test', help='Test output path')
    
    args = parser.parse_args()
    
    if args.command == 'stats':
        stats = get_fasta_stats(args.input)
        for k, v in stats.items():
            print(f'{k}: {v}')
    elif args.command == 'filter':
        filter_fasta(args.input, args.output, args.min_len, args.max_len)
    elif args.command == 'split':
        split_fasta(args.input, args.train, args.val, args.test)