import torch
import protein_search_gpu as psg
from protein_search_gpu.datasets import parse_fasta, ProteinDataset

# Create a small test FASTA
with open('test.fasta', 'w') as f:
    f.write('>seq1\nACDEFGHIKLMNPQRSTVWY\n>seq2\nACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY\n')

ids, seqs = parse_fasta('test.fasta')
print(f'Parsed {len(seqs)} sequences')
print(f'Lengths: {[len(s) for s in seqs]}')

# Test ProteinDataset
ds = ProteinDataset('test.fasta', max_len=128)
print(f'Dataset size: {len(ds)}')
sample = ds[0]
print(f'Sample type: {type(sample)}')
print(f'Sample shape: {sample.shape}')
print(f'Sample: {sample[:10]}')

# Test with return_ids
ds2 = ProteinDataset('test.fasta', max_len=128, return_ids=True)
sample2 = ds2[0]
print(f'Sample with ids: {sample2[0].shape}, {sample2[1]}')

print("Dataset tests passed!")