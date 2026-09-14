import random
import torch
from torch.utils.data import Dataset
from configs.config import PRIME_P

OPERATORS = {
    "x*y": (lambda a, b, p: (a * b) % p, "*", True),
    "x+y": (lambda a, b, p: (a + b) % p, "+", True),
    "x-y": (lambda a, b, p: (a - b) % p, "-", False),
    "x2+y2": (lambda a, b, p: (a**2 + b**2) % p, "^", True),
}

NUM_TOKENS = [f"N_{i}" for i in range(PRIME_P)]
SPECIAL_TOKENS = ['+', '-', '*', '/', '^', '=', '<PAD>', '<SOS>', '<EOS>']
VOCAB = NUM_TOKENS + SPECIAL_TOKENS
CHAR2IDX = {ch: idx for idx, ch in enumerate(VOCAB)}
IDX2CHAR = {idx: ch for idx, ch in enumerate(VOCAB)}

def generate_datasets(p=PRIME_P, op_name="x*y", train_fraction=0.15, seed=42):
    """Sinh dữ liệu chống Leakage cho bài toán Grokking."""
    random.seed(seed)
    op_fn, op_symbol, is_symmetric = OPERATORS[op_name]

    if is_symmetric:
        unique_pairs = [(a, b) for a in range(p) for b in range(a, p)]
    else:
        unique_pairs = [(a, b) for a in range(p) for b in range(p)]

    random.shuffle(unique_pairs)
    train_size = int(len(unique_pairs) * train_fraction)
    train_pairs = set(unique_pairs[:train_size])
    val_pairs = set(unique_pairs[train_size:])

    def build_samples(pair_set):
        samples = []
        for a, b in pair_set:
            y = op_fn(a, b, p)
            seq1 = ['<SOS>', f"N_{a}", op_symbol, f"N_{b}", '=', f"N_{y}", '<EOS>']
            samples.append(torch.tensor([CHAR2IDX[ch] for ch in seq1], dtype=torch.long))

            if is_symmetric and a != b:
                seq2 = ['<SOS>', f"N_{b}", op_symbol, f"N_{a}", '=', f"N_{y}", '<EOS>']
                samples.append(torch.tensor([CHAR2IDX[ch] for ch in seq2], dtype=torch.long))
        return samples

    train_data = build_samples(train_pairs)
    val_data = build_samples(val_pairs)
    random.shuffle(train_data)
    random.shuffle(val_data)

    print(f"Data '{op_name}' (p={p}) | Train: {len(train_data)} | Val: {len(val_data)}")
    return train_data, val_data

class FixedDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]