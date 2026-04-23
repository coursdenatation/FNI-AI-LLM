"""
Phase 3.4 - Data Loaders
Batching, shuffling, train/val/test splits for all Cameroon languages
"""

import numpy as np
import os
from src.year3.data_processing.dataset import TextDataset


class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset    = dataset
        self.batch_size = batch_size
        self.shuffle    = shuffle

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start: start + self.batch_size]
            if len(batch_idx) < self.batch_size:
                continue  # drop last incomplete batch

            X_batch = np.stack([self.dataset[i][0] for i in batch_idx])
            y_batch = np.stack([self.dataset[i][1] for i in batch_idx])
            yield X_batch, y_batch

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __repr__(self):
        return (f"DataLoader(samples={len(self.dataset)}, "
                f"batch_size={self.batch_size}, "
                f"batches={len(self)})")


def train_val_test_split(dataset, train=0.8, val=0.1, seed=42):
    """Split dataset into train/val/test subsets"""
    np.random.seed(seed)
    n = len(dataset)
    indices = np.random.permutation(n)

    train_end = int(n * train)
    val_end   = int(n * (train + val))

    class Subset:
        def __init__(self, ds, idx):
            self.ds  = ds
            self.idx = idx
            self.tokenizer  = ds.tokenizer
            self.vocab_size = ds.vocab_size
            self.language   = ds.language

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, i):
            return self.ds[self.idx[i]]

    return (Subset(dataset, indices[:train_end]),
            Subset(dataset, indices[train_end:val_end]),
            Subset(dataset, indices[val_end:]))


if __name__ == "__main__":
    DATA_ROOT = "data/cameroon_languages"
    LANGUAGES = ["english", "french", "bayangi", "douala"]

    print("=== DATA LOADERS ===\n")
    for lang in LANGUAGES:
        path = os.path.join(DATA_ROOT, lang, "processed", f"{lang}_clean.txt")
        if not os.path.exists(path):
            continue

        ds = TextDataset(language=lang, seq_len=32)
        ds.load(path)

        train_ds, val_ds, test_ds = train_val_test_split(ds)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False)

        print(f"{lang.upper()}")
        print(f"  Total samples: {len(ds)}")
        print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
        print(f"  Train loader: {train_loader}")

        # Test one batch
        for X, y in train_loader:
            print(f"  Batch X shape: {X.shape}, y shape: {y.shape}")
            break
        print()

    print("Done: DataLoaders complete.")
