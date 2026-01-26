# src/data/token_dataset.py

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import IterableDataset

class TokenChunkDataset(IterableDataset):
    def __init__(
        self,
        folder: str,
        chunk_id: int,
        seq_len: int,
        stride: int | None = None,
    ):
        """
        folder: path to fineweb_tokens
        chunk_id: which chunk to train (e.g. 1 for chunk_001.npy)
        seq_len: model context length
        stride: step size between sequences (defaults to seq_len)
        """
        self.folder = Path(folder)
        self.chunk_id = chunk_id
        self.seq_len = seq_len
        self.stride = stride or seq_len

        self.chunk_path = self.folder / f"chunk_{chunk_id:03d}.npy"
        if not self.chunk_path.exists():
            raise FileNotFoundError(self.chunk_path)

    def __iter__(self):
        tokens = np.load(self.chunk_path, mmap_mode="r")

        max_start = len(tokens) - self.seq_len - 1

        for start in range(0, max_start, self.stride):
            x = torch.from_numpy(
                tokens[start : start + self.seq_len].astype("int64")
            )
            y = torch.from_numpy(
                tokens[start + 1 : start + self.seq_len + 1].astype("int64")
            )

            yield x, y
