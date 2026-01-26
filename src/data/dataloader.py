from pathlib import Path
from torch.utils.data import DataLoader
from .token_dataset import TokenChunkDataset


def build_dataloader(
    token_folder: str,
    chunk_id: int,
    seq_len: int,
    batch_size: int,
    num_workers: int = 0,
):

    chunk_file = Path(token_folder) / f"chunk_{chunk_id:03d}.npy"
    print(f"📦 Loading token chunk: {chunk_file}")

    dataset = TokenChunkDataset(
        folder=token_folder,
        chunk_id=chunk_id,
        seq_len=seq_len,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
