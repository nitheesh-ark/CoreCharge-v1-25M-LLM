import torch
from pathlib import Path


def save_checkpoint(
    path,
    model,
    optimizer,
    step,
    tokens_seen,
    chunk_id,
):
    path = Path(path)

    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "chunk_id": chunk_id,
    }

    torch.save(state, path)


def load_checkpoint(
    path,
    model,
    optimizer,
    device="cpu",
):
    path = Path(path)

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
