from dataclasses import dataclass
import torch
from pathlib import Path


@dataclass
class TrainConfig():

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    epochs: int = 2                 
    grad_clip: float = 1.0
    lr: float = 2e-4
    weight_decay: float = 0.01

    # 🔥 memory + stability
    use_fp16: bool = True
    gradient_accumulation_steps: int = 8

    context_length: int = 512            
    batch_size: int = 12                  # real batch --
    chunk_tokens: int = 99_999_744 * 3   #199_999_488  streamed tokens per run
    chunk_id = None
    token_folder = "../fineweb_tokens"
    num_workers = 0

    checkpoint_every_tokens: int = 1_000_000
    max_checkpoints_per_run: int = 4
    ckpt_dir: Path = Path("checkpoints")
