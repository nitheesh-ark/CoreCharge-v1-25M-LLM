from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class Device:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass(frozen=True)
class ModelConfig(Device):
    d_model: int = 384
    num_heads: int = 6
    num_layers: int = 8
    context_len: int = 512
    dropout: float = 0.1
    qkv_bias: bool = False
    vocab_size: int = 32000
    rotary: bool = True
    tie_embeddings: bool = True

if __name__ == "__main__":
    print(ModelConfig())
