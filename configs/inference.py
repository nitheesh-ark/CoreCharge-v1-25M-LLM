from dataclasses import dataclass

import torch

@dataclass(frozen=True)
class Device:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
@dataclass(frozen=True)
class InferenceConfig(Device):
    context_len: int = 512
    max_new_token: int = 20
    temperature: float = 0.98
    top_k: int = 50
    top_p: float = 0.95
    use_kv_cache: bool = True

if ("__main__" == __name__):
    config = InferenceConfig()
    print(config)