import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt
from configs.model import ModelConfig
from .RoPE import RotaryEmbedding

class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        assert config.d_model % config.num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = config.d_model
        self.num_heads:int = config.num_heads
        self.head_dim:int = config.d_model // config.num_heads
        self.dropout:float = config.dropout

        self.rope = RotaryEmbedding(head_dim=self.head_dim)
        # Fused QKV projection
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.qkv_bias)

        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.attn_dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(config.context_len, config.context_len), diagonal=1).bool()
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv(x)          

        q, k, v = qkv.chunk(3, dim=-1)

        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = self.rope(q, k)
        
        if hasattr(F, "scaled_dot_product_attention"):
            
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
            
        else:

            attn_scores = torch.matmul(q, k.transpose(-2, -1))
            attn_scores = attn_scores / sqrt(self.head_dim)

            causal_mask = self.mask[:T, :T]
            attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

            attn_weights = torch.softmax(attn_scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)

            out = torch.matmul(attn_weights, v)

        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.out_proj(out)

        return out
    

if __name__ == "__main__":
    
    config = ModelConfig()
    x = torch.randn(1, 3, 512)
    attn = MultiHeadAttention(config)
    y = attn(x)
    print(y.shape)  
