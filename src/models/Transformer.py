import torch
import torch.nn as nn

from configs.model import ModelConfig
from .Multi_head_attention import MultiHeadAttention
from .Feed_forward import FeedForward

class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.MultiHeadAttention: MultiHeadAttention = MultiHeadAttention(config)
        self.FeedForward:FeedForward = FeedForward(config)
        self.Normalization_1 = nn.LayerNorm(config.d_model)
        self.Normalization_2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self,x: torch.Tensor)-> torch.Tensor:  
        
        residual = x
        x = self.Normalization_1(x)
        x = self.MultiHeadAttention(x)
        x = self.dropout(x)
        x = x + residual

        residual = x
        x = self.Normalization_2(x)
        x = self.FeedForward(x)
        x = self.dropout(x)
        x = x + residual

        return x

if ("__main__" == __name__):
    config = ModelConfig()
    x = torch.randn(2, 4, 512)
    model: Transformer = Transformer(config)
    output = model(x)
    print(output.shape)
