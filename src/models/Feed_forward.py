import torch
import torch.nn as nn

from configs.model import ModelConfig
config = ModelConfig()

class FeedForward(nn.Module):

    def __init__(self,config : ModelConfig):
        super().__init__()
    
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_model*4),
            nn.GELU(),
            nn.Linear(config.d_model*4, config.d_model)
        )

    def forward(self,x: torch.Tensor) -> torch.Tensor:  
        
        return self.feed_forward(x)

if __name__ == "__main__":
    model = FeedForward(config)
    x = torch.randn(10, 20, 512)  
    output = model(x)
    print(output.shape)  
    