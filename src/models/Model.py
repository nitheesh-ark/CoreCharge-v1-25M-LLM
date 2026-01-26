import torch
import torch.nn as nn
from .Transformer import Transformer
from configs.model import ModelConfig


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config


        self.word_embedding = nn.Embedding(
            config.vocab_size, config.d_model
        )

        self.initial_dropout = nn.Dropout(config.dropout)


        self.transformer_layers = nn.Sequential(
            *[Transformer(config=config) for _ in range(config.num_layers)]
        )


        self.final_normalization = nn.LayerNorm(config.d_model)


        self.final_linear_layer = nn.Linear(
            config.d_model, config.vocab_size, bias=False
        )


        if config.tie_embeddings:
            self.final_linear_layer.weight = self.word_embedding.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.word_embedding(x)
        x = self.initial_dropout(x)

        x = self.transformer_layers(x)
        x = self.final_normalization(x)

        logits = self.final_linear_layer(x)
        return logits


if __name__ == "__main__":
    config = ModelConfig()
    model = Model(config)

    idx = torch.randint(0, config.vocab_size, (2, 4))
    logits = model(idx)

    print("Logits shape:", logits.shape)
    print("Tied embeddings:", 
          model.final_linear_layer.weight is model.word_embedding.weight)
