import torch
import torch.nn.functional as F
from configs.inference import InferenceConfig


class Inference:
    def __init__(self, model, tokenizer, config: InferenceConfig):
        self.model = model.to(config.device)
        self.tokenizer = tokenizer
        self.config = config
        self.device = config.device

        self.model.eval()

    @torch.no_grad()
    def run(self, prompt: str) -> str:
        idx = torch.tensor(
            [self.tokenizer.encode(prompt)],
            dtype=torch.long,
            device=self.device,
        )

        # 🔒 FIXED max_new_token loop
        for _ in range(self.config.max_new_token):
            idx_cond = idx[:, -self.config.context_len:]

            logits = self.model(idx_cond)
            logits = logits[:, -1, :]
            logits = logits / self.config.temperature

            values, indices = torch.topk(logits, self.config.top_k)
            probs = F.softmax(values, dim=-1)

            next_token = indices.gather(
                -1,
                torch.multinomial(probs, 1)
            )

            idx = torch.cat((idx, next_token), dim=1)

        return self.tokenizer.decode(idx[0].tolist())

