import torch
import torch.nn as nn

# by AI

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: int = 10000):
        super().__init__()
        self.head_dim = head_dim
        self.base = base

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        """
        q, k: (B, num_heads, T, head_dim)
        """
        B, H, T, D = q.shape
        device = q.device

        assert D == self.head_dim
        assert D % 2 == 0, "RoPE requires even head_dim"

        # positions
        pos = torch.arange(T, device=device).float()

        # inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2, device=device).float() / D))

        # angles: (T, D/2)
        theta = torch.einsum("t,d->td", pos, inv_freq)

        sin = theta.sin()[None, None, :, :]
        cos = theta.cos()[None, None, :, :]

        q = self._apply_rope(q, sin, cos)
        k = self._apply_rope(k, sin, cos)

        return q, k

    def _apply_rope(self, x, sin, cos):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        x_rotated = torch.stack(
            (-x2, x1), dim=-1
        ).reshape_as(x)

        return (x * cos.repeat_interleave(2, dim=-1)) + \
               (x_rotated * sin.repeat_interleave(2, dim=-1))
