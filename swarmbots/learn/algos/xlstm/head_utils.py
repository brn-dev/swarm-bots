import torch
from torch import nn
from torch.nn import functional as F


def require_heads_divide_hidden_dim(*, hidden_dim: int, num_heads: int) -> None:
    if hidden_dim % num_heads != 0:
        raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")


def init_linspace_bias_(bias: torch.Tensor, *, start: float, end: float) -> torch.Tensor:
    values = torch.linspace(start, end, bias.numel(), device=bias.device, dtype=bias.dtype)
    with torch.no_grad():
        bias.copy_(values)
    return bias


class MultiHeadLayerNorm(nn.Module):

    def __init__(self, *, hidden_dim: int, num_heads: int, eps: float = 1e-5, bias: bool = False) -> None:
        super().__init__()
        require_heads_divide_hidden_dim(hidden_dim=hidden_dim, num_heads=num_heads)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim)) if bias else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError(f"Expected inputs shape (B, NH, T, DH), got {tuple(inputs.shape)}")
        batch_size, num_heads, sequence_length, head_dim = inputs.shape
        normalized_inputs = inputs.transpose(1, 2).reshape(batch_size * sequence_length, num_heads * head_dim)
        outputs = F.group_norm(
            normalized_inputs,
            num_groups=num_heads,
            weight=self.weight,
            bias=self.bias,
            eps=self.eps,
        )
        return outputs.reshape(batch_size, sequence_length, num_heads, head_dim).transpose(1, 2)

