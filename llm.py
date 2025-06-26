import math
from typing import Optional

from einops import einsum, rearrange
from jaxtyping import Float, Int
import torch

class Linear(torch.nn.Module):
    def __init__(self, dim_in: int, dim_out: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.device = device

        self.weights = torch.nn.Parameter(torch.zeros(self.dim_out, self.dim_in, device=self.device, dtype=dtype))
        self.variance = (2 / (self.dim_in + self.dim_out))
        torch.nn.init.trunc_normal_(self.weights, mean=0, std=math.sqrt(self.variance), a=-3*self.variance, b=3*self.variance)

    def forward(self, x: Float[torch.Tensor, "... dim_in"]) -> Float[torch.Tensor, "... dim_out"]:
        result: Float[torch.Tensor, "... dim_out"] = einsum(x, self.weights, "... dim_in, dim_out dim_in -> ... dim_out")
        return result

class Embedding(torch.nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.device = device

        self.embeddings = torch.nn.Parameter(torch.zeros(self.vocab_size, self.embedding_dim, device=self.device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.embeddings, mean=0, std=1, a=-3, b=3)

    def forward(self, x: Int[torch.Tensor, "..."]) -> Float[torch.Tensor, "... embedding_dim"]:
        input_onehot: Int[torch.Tensor, "... vocab_size"] = rearrange(x, "... -> ... 1") == torch.arange(self.vocab_size)
        result: Float[torch.Tensor, "... embedding_dim"] = einsum(
            input_onehot.to(torch.float),
            self.embeddings,
            "... vocab_size, vocab_size embedding_dim -> ... embedding_dim"
        )
        return result