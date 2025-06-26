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


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        self.weights = torch.nn.Parameter(torch.ones(self.d_model))

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        original_dtype = x.dtype
        x.to(torch.float32)

        squared: Float[torch.Tensor, "... d_model"] = torch.pow(x, 2)
        rms: Float[torch.Tensor, "... 1"] = torch.sqrt(self.eps + (squared.sum(dim=-1, keepdim=True)) / self.d_model)
        weighted: Float[torch.Tensor, "... d_model"] = einsum(
            x / rms,
            self.weights,
            "... d_model, d_model -> ... d_model"
        )

        return weighted.to(original_dtype)


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: Optional[int] = None, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff or (math.ceil(self.d_model / 64.) * 64)

        self.w1 = torch.nn.Parameter(torch.randn(self.d_ff, self.d_model))
        self.w2 = torch.nn.Parameter(torch.randn(self.d_model, self.d_ff))
        self.w3 = torch.nn.Parameter(torch.randn(self.d_ff, self.d_model))

        self.einsum = False

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        if self.einsum:
            p1: Float[torch.Tensor, "... d_ff"] = einsum(x, self.w1, "... d_model, d_ff d_model-> ... d_ff")
            p1 = einsum(p1, torch.sigmoid(p1), "... d_ff, ... d_ff -> ... d_ff")
            p3: Float[torch.Tensor, "... d_ff"] = einsum(x, self.w3, "... d_model, d_ff d_model -> ... d_ff")
            inner_product: Float[torch.Tensor, "... d_ff"] = einsum(p1, p3, "... d_ff, ... d_ff -> ... d_ff")
            return einsum(inner_product, self.w2, "... d_ff, d_model d_ff -> ... d_model")
        else:
            p1 = x @ self.w1.T
            p1 *= torch.sigmoid(p1)
            p3 = x @ self.w3.T
            inner_product = p1 * p3
            return inner_product @ self.w2.T