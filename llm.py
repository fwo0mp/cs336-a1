import math
from typing import Optional

from einops import einsum, rearrange
from jaxtyping import Float, Int, Bool
import torch
from torch import Tensor

class Linear(torch.nn.Module):
    def __init__(self, dim_in: int, dim_out: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.device = device

        self.weights = torch.nn.Parameter(torch.zeros(self.dim_out, self.dim_in, device=self.device, dtype=dtype))
        self.variance = (2 / (self.dim_in + self.dim_out))
        torch.nn.init.trunc_normal_(self.weights, mean=0, std=math.sqrt(self.variance), a=-3*self.variance, b=3*self.variance)

    def forward(self, x: Float[Tensor, "... dim_in"]) -> Float[Tensor, "... dim_out"]:
        result: Float[Tensor, "... dim_out"] = einsum(x, self.weights, "... dim_in, dim_out dim_in -> ... dim_out")
        return result

class Embedding(torch.nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.device = device

        self.embeddings = torch.nn.Parameter(torch.zeros(self.vocab_size, self.embedding_dim, device=self.device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.embeddings, mean=0, std=1, a=-3, b=3)

    def forward(self, x: Int[Tensor, "..."]) -> Float[Tensor, "... embedding_dim"]:
        input_onehot: Int[Tensor, "... vocab_size"] = rearrange(x, "... -> ... 1") == torch.arange(self.vocab_size)
        result: Float[Tensor, "... embedding_dim"] = einsum(
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

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        original_dtype = x.dtype
        x.to(torch.float32)

        squared: Float[Tensor, "... d_model"] = torch.pow(x, 2)
        rms: Float[Tensor, "... 1"] = torch.sqrt(self.eps + (squared.sum(dim=-1, keepdim=True)) / self.d_model)
        weighted: Float[Tensor, "... d_model"] = einsum(
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

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        if self.einsum:
            p1: Float[Tensor, "... d_ff"] = einsum(x, self.w1, "... d_model, d_ff d_model-> ... d_ff")
            p1 = einsum(p1, torch.sigmoid(p1), "... d_ff, ... d_ff -> ... d_ff")
            p3: Float[Tensor, "... d_ff"] = einsum(x, self.w3, "... d_model, d_ff d_model -> ... d_ff")
            inner_product: Float[Tensor, "... d_ff"] = einsum(p1, p3, "... d_ff, ... d_ff -> ... d_ff")
            return einsum(inner_product, self.w2, "... d_ff, d_model d_ff -> ... d_model")
        else:
            p1 = x @ self.w1.T
            p1 *= torch.sigmoid(p1)
            p3 = x @ self.w3.T
            inner_product = p1 * p3
            return inner_product @ self.w2.T


class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, context_length: int, device: Optional[torch.device] = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.context_length = context_length
        self.device = device

        i: Int[Tensor, "context_length"] = torch.arange(self.context_length)
        
        # !!! according to doc these should have all been + 1 to be one-indexed instead of zero-indexed?
        # but in order to match the test cases we have to remove the +1 at the end
        k: Int[Tensor, "d_k"] = (torch.arange(self.d_k) // 2)

        thetas: Float[Tensor, "context_length d_k"] = i.unsqueeze(1) / torch.pow(self.theta, (2 * k) / self.d_k)

        # pre-cached 2-d arrays with all necessary cos/sin values precomputed
        cos: Float[Tensor, "context_length d_k"] = torch.cos(thetas)
        sin: Float[Tensor, "context_length d_k"] = torch.sin(thetas)
        even_signs: Int[Tensor, "d_k"] = torch.where(torch.arange(self.d_k) % 2 == 0, -1, 1)

        # apply sign swaps to sin buffer before saving because it will be the same every time
        sin = sin * even_signs

        # indices to use for slicing when swapping each pair of values for sin multiplication
        pair_increments: Int[Tensor, "d_k"] = torch.where(torch.arange(self.d_k) % 2 == 0, 1, -1)
        pair_swaps: Int[Tensor, "d_k"] = torch.arange(self.d_k) + pair_increments

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("pair_swaps", pair_swaps, persistent=False)

    def forward(self, x: Float[Tensor, "... seq_len d_k"], token_positions: Int[Tensor, "... seq_len"]) -> Float[Tensor, "... seq_len d_k"]:
        cos_slice: Float[Tensor, "... seq_len d_k"] = self.get_buffer("cos")[token_positions,:]
        sin_slice: Float[Tensor, "... seq_len d_k"] = self.get_buffer("sin")[token_positions,:]

        # swap each pair of indices before multiplying with sins
        swapped_pairs: Float[Tensor, "... seq_len d_k"] = x[...,self.get_buffer("pair_swaps")]

        """
        for any fixed token position i, we will get results like the following:
            index 0: x_0 * cos(theta_ik) + x_1 * sin(theta_ik)
            index 1: x_1 * cos(theta_ik) - x_0 * sin(theta_ik)
            index 2: x_2 * cos(theta_ik) + x_3 * sin(theta_ik)
            index 3: x_3 * cos(theta_ik) - x_2 * sin(theta_ik)
            ...

            in other words, each index should have its own value times the relevant cos, and its "paired" index
            multiplied by sin, but with alternating signs for the sin term.

            we apply the sign swaps when constructing the main sin buffer, and use the swapped_pairs call to
            align each paired element for this addition
        """
        result: Float[Tensor, "... seq_len d_k"] = x * cos_slice + swapped_pairs * sin_slice
        return result


def softmax(x: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    max_values = x.max(dim=dim, keepdim=True).values
    adjusted: Float[Tensor, "..."] = x - max_values
    exps: Float[Tensor, "..."] = torch.exp(adjusted)
    denoms = exps.sum(dim=dim, keepdim=True)
    return exps / denoms


def attention(
        queries: Float[Tensor, "... query_len d_k"],
        keys: Float[Tensor, "... key_len d_k"],
        values: Float[Tensor, "... key_len d_v"],
        mask: Optional[Bool[Tensor, "query_len key_len"]] = None,
        ) -> Float[Tensor, "... d_v"]:
    assert keys.shape[-2] == values.shape[-2]

    d_k: int = keys.shape[-1]
    product: Float[Tensor, "... seq_len seq_len"] = einsum(
        queries,
        keys,
        "... query_len d_k, ... key_len d_k -> ... query_len key_len"
    )
    scaled_product: Float[Tensor, "... seq_len seq_len"] = product / math.sqrt(d_k)
    masked_product: Float[Tensor, "... seq_len seq_len"] = scaled_product
    if mask is not None:
        masked_product = masked_product.where(mask, -torch.inf)

    result: Float[Tensor, "... query_len d_v"] = einsum(
        softmax(masked_product, dim=-1),
        values,
        "... query_len key_len, ... key_len d_v -> ... query_len d_v"
    )

    return result
    