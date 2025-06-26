import math
from typing import Optional

from einops import einsum
from jaxtyping import Float
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

    