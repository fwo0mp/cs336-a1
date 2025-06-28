import math
from typing import Callable, Dict, Optional, Tuple
import torch

from einops import einsum, rearrange
from torch import Tensor
from jaxtyping import Float, Int


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float, betas: Tuple[float, float], weight_decay: float, eps: float=1e-8):
        defaults: Dict[str, float] = {
            "lr": lr,
            "b1": betas[0],
            "b2": betas[1],
            "wdr": weight_decay,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            wdr = group["wdr"]
            b1 = group["b1"]
            b2 = group["b2"]
            eps = group["eps"]

            param: torch.nn.Parameter
            for param in group["params"]:
                if param.grad is None:
                    continue
                    
                param_state = self.state[param]
                t = param_state.get("t", 1)
                grad = param.grad.data

                param_state["m1"] = b1 * param_state.get("m1", 0) + (1 - b1) * grad
                param_state["m2"] = b2 * param_state.get("m2", 0) + (1 - b2) * (grad ** 2)

                # time step learning rate
                lr_t = lr * math.sqrt(1 - math.pow(b2, t)) / (1 - math.pow(b1, t))

                # core param update
                param.data -= lr_t * param_state["m1"] / (torch.sqrt(param_state["m2"]) + eps)

                # apply weight decay
                param.data -= param.data * lr * wdr
                
                # increment time step
                param_state["t"] = t + 1

        return loss