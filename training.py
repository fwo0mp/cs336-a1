import random
from typing import List, Tuple

import numpy as np
import torch

from einops import rearrange
from torch import Tensor
from jaxtyping import Float, Int

def cross_entropy(logits: Float[Tensor, "... batch_size vocab_size"], targets: Int[Tensor, "... batch_size"]) -> Float[Tensor, "..."]:
    """
        naive implementation calling softmax directly runs into numeric stability issues b/c w/ extremely
        small (effectively zero) probabilities for targets, -log(p) -> inf.  instead, after adjusting everything
        against the maximum value in each prediction, we restructure the calculation as follows:
        (i need to learn latex)

        (1) -log(softmax(x_t))
        (2) = -log(exp(x_t) / sum(exp(x_i)))
        (3) = -log(exp(x_t)) + log(sum(exp(x_i)))
        (4) = -x_t + log(sum(exp(x_i)))

        we still fully calculate the second term the same as in normal softmax, and because it's summing across all
        values for a row, the largest of which is guaranteed to be 1, that term is stable.  
    """
    assert logits.shape[-2] == targets.shape[-1]

    max_values: Float[Tensor, "... batch_size 1"] = logits.max(dim=-1, keepdim=True).values
    adjusted: Float[Tensor, "... batch_size vocab_size"] = logits - max_values
    exps: Float[Tensor, "... batch_size vocab_size"] = torch.exp(adjusted)
    
    # log of sum of exps, represents second term in (4) above
    softmax_denoms: Float[Tensor, "... batch_size 1"] = torch.log(exps.sum(dim=-1, keepdim=True))

    # just select x_t (*after* adjusting for max value), represents first term in (4) above
    target_logits: Float[Tensor, "... batch_size 1"] = adjusted.gather(-1, targets.unsqueeze(1))

    # (4)
    element_cross_entropy: Float[Tensor, "... batch_size 1"] = softmax_denoms - target_logits

    element_cross_entropy = rearrange(element_cross_entropy, "... batch_size 1 -> ... batch_size")
    result: Float[Tensor, "..."] = element_cross_entropy.mean(dim=-1)
    return result


def create_training_batch(tokens: np.typing.NDArray, batch_size: int, context_length: int, device: str) -> Tuple[Tensor, Tensor]:
    """
        returns: (tensor of input tokens w/ shape (batch_size, context_length), tensor of target next tokens w/ same shape)

        given signature of function, which is stateless wrt previous batches, we should be able to just sample uniformly
        randomly from the entire dataset.
    """
    n_token: int = tokens.shape[-1]

    input_result = torch.zeros(batch_size, context_length, device=device, dtype=torch.int)
    output_result = torch.zeros(batch_size, context_length, device=device, dtype=torch.int)

    batch_starts = (random.randint(0, n_token - context_length - 1) for _ in range(batch_size))
    for i, batch_start in enumerate(batch_starts):
        input_result[i, : ] = torch.Tensor(tokens[batch_start : batch_start+context_length])
        output_result[i, : ] = torch.Tensor(tokens[batch_start + 1 : batch_start + 1 + context_length])

    return input_result, output_result

