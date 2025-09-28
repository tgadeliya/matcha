from torch import Tensor

from .losses import cross_entropy_loss


def perplexity(x: Tensor, target: Tensor) -> float:
    return cross_entropy_loss(x, target).exp().item()
