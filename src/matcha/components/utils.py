import torch
from torch import Tensor


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    xs = torch.exp(x - x.max())
    xs /= xs.sum(dim=-1, keepdim=True)
    return xs
