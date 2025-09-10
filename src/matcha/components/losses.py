import torch
from torch import Tensor

def cross_entropy_loss(x: Tensor, target: Tensor) -> Tensor:
    xs = x - x.max(dim=-1, keepdim=True).values
    logsoftmax = xs - xs.exp().sum(dim=-1, keepdim=True).log()
    return (-1 * logsoftmax[torch.arange(xs.size()[0]), target]).mean()
