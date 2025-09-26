from torch import Tensor

def cross_entropy_loss(x: Tensor, target: Tensor) -> Tensor:
    xs = x - x.max(dim=-1, keepdim=True).values
    logsoftmax = xs - xs.exp().sum(dim=-1, keepdim=True).log()
    log_probs = logsoftmax.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    return (-log_probs).mean()
