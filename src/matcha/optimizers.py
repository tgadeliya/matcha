from collections.abc import Callable
from math import cos, pi

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "wd": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, defaults)
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay

    def step(self, closure: Callable | None = None) -> None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t, mu, v = state.get("t", 1), state.get("mu", 0.0), state.get("v", 0.0)

                g: torch.Tensor = p.grad
                mu = self.b1 * mu + (1 - self.b1) * g
                v = self.b2 * v + (1 - self.b2) * g.pow(2)

                lr_t = lr * (1 - self.b2**t) ** 0.5 / (1 - self.b1**t)

                p.data -= lr_t * mu / (v.pow(0.5) + self.eps)
                p.data -= self.wd * lr * p.data

                state["t"], state["mu"], state["v"] = t + 1, mu, v

        return loss


def learning_rate_schedule(t, lr_max, lr_min, T_w, T_c) -> float:
    if t < T_w:
        return t / T_w * lr_max
    elif T_w <= t <= T_c:
        cos_term = 1 + cos(((t - T_w) / (T_c - T_w)) * pi)
        return lr_min + 1 / 2 * cos_term * (lr_max - lr_min)
    else:
        return lr_min


def gradient_clipping(params, M) -> None:
    grads = []
    for p in params:
        if p.grad is not None:
            grads.append(p.grad)

    if not grads:
        return

    l2_norm = torch.norm(torch.Tensor([g.norm(p=2) for g in grads]), p=2)

    if l2_norm > M:
        clip_factor = M / (l2_norm + 1e-6)
        for p in params:
            if p.grad is not None:
                p.grad *= clip_factor
