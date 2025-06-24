from math import sqrt

import einops
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter

from .activations import SiLU


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_ff = d_ff  # int(((d_model * 8 / 3) + 63) // 64) * 64
        self.d_model = d_model
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        self.W1 = torch.nn.Parameter(
            torch.empty(self.d_ff, self.d_model, device=device, dtype=dtype)
        ).to(self.device)
        self.W2 = torch.nn.Parameter(
            torch.empty(self.d_model, self.d_ff, device=device, dtype=dtype)
        ).to(self.device)
        self.W3 = torch.nn.Parameter(
            torch.empty(self.d_ff, self.d_model, device=device, dtype=dtype)
        ).to(self.device)
        self.act_func = SiLU()

        std_val = sqrt(2 / (self.d_model + self.d_ff))
        nn.init.trunc_normal_(
            self.W1, mean=0, std=std_val, a=-3 * std_val, b=3 * std_val
        )
        nn.init.trunc_normal_(
            self.W2, mean=0, std=std_val, a=-3 * std_val, b=3 * std_val
        )
        nn.init.trunc_normal_(
            self.W3, mean=0, std=std_val, a=-3 * std_val, b=3 * std_val
        )

    def forward(self, x: Tensor) -> Tensor:
        out = einops.einsum(x, self.W1, "... d_model, d_ff d_model -> ... d_ff")
        out1 = self.act_func(out)
        out2 = einops.einsum(x, self.W3, "... d_model, d_ff d_model -> ... d_ff")
        out = out1 * out2
        out = einops.einsum(out, self.W2, "... d_ff, d_model d_ff -> ... d_model")
        return out
