from math import sqrt

import einops
import torch
import torch.nn as nn
from torch import Tensor


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.W = torch.nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        std_val = sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(
            self.W, mean=0, std=std_val, a=-3 * std_val, b=3 * std_val
        )

    def forward(self, x: Tensor) -> Tensor:
        return einops.einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")
