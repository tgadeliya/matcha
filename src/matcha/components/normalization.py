
import einops
import torch
import torch.nn as nn
from torch import Tensor


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.device = device
        self.dtype = dtype

        self.g = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        x_dtype = x.dtype
        x = x.to(torch.float32)

        x_rms = (x.square().sum(dim=-1, keepdim=True) / self.d_model + self.eps).sqrt()
        result = einops.einsum(
            (x / x_rms), self.g, " ... d_model, d_model -> ... d_model"
        )
        return result.to(x_dtype)
