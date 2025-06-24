from math import sqrt

import einops
import torch
import torch.nn as nn
from torch import Tensor


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        R = self._generate_rotary_matrix()
        self.register_buffer("R", R, persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        x_pos: Tensor = einops.einsum(
            x,
            self.R[token_positions, ...],
            "... seq_len d_k, seq_len d_kk d_k -> ... seq_len d_kk",
        )
        return x_pos

    def _generate_rotary_matrix(self) -> Tensor:
        rm = torch.ones(self.max_seq_len, self.d_k, self.d_k, dtype=torch.float32)
        for i in range(self.max_seq_len):
            rm[i, :, :] = self.gen_pos_i(i)
        return rm

    def gen_pos_i(self, i: int) -> Tensor:
        pows = (2 * torch.arange(self.d_k // 2, dtype=torch.float32)) / self.d_k
        thetai: Tensor = i / (self.theta**pows)
        ts = torch.sin(thetai)
        tc = torch.cos(thetai)
        rot_matrices = torch.empty(
            (self.d_k // 2, 2, 2), device=self.device, dtype=torch.float32
        )
        rot_matrices[:, 0, 0] = tc
        rot_matrices[:, 0, 1] = -ts
        rot_matrices[:, 1, 0] = ts
        rot_matrices[:, 1, 1] = tc
        bm: Tensor = torch.block_diag(*rot_matrices)
        return bm
