from math import sqrt

import einops
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter

from matcha.components.embeddings import RotaryPositionalEmbedding
from matcha.components.utils import softmax


def scaled_dot_product_attention(
    Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None
) -> Tensor:
    KQ = einops.einsum(
        K, Q, "... seq_lenk d , ... seq_lenq d -> ... seq_lenq seq_lenk"
    ) / (Q.size()[-1] ** 0.5)
    if mask is not None:
        KQ.masked_fill_(~mask, -torch.inf)
    KQ = softmax(KQ, dim=-1)
    return einops.einsum(
        KQ, V, "... seq_lenq seq_lenk , ... seq_lenk d_v -> ... seq_lenq d_v"
    )

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        pos_emb_func: RotaryPositionalEmbedding | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        d_k = d_v = d_model // num_heads
        self.q_proj = Parameter(torch.empty(num_heads * d_k, d_model))
        self.k_proj = Parameter(torch.empty(num_heads * d_k, d_model))
        self.v_proj = Parameter(torch.empty(num_heads * d_v, d_model))
        self.o_proj = Parameter(torch.empty(d_model, num_heads * d_v))
        self.pos_emb_func = pos_emb_func

        std_val = sqrt(2 / (self.d_model + num_heads * d_k))
        nn.init.trunc_normal_(
            self.q_proj, mean=0, std=std_val, a=-3 * std_val, b=3 * std_val
        )
        nn.init.trunc_normal_(
            self.k_proj, mean=0, std=std_val, a=-3 * std_val, b=3 * std_val
        )
        nn.init.trunc_normal_(
            self.v_proj, mean=0, std=std_val, a=-3 * std_val, b=3 * std_val
        )
        nn.init.trunc_normal_(
            self.o_proj, mean=0, std=std_val, a=-3 * std_val, b=3 * std_val
        )

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        q = einops.rearrange(
            einops.einsum(x, self.q_proj, "... d_in, d_out d_in  -> ... d_out"),
            "... seq_len (nh d_k) -> ... nh seq_len d_k",
            nh=self.num_heads,
        )

        k = einops.rearrange(
            einops.einsum(x, self.k_proj, "... d_in, d_out d_in  -> ... d_out"),
            "... seq_len (nh d_k) -> ... nh seq_len d_k",
            nh=self.num_heads,
        )

        v = einops.rearrange(
            einops.einsum(x, self.v_proj, "... d_in, d_out d_in  -> ... d_out"),
            "... seq_len (nh d_k) -> ... nh seq_len d_k",
            nh=self.num_heads,
        )

        seq_len_q, seq_len_k = q.size()[-2], k.size()[-2]
        if self.pos_emb_func:
            if token_positions is None:
                token_positions = torch.arange(seq_len_q).reshape(1, -1)
            q = self.pos_emb_func(q, token_positions=token_positions[0])
            k = self.pos_emb_func(k, token_positions=token_positions[0])

        mask = ~torch.triu(
            input=torch.ones(seq_len_q, seq_len_k, device=q.device), diagonal=1
        ).bool()

        attn = scaled_dot_product_attention(q, k, v, mask=mask)
        return einops.einsum(
            einops.rearrange(attn, "... nh seq_len d_k -> ... seq_len (nh d_k)"),
            self.o_proj,
            "bs sl d, d_model d  -> bs sl d_model",
        )
