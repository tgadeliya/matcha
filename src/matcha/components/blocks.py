import torch.nn as nn
from torch import Tensor

from matcha.components.attention import MultiHeadAttention
from matcha.components.embeddings import RotaryPositionalEmbedding
from matcha.components.feedforward import SwiGLU
from matcha.components.normalization import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.d_k = d_model // num_heads
        pos_emb_func = RotaryPositionalEmbedding(
            theta=theta, max_seq_len=max_seq_len, d_k=self.d_k
        )
        self.attn = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads, pos_emb_func=pos_emb_func
        )
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)
        self.ln1 = RMSNorm(d_model=d_model)
        self.ln2 = RMSNorm(d_model=d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
