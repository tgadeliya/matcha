from collections import OrderedDict

import torch.nn as nn
from torch import Tensor

from matcha.components.embeddings import Embedding
from matcha.components.blocks import TransformerBlock
from matcha.components.normalization import RMSNorm
from matcha.components.linear import Linear


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
    ) -> None:
        super().__init__()  # type: ignore
        self.emb = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[str(i)] = TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                theta=theta,
                max_seq_len=context_length,
            )
        self.layers = nn.Sequential(layers)
        self.ln_final = RMSNorm(d_model=d_model)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.emb(x)
        x = self.layers(x)
        x = self.ln_final(x)
        return self.lm_head(x)
