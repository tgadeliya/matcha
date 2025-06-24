from dataclasses import dataclass, asdict

import pytest
import torch

from matcha.models import TransformerLM



@dataclass
class TransformerArgs:
    vocab_size: int = 1024
    context_length: int = 512
    num_layers: int = 8
    d_model: int = 124
    num_heads: int = 12
    d_ff: int = 1024
    theta: float = 1000


@pytest.fixture(scope="module")
def transformer_args() -> dict[str, float]:
    return asdict(TransformerArgs())


class TestTransformerModel:
    def test_init(self, transformer_args):
        TransformerLM(**transformer_args)

    @pytest.mark.parametrize("bs,seq_len", [(1, 256), (2, 17), (4, 1)])
    def test_dry_run(self, bs, seq_len, transformer_args):
        vocab_size = transformer_args["vocab_size"]
        x = torch.randint(0, vocab_size, (bs, seq_len))
        model = TransformerLM(**transformer_args)
        out = model(x)
        
        out_bs, out_seq_len, out_vocab_size = out.size()
        
        assert out_bs == bs
        assert out_seq_len == seq_len
        assert out_vocab_size == vocab_size
