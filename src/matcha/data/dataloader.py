import numpy as np
from typing import Literal
from torch import Tensor, LongTensor
import torch

DEVICE = Literal["cpu", "cuda:0"]


def data_loading(
    x: np.ndarray, batch_size: int, context_length: int, device: DEVICE
) -> tuple[Tensor, Tensor]:
    sampled_start_tokens = torch.randint(
        low=0, high=len(x) - context_length, size=(batch_size,)
    )

    input_seq, labels = [], []
    for start_idx in sampled_start_tokens:
        input_seq.append(LongTensor(x[start_idx : start_idx + context_length]))
        labels.append(
            LongTensor(x[start_idx + 1 : start_idx + context_length + 1])
        )

    input_seq = torch.stack(input_seq, dim=0).to(device)
    labels = torch.stack(labels, dim=0).to(device)
    return input_seq, labels


