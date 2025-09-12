import os
from typing import IO, BinaryIO

import numpy as np
import torch
from torch import Tensor, nn, optim

from matcha.components.utils import softmax
from matcha.tokenizers.bpe import BPETokenizer


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    global_state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(global_state_dict, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: optim.Optimizer,
) -> None:
    global_state_dict = torch.load(src)
    model.load_state_dict(global_state_dict["model"])
    optimizer.load_state_dict(global_state_dict["optimizer"])
    return global_state_dict["iteration"]


@torch.inference_mode()
def generate(
    model,
    tokenizer: BPETokenizer,
    prompt: str,
    max_tokens: int = 20,
    temperature: float = 1.0,
    top_p: float = 0.0,
    random_seed: int = 25,
) -> list[str]:
    def top_p_sampling(probs: Tensor, top_p: float):
        probs_sorted = sorted(
            [(i, p) for i, p in enumerate(probs.tolist())], key=lambda x: x[1]
        )
        probs_sorted_filtered = [p for p in probs_sorted if p[1] >= top_p]
        rng = np.random.default_rng(seed=random_seed)
        sampled = rng.choice(probs_sorted_filtered, 3)
        return sampled[0]

    model.eval()
    enc_prompt = tokenizer.encode(prompt)
    for i in range(max_tokens):
        logits = model(enc_prompt)
        prob = softmax(logits[:, -1, :])
        prob = prob / temperature
        if top_p > 0.0:
            pred = top_p_sampling(prob, top_p)
        else:
            pred = torch.max(prob, dim=-1).indices.tolist()

        enc_prompt = [enc_prompt[i] + [pred[i]] for i in range(len(enc_prompt))]

    return [tokenizer.decode(prompt) for prompt in enc_prompt]
