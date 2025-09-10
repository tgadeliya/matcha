import os
from typing import IO, BinaryIO

import torch
from torch import nn, optim


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, iteration: int, out: str| os.PathLike | BinaryIO | IO[bytes] ) -> None:
    global_state_dict ={
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(global_state_dict, out)


def load_checkpoint(src: str| os.PathLike | BinaryIO | IO[bytes], model: nn.Module, optimizer: optim.Optimizer) -> None:
    global_state_dict = torch.load(src)
    model.load_state_dict(global_state_dict["model"])
    optimizer.load_state_dict(global_state_dict["optimizer"])
    return global_state_dict["iteration"]