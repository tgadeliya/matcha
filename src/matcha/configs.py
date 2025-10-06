import tomllib
from dataclasses import dataclass
from pathlib import Path


class BasicConfig:
    @classmethod
    def load_from_file(cls, path: Path) -> "BasicConfig":
        with path.open("rb") as f:
            params = tomllib.load(f)
        return cls(**params)


@dataclass
class TokenizerTrainConfig(BasicConfig):
    corpus_path: str
    vocab_size: int
    special_tokens: list[str]
    save_dir_path: str
    num_processes: int
    multiprocessing: bool = False


@dataclass
class TrainerConfig(BasicConfig):
    # Model params
    vocab_size: int
    max_steps: int
    val_steps: int
    val_every_steps: int
    batch_size: int
    context_length: int
    d_model: int
    device: str
    num_layers: int
    num_heads: int
    d_ff: int
    theta: int
    # optimizer
    lr: float
    weight_decay: float
    betas: tuple[float, float]
    eps: float
    # checkpointing params
    checkpoint_dir: str
    # training params
    train_data_path: str
    val_data_path: str
