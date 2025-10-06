import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

import tomli_w

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()

@dataclass
class BasicConfig:
    @classmethod
    def load_from_file(cls, path: Path) -> Self:
        with path.open("rb") as f:
            params = tomllib.load(f)

        for param, val in params.items():
            if param.endswith("_path"):
                params[param] = str(ROOT_DIR / Path(val).resolve())
        
        return cls(**params)

    def write_to_file(self, path: Path) -> None:
        with path.open("wb") as f:
            toml_string = tomli_w.dumps(asdict(self))
            f.write(toml_string.encode("utf-8"))


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
    checkpoint_dir_path: str
    # training params
    train_data_path: str
    val_data_path: str
