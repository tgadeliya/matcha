import argparse
from collections.abc import Callable
from pathlib import Path

from matcha.configs import TokenizerTrainConfig, TrainerConfig
from matcha.tokenizers import BPETrainer
from matcha.trainer import Trainer


def train_tokenizer(cfg: TokenizerTrainConfig):
    bpe_trainer = BPETrainer(cfg)
    bpe_trainer.train()
    bpe_trainer.save_tokenizer()


def train_language_model(cfg: TrainerConfig):
    trainer = Trainer(cfg)
    trainer.train()


TASKS: dict[str, tuple[type, Callable]] = {
    "train": (TrainerConfig, train_language_model),
    "train_tokenizer": (TokenizerTrainConfig, train_tokenizer),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="matcha-run",
        description="Run Matcha workflows from a TOML config.",
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=tuple(TASKS.keys()),
        help="What task to run.",
    )
    parser.add_argument(
        "--config_path",
        required=True,
        help="Path to the TOML config file.",
    )
    args = parser.parse_args(argv)
    config_cls, task_runner = TASKS[args.task]
    cfg = config_cls.from_file(Path(args.config_path))
    task_runner(cfg)


if __name__ == "__main__":
    main()
