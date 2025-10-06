import argparse
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Union, get_args, get_origin

from matcha.configs import TokenizerTrainConfig, TrainerConfig
from matcha.tokenizers.bpe import BPETrainer
from matcha.trainer import Trainer

TASKS = {
    "train": (TrainerConfig, Trainer),
    "train_tokenizer": (TokenizerTrainConfig, BPETrainer),
}


def train_tokenizer(cfg: TokenizerTrainConfig):
    bpe_trainer = BPETrainer(
        corpus_path=Path("data/TinyStoriesV2-GPT4-valid.txt"),
        vocab_size=10_000,
        special_tokens=["<|endoftext|>"],
    )

    bpe_trainer.train()
    bpe_trainer.save_tokenizer()


def train_language_model():
    cfg = TrainerConfig(
        vocab_size=20000,
        max_steps=10,
        val_steps=5,
        val_every_steps=5,
        batch_size=16,
        context_length=512,
        d_model=256,
        device="cuda:0",
        num_layers=3,
        num_heads=8,
        d_ff=512,
        theta=10000,
        lr=1e-4,
        weight_decay=1e-1,
        betas=(0.9, 0.95),
        eps=1e-8,
        checkpoint_dir="data/checkpoints",
        train_data_path="/Users/tsimur.hadeliya/code/language_modeling/data/corpus_tokenized/tiny_stories_validation.npy",
        val_data_path="/Users/tsimur.hadeliya/code/language_modeling/data/corpus_tokenized/tiny_stories_validation.npy",
    )

    trainer = Trainer(cfg)
    trainer.train()


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
    config_cls, task_runner = TASKS[args.task][0]
    config = config_cls.from_file(Path(args.config_path))
    task_runner(config).run()


if __name__ == "__main__":
    main()
