from pathlib import Path

from matcha.tokenizers.bpe_trainer import BPETrainer
from matcha.trainer import Trainer, TrainerConfig


def train_tokenizer():
    bpe_trainer = BPETrainer(
        corpus_path=Path("data/TinyStoriesV2-GPT4-valid.txt"),
        vocab_size=10_000,
        special_tokens=["<|endoftext|>"],
    )

    bpe_trainer.train(multiprocessing=True, num_processes=4)
    bpe_trainer.save_tokenizer(
        dir_path=Path("data/tokenizer/tiny_stories_valid"),
    )


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
