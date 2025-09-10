from matcha.trainer import Trainer, TrainerConfig
from matcha.tokenizers.bpe import BPETokenizer
from matcha.tokenizers.trainer import train_bpe, save_vocab_and_merges




def run_tokenizer():
    input_path = "/Users/tsimur.hadeliya/code/_other/CS336/assignment1-basics/data/owt_valid.txt"
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=32_000,
        special_tokens=["<|endoftext|>"],
        multiprocessing=True,
    )

    save_vocab_and_merges(
        vocab=vocab,
        merges=merges,
        vocab_path="vocab_owt_valid.json",
        merges_path="merges_owt_valid.json",
    )



def run():
    cfg = TrainerConfig(
        vocab_size=10000,
        max_steps=10000,
        val_steps=100,
        val_every_steps=1000,
        batch_size=128,
        context_length=1024,
        d_model=512,
        device="cuda:0",
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        lr=6e-4,
        weight_decay=1e-1,
        betas=(0.9, 0.95),
        eps=1e-8,
        checkpoint_dir="checkpoints",
        train_data_path="data/train.bin",
        val_data_path="data/val.bin",
    )   

    trainer = Trainer(cfg)
    trainer.train()




if __name__ == "__main__":
    run_tokenizer()
    run()