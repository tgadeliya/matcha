import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from torch import Tensor, nn

from matcha.data.dataloader import data_loading
from matcha.optimizers import AdamW
from matcha.models.decoders import TransformerLM
from matcha.components.losses import cross_entropy_loss
from matcha.components.metrics import perplexity
from matcha.utils import load_checkpoint, save_checkpoint

load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))

@dataclass
class TrainerConfig:
    # Model params
    # training params
    # checkpointing params
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
    # optimizer
    lr: float
    weight_decay: float
    betas: tuple[float, float]
    eps: float
    checkpoint_dir: str
    train_data_path: str
    val_data_path: str

    @classmethod
    def load_from_json(cls, path) -> "TrainerConfig":
        trainer_params = json.loads(path)
        return cls(**trainer_params)


class Trainer:
    def __init__(self, cfg: TrainerConfig) -> None:
        self.cfg = cfg
        # self._setup(cfg)
        # self._setup_dataloaders(cfg)
        self.device: str = self.cfg.device
        self.train_dataloder
        self.validation_dataloader
        self.model: nn.Module
        self.optimizer: AdamW
        self.iteration: int = 0

    def _setup_dataloaders(self) -> None:
        self.train_dataloder = ...
        self.validation_dataloader = ...

    def _setup(self) -> None:
        self.model = TransformerLM(**asdict(self.cfg))
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            betas=self.cfg.betas,
            eps=self.cfg.eps,
        )

    def get_train_batch(self) -> dict[str, Tensor]:
        x = np.memmap(self.cfg.train_data_path, dtype=np.uint16, mode="r")
        batch, labels = data_loading(
            x, self.cfg.batch_size, self.cfg.context_length, self.device
        )

        return {"input": batch, "labels": labels}

    def get_val_batch(self) -> dict[str, Tensor]:
        batch, labels = data_loading(
            x, self.cfg.batch_size, self.cfg.context_length, self.device
        )
        return {"input": batch, "labels": labels}

    def train(self):
        with wandb.init(project="matcha", config=asdict(self.cfg)) as run:
            self._setup()
            self._setup_dataloaders()

            for i in range(self.cfg.max_steps):
                loss = self.training_step()
                run.log({"loss": loss.item(), "step": i})
                if i % self.cfg.val_every_steps == 0:
                    val_perplexity = self.validate()
                    run.log({"val_perplexity": val_perplexity, "step": i})
                    self.model.train()

    def training_step(self):
        batch = self.get_train_batch()
        self.model.train()
        out = self.model(batch["input"])
        loss = cross_entropy_loss(out, batch["labels"])

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

    @torch.inference_mode()
    def validate(self) -> float:
        self.model.eval()
        val_perplexities = []
        for _ in range(self.cfg.val_steps):
            val_perp = self.validation_step()
            val_perplexities.append(val_perp)
        
        self.save_checkpoint(self.iteration)
        return sum(val_perplexities) / len(val_perplexities)

    def validation_step(self):
        batch = self.get_val_batch()
        return perplexity(batch["input"], batch["labels"])

    def save_checkpoint(self, step: int) -> None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        ckpt_path: Path = Path(
            f"{self.cfg.checkpoint_dir}", f"ckpt_{step=:07d}_{timestamp}.pt"
        )
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            iteration=step,
            out=ckpt_path,
        )
        with open(
            Path(self.cfg.checkpoint_dir, "trainer_config.json"), "w"
        ) as f:
            json.dump(asdict(self.cfg), f)

    @classmethod
    def load_from_checkpoint(cls, ckpt_file_path: Path) -> "Trainer":
        ckpt_dir = ckpt_file_path.parent
        cfg: TrainerConfig = TrainerConfig.load_from_json(
            ckpt_dir / "trainer_config.json"
        )
        trainer = cls(cfg)
        trainer._setup()
        load_checkpoint(
            src=ckpt_file_path,
            model=trainer.model,
            optimizer=trainer.optimizer,
        )
        return trainer
