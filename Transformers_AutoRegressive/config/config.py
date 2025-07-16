from typing import Optional
from dataclasses import dataclass
import torch

# ---------------------
# Data Configuration
# ---------------------
@dataclass
class DataConfig:
    name: str
    vocab_size: int 
    num_train_tokens: int
    block_size: int
    train_path: str
    test_path: str
    val_path: Optional[str]
    pad_id: int = 0
    unk_id: int = 1
    bos_id: int = 2
    eos_id: int = 3

# ---------------------
# Model Configuration
# ---------------------
@dataclass
class AutoregModelConfig:
    vocab_size: int = 2000
    block_size: int = 128
    n_layer: int = 4
    n_heads: int = 4
    n_embeds: int = 256
    dropout: float = 0.3


# ---------------------
# Training Configuration
# ---------------------
@dataclass
class TrainingConfig:
    batch_size: int = 512
    learning_rate: float = 3e-4
    max_iters: int = 5000
    eval_interval: int = 500
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "./checkpoints"
    num_epochs: int = 2
    

## Define some models
## --------------------
autoreg_tiny_config = AutoregModelConfig(n_layer=4, n_heads=4, n_embeds=256)
autoreg_small_config = AutoregModelConfig(n_layer=6, n_heads=6, n_embeds=384)
autoreg_base_config = AutoregModelConfig(n_layer=8, n_heads=8, n_embeds=512)
## --------------------

## Define some datasets
## --------------------
wikihow_103_config = DataConfig(name = "wikihow_103", 
                         vocab_size= 2000, 
                         num_train_tokens=131_361_015,
                         block_size=128,
                         train_path="../data/wikitext-103/train.pt",
                         test_path="../data/wikitext-103/test.pt",
                         val_path="../data/wikitext-103/val.pt",
                         )
## --------------------