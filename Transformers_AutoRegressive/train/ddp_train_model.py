import os
import sys
import json
import argparse
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

# make sure project root is on PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train.dataset import ModelDataset
from config.config import autoreg_tiny_config, TrainingConfig
from model.autoreg import AutoregressiveModel

def train_one_epoch(model, loader, optimizer, scaler, device, rank, epoch):
    model.train()
    loader.sampler.set_epoch(epoch)
    total_loss = 0.0
    for batch in tqdm(loader, desc=f"[Rank {rank}] Training", disable=(rank != 0)):
        x, y = batch
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        with autocast():
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, device, rank):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"[Rank {rank}] Validation", disable=(rank != 0)):
            x, y = batch
            x, y = x.to(device), y.to(device)
            with autocast():
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
    return total_loss / len(loader)

def main(local_rank):
    # --- 0) Read env vars ---
    rank       = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]

    print(f"[{rank}] MASTER={master_addr}:{master_port} WORLD={world_size}", flush=True)

    # --- 1) NCCL init ---
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    print(f"[{rank}] Process group initialized with NCCL", flush=True)

    # --- 2) Load data & config ---
    DATA_DIR = "../data/wikitext-103"
    token_to_id = json.load(open(os.path.join(DATA_DIR, "bpe_tokenizer", "token_to_id.json")))
    true_vocab = len(token_to_id)
    autoreg_tiny_config.vocab_size = true_vocab
    print(f"[{rank}] Using vocab_size={true_vocab}", flush=True)

    train_ids = torch.load(os.path.join(DATA_DIR, "train.pt"))[:10_000_000]
    val_ids   = torch.load(os.path.join(DATA_DIR, "val.pt"))
    print(f"[{rank}] PT loaded", flush=True)

    train_ds = ModelDataset(train_ids, autoreg_tiny_config.block_size)
    val_ds   = ModelDataset(val_ids,   autoreg_tiny_config.block_size)

    # --- 3) DDP samplers & loaders ---
    train_sampler = torch.utils.data.DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    val_sampler   = torch.utils.data.DistributedSampler(
        val_ds,   num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=TrainingConfig().batch_size,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=TrainingConfig().batch_size,
        sampler=val_sampler,
    )
    print(f"[{rank}] Loaders ready", flush=True)

    # --- 4) Model / optimizer / scaler ---
    device = torch.device("cuda", local_rank)
    model = AutoregressiveModel(autoreg_tiny_config).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = optim.AdamW(model.parameters(), lr=TrainingConfig().learning_rate)
    scaler    = GradScaler()
    print(f"[{rank}] Model & optimizer set up", flush=True)

    # --- 5) Training loop ---
    for epoch in range(3):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, rank, epoch)
        val_loss   = validate(model, val_loader, device, rank)
        if rank == 0:
            print(f"Epoch {epoch+1}: Train={train_loss:.4f} Val={val_loss:.4f}", flush=True)
            torch.save(model.module.state_dict(), f"gpt_autoreg_epoch{epoch+1}.pt")

    dist.destroy_process_group()
    print(f"[{rank}] Finished", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=int(os.environ.get('LOCAL_RANK', 0)))
    args = parser.parse_args()
    main(args.local_rank)
