import os, sys, json, torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# ensure project root is on PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train.dataset import ModelDataset
from config.config import autoreg_tiny_config, TrainingConfig
from model.autoreg import AutoregressiveModel

# ─── 1) Load true vocab size from token_to_id.json ────────────────────────────
DATA_DIR = "../data/wikitext-103"
token_to_id = json.load(open(os.path.join(DATA_DIR, "bpe_tokenizer", "token_to_id.json"), "r"))
true_vocab = len(token_to_id)
print(f">>> Using vocab_size = {true_vocab}")
autoreg_tiny_config.vocab_size = true_vocab

# ─── 2) Load token streams ─────────────────────────────────────────────────────
train_ids = torch.load(os.path.join(DATA_DIR, "train.pt"))[:10_000_000]
val_ids   = torch.load(os.path.join(DATA_DIR, "val.pt"))

# sanity check
assert train_ids.max().item() < true_vocab, f"Found ID {train_ids.max().item()} ≥ vocab_size {true_vocab}"

# ─── 3) DataLoaders ────────────────────────────────────────────────────────────
train_ds = ModelDataset(train_ids, autoreg_tiny_config.block_size)
val_ds   = ModelDataset(val_ids,   autoreg_tiny_config.block_size)

train_loader = DataLoader(train_ds,
                          batch_size=TrainingConfig().batch_size,
                          shuffle=True)
val_loader   = DataLoader(val_ds,
                          batch_size=TrainingConfig().batch_size,
                          shuffle=False)

# ─── 4) Model setup ─────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = AutoregressiveModel(autoreg_tiny_config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=TrainingConfig().learning_rate)
loss_fn   = nn.CrossEntropyLoss()
scaler    = GradScaler()

# ─── 5) Train / Validate with tqdm ─────────────────────────────────────────────
for epoch in range(1, TrainingConfig().num_epochs + 1):
    # — training —
    model.train()
    train_loss = 0.0
    train_bar = tqdm(train_loader,
                     desc=f"Epoch {epoch}/{TrainingConfig().num_epochs} [train]",
                     unit="batch")
    for batch_idx, (xb, yb) in enumerate(train_bar, start=1):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(dtype=torch.float16):
            logits = model(xb)
            loss   = loss_fn(logits.view(-1, logits.size(-1)), yb.view(-1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        avg_loss = train_loss / batch_idx
        train_bar.set_postfix(train_loss=f"{avg_loss:.4f}")

    # — validation —
    model.eval()
    val_loss = 0.0
    val_bar = tqdm(val_loader,
                   desc=f"Epoch {epoch}/{TrainingConfig().num_epochs} [val]  ",
                   unit="batch")
    for batch_idx, (xb, yb) in enumerate(val_bar, start=1):
        xb, yb = xb.to(device), yb.to(device)
        with autocast(dtype=torch.float16):
            logits = model(xb)
            loss   = loss_fn(logits.view(-1, logits.size(-1)), yb.view(-1))
        val_loss += loss.item()
        avg_vloss = val_loss / batch_idx
        val_bar.set_postfix(val_loss=f"{avg_vloss:.4f}")

torch.save(model.state_dict(), "autoreg_tiny.pt")
