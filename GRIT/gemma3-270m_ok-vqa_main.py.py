# idefics 
# HuggingFaceM4/idefics-9b - https://huggingface.co/docs/transformers/en/model_doc/idefics
# HuggingFaceM4/idefics2-8b - https://huggingface.co/docs/transformers/en/model_doc/idefics2
# HuggingFaceM4/Idefics3-8B-Llama3 - https://huggingface.co/docs/transformers/en/model_doc/idefics3

# PaliGemma
# google/paligemma2-3b-mix-224 - https://huggingface.co/docs/transformers/en/model_doc/paligemma
# google/paligemma2-28b-mix-224 - https://huggingface.co/docs/transformers/en/model_doc/paligemma

# SmolVLM
# HuggingFaceTB/SmolVLM-Base - https://huggingface.co/HuggingFaceTB/SmolVLM-Base
# HuggingFaceTB/SmolVLM-Instruct - https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct

# Llama
# meta-llama/Llama-3.2-11B-Vision - https://huggingface.co/meta-llama/Llama-3.2-11B-Vision


from grit_from_scratch import load_model, apply_grit_to_model, UniversalVLMGritCollator, get_grit_parameters, grit_training_step, DEVICE, evaluate, OKQACollator

from utils import count_parameters, freeze_non_grit_parameters
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

print("[LOG] Starting GRIT Fine-tuning on OK-VQA")


grit_config = {
    "rank": 4,
    "alpha": 16,
    "k": 2,
}
train_config = {
    "batch_size": 4,
    "learning_rate": 2e-4,
    "num_epochs": 50,
    "max_samples": 200,
    "gradient_accumulation_steps": 5,
    "eval_steps": 20,
    "warmup_steps": 50,
}

print("\n[LOG] Loading model and tokenizer...")

model_name = "google/gemma-3-270m"
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float32, 
    attn_implementation="eager"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.to(DEVICE)

model = apply_grit_to_model(model, target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "dense", "linear", "proj", "projection", "down_proj"], **grit_config)
freeze_non_grit_parameters(model)

total_params, trainable_params = count_parameters(model)
print(f"[LOG] Total params: {total_params:,}, Trainable: {trainable_params:,}")

grit_params = get_grit_parameters(model)
if not grit_params:
    print("[ERROR] No GRIT params found!")

dataset = load_dataset("lmms-lab/OK-VQA")
train_data = dataset["val2014"].select(range(0, train_config["max_samples"] * 3// 4))
val_data = dataset["val2014"].select(range(train_config["max_samples"] * 3 //4, train_config["max_samples"]))

collator = OKQACollator(tokenizer)

train_loader = DataLoader(train_data, batch_size=train_config["batch_size"], shuffle=True, collate_fn=collator)
val_loader = DataLoader(val_data, batch_size=train_config["batch_size"], shuffle=True, collate_fn=collator)

optimizer = torch.optim.AdamW(grit_params, lr=train_config["learning_rate"])
total_steps = len(train_loader) * train_config["num_epochs"]
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)


print("[LOG] Training...")
model.train()
global_step = 0
best_val_loss = float('inf')

for epoch in range(train_config["num_epochs"]):
    print(f"[LOG] Epoch {epoch+1}/{train_config['num_epochs']}")
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in progress:
        batch = {k: v.to(DEVICE) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        loss = grit_training_step(model, batch, optimizer, train_config["gradient_accumulation_steps"], global_step)
        progress.set_postfix({"loss": loss["total_loss"]})

        if (global_step + 1) % train_config["gradient_accumulation_steps"] == 0:
            scheduler.step()

        global_step += 1

        if global_step % train_config["eval_steps"] == 0:
            val_loss = evaluate(model, val_loader, DEVICE)
            print(f"[LOG] Step {global_step} - Val loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            model.train()

print("[LOG] Training complete!")

print(f"[LOG] Best validation loss: {best_val_loss:.4f}")