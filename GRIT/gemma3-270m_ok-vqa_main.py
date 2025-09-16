import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from grit_tools import apply_grit_to_model, get_grit_parameters, save_grit_weights, evaluate, reproject_grit_modules
from utils import count_parameters, freeze_non_grit_parameters


from torchvision import transforms

class OKVQACollator:
    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        # Build full "Q → A" text
        texts = [f"Question: {ex['question']}\nAnswer: {ex['answers'][0] if isinstance(ex['answers'], list) else ex['answer']}" 
                 for ex in batch]

        # Tokenize once
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # For causal LM, labels = input_ids (model predicts next token)
        labels = encodings["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        encodings["labels"] = labels
        return encodings


def main():
    print("🚀 Starting GRIT Fine-tuning on OK-VQA")

    # Config
    model_name = "google/gemma-3-270m"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    grit_config = {
        "r": 8,
        "alpha": 16,
        "k": 4,
        "dropout_p": 0.1,
        "ema": 0.95,
    }
    train_config = {
        "batch_size": 4,
        "learning_rate": 2e-4,
        "num_epochs": 20,
        "max_samples": 200,  # small subset for testing
        "gradient_accumulation_steps": 2,
        "eval_steps": 20,
    }

    print("\nLoading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    print("\nApplying GRIT...")
    model = apply_grit_to_model(model, **grit_config)
    freeze_non_grit_parameters(model)

    total_params, trainable_params = count_parameters(model)
    print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

    grit_params = get_grit_parameters(model)
    if not grit_params:
        print("❌ No GRIT params found!")
        return

    print("\nLoading OK-VQA dataset...")
    dataset = load_dataset("lmms-lab/OK-VQA")
    # train_data = dataset["train"].select(range(train_config["max_samples"]))
    val_data = dataset["val2014"].select(range(train_config["max_samples"] // 4))

    collator = OKVQACollator(tokenizer)

    # train_loader = DataLoader(
    #     train_data, batch_size=train_config["batch_size"], shuffle=True, collate_fn=collator
    # )
    train_loader = DataLoader(
        val_data, batch_size=train_config["batch_size"], shuffle=False, collate_fn=collator
    )

    print("✅ Dataset ready")

    optimizer = torch.optim.AdamW(grit_params, lr=train_config["learning_rate"])
    total_steps = len(train_loader) * train_config["num_epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    print("\n🚦 Training...")
    model.train()
    global_step = 0

    for epoch in range(train_config["num_epochs"]):
        print(f"\nEpoch {epoch+1}/{train_config['num_epochs']}")
        progress = tqdm(train_loader)

        for batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / train_config["gradient_accumulation_steps"]
            loss.backward()

            if (global_step + 1) % train_config["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(grit_params, 1.0)
                optimizer.step()
                scheduler.step()
                reproject_grit_modules(model)
                optimizer.zero_grad()

            progress.set_postfix({"loss": loss.item() * train_config["gradient_accumulation_steps"]})
            global_step += 1

            if global_step % train_config["eval_steps"] == 0:
                val_loss = evaluate(model, train_loader, device)
                print(f"Step {global_step} - Val loss: {val_loss:.4f}")
                model.train()

    print("\n🎉 Training complete!")
    save_grit_weights(model, "okvqa_grit.pt")


if __name__ == "__main__":
    main()
