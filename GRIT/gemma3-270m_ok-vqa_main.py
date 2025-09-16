import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from torch import nn

from grit_tools import apply_grit_to_model, get_grit_parameters, save_grit_weights, grit_natural_gradient_step, grit_neural_reprojection, GritLinear

from utils import count_parameters, freeze_non_grit_parameters


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


def grit_training_step(model, batch, optimizer, accumulation_steps=1, step_count=0):
    """
    Complete GRIT training step following the pipeline:
    1. Forward pass with LoRA
    2. Compute loss and gradients  
    3. Update K-FAC statistics
    4. Natural gradient step with K-FAC preconditioning
    5. Neural reprojection
    """
    # Forward pass
    outputs = model(**batch)
    loss = outputs.loss / accumulation_steps
    
    # Backward pass - this will trigger gradient computation and K-FAC updates
    loss.backward()
    
    # Only apply GRIT updates on actual optimization steps
    should_step = (step_count + 1) % accumulation_steps == 0
    
    if should_step:
        for module in model.modules():
            if isinstance(module, GritLinear) and module.enable_grit and module.training:
                module.compute_subspace_bases()
        
        # Apply gradient clipping before natural gradient step
        grit_params = get_grit_parameters(model)
        torch.nn.utils.clip_grad_norm_(grit_params, 1.0)
        
        # Apply natural gradient step instead of regular optimizer.step()
        grit_natural_gradient_step(model, optimizer)
        
        # Apply neural reprojection: θ_new = U_k U_k^T θ_updated
        grit_neural_reprojection(model)
        
        # Clear gradients
        optimizer.zero_grad()
    
    return loss.item() * accumulation_steps


def update_kfac_statistics_hook(module, grad_input, grad_output):
    """Hook to update K-FAC statistics during backward pass"""
    if isinstance(module, GritLinear) and module.enable_grit and module.training:
        if grad_output[0] is not None:
            module.update_kfac_statistics(grad_output[0])

def evaluate(model, val_loader, device):
    """Evaluate the model on validation set"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            try:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = model(**batch)
                total_loss += outputs.loss.item()
                num_batches += 1
                
            except Exception as e:
                print(f"Evaluation error: {e}")
                continue
    
    return total_loss / max(1, num_batches)


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
        "kfac_damping": 1e-5,
    }
    train_config = {
        "batch_size": 4,
        "learning_rate": 2e-4,
        "num_epochs": 20,
        "max_samples": 200,
        "gradient_accumulation_steps": 2,
        "eval_steps": 20,
        "warmup_steps": 50,
    }

    print("\nLoading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float32, 
        attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    print("\nApplying GRIT...")
    model = apply_grit_to_model(model, **grit_config)
    freeze_non_grit_parameters(model)

    # Register hooks for K-FAC statistics updates
    handles = []
    for module in model.modules():
        if isinstance(module, GritLinear) and module.enable_grit:
            handle = module.register_backward_hook(update_kfac_statistics_hook)
            handles.append(handle)

    total_params, trainable_params = count_parameters(model)
    print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

    grit_params = get_grit_parameters(model)
    if not grit_params:
        print("❌ No GRIT params found!")
        return

    print(f"GRIT parameters: {len(grit_params)} tensors")

    print("\nLoading OK-VQA dataset...")
    dataset = load_dataset("lmms-lab/OK-VQA")
    # train_data = dataset["train"].select(range(train_config["max_samples"]))
    train_data = dataset["val2014"].select(range(train_config["max_samples"] // 4))

    collator = OKVQACollator(tokenizer)

    train_loader = DataLoader(
        train_data, 
        batch_size=train_config["batch_size"], 
        shuffle=True, 
        collate_fn=collator
    )

    print("✅ Dataset ready")

    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(grit_params, lr=train_config["learning_rate"])
    total_steps = len(train_loader) * train_config["num_epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    print("\n🚦 Training with GRIT pipeline...")
    model.train()
    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(train_config["num_epochs"]):
        print(f"\nEpoch {epoch+1}/{train_config['num_epochs']}")
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        epoch_loss = 0.0
        num_batches = 0

        for batch in progress:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # GRIT training step (replaces manual loss.backward() + optimizer.step())
            loss = grit_training_step(
                model, 
                batch, 
                optimizer, 
                train_config["gradient_accumulation_steps"],
                global_step
            )
            
            epoch_loss += loss
            num_batches += 1

            # Step scheduler only on actual optimization steps
            if (global_step + 1) % train_config["gradient_accumulation_steps"] == 0:
                scheduler.step()

            # Update progress bar
            progress.set_postfix({
                "loss": f"{loss:.4f}", 
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                "step": global_step
            })
            
            global_step += 1

            # Evaluation
            if global_step % train_config["eval_steps"] == 0:
                print(f"\nEvaluating at step {global_step}...")
                val_loss = evaluate(model, train_loader, device)
                print(f"Step {global_step} - Val loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_grit_weights(model, f"okvqa_grit_best_step_{global_step}.pt")
                    print(f"💾 Saved best model (val_loss: {val_loss:.4f})")
                
                model.train()

        avg_epoch_loss = epoch_loss / max(1, num_batches)
        print(f"Epoch {epoch+1} complete - Avg loss: {avg_epoch_loss:.4f}")

        # End of epoch evaluation
        print(f"\nEnd of epoch {epoch+1} evaluation...")
        val_loss = evaluate(model, train_loader, device)
        print(f"Epoch {epoch+1} - Val loss: {val_loss:.4f}")
        model.train()

    print("\n🎉 Training complete!")
    
    # Final save
    save_grit_weights(model, "okvqa_grit_final.pt")
    print(f"💾 Final model saved")
    
    # Cleanup hooks
    for handle in handles:
        handle.remove()
    
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()