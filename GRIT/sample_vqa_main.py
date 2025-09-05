import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from tqdm import tqdm

from grit_tools import apply_grit_to_model, get_grit_parameters, save_grit_weights, evaluate
from utils import count_parameters, freeze_non_grit_parameters
from sample_vqa_loader import SimpleVQACollator, SimpleVQADataset


def main():
    print("🚀 Starting GRIT K-FAC Fine-tuning with Parameter Efficiency Fix")
    
    # Configuration
    model_name = "google/gemma-3-270m"  # Smaller model for testing
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # GRIT configuration
    grit_config = {
        'r': 8,            # LoRA rank - smaller for efficiency
        'alpha': 16,       # LoRA scaling
        'k': 4,            # SVD rank reduction
        'dropout_p': 0.1,  # Dropout probability
        'ema': 0.95        # EMA for covariances
    }
    
    # Training configuration
    train_config = {
        'batch_size': 4,
        'learning_rate': 2e-4,
        'num_epochs': 2,
        'max_samples': 50,  # Small for testing
        'gradient_accumulation_steps': 2,
        'save_steps': 10,
        'eval_steps': 5
    }
    
    print("\nLoading model and tokenizer...")
    try:
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for stability
            trust_remote_code=True
        )
        
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model.to(device)
        print("Model and tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print(f"\nApplying GRIT to model...")
    # Apply GRIT to the model
    model = apply_grit_to_model(model, **grit_config)
    
    # Freeze non-GRIT parameters
    freeze_non_grit_parameters(model)
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable percentage: {100 * trainable_params / total_params:.4f}%")
    
    # Get GRIT parameters specifically
    grit_params = get_grit_parameters(model)
    grit_param_count = sum(p.numel() for p in grit_params)
    print(f"GRIT parameters: {grit_param_count:,}")
    
    if trainable_params != grit_param_count:
        print("Warning: Trainable params don't match GRIT params. Checking parameters...")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"  Trainable: {name} ({param.numel():,} params)")
    
    print("\nCreating dataset...")
    try:
        # Create datasets
        train_dataset = SimpleVQADataset('train', max_samples=train_config['max_samples'])
        val_dataset = SimpleVQADataset('val', max_samples=train_config['max_samples']//4)
        
        collator = SimpleVQACollator(tokenizer)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_config['batch_size'],
            shuffle=True,
            collate_fn=collator,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_config['batch_size'],
            shuffle=False,
            collate_fn=collator,
            num_workers=0
        )
        
        print("Dataset created successfully")
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return
    
    print("\nSetting up training...")
    
    if not grit_params:
        print("No GRIT parameters found! Check target modules.")
        return
    
    print(f"Optimizing {len(grit_params)} GRIT parameter tensors")
    
    optimizer = torch.optim.AdamW(grit_params, lr=train_config['learning_rate'])
    
    # Learning rate scheduler
    total_steps = len(train_loader) * train_config['num_epochs']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    print("\nStarting training...")
    model.train()
    
    global_step = 0

    
    for epoch in range(train_config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{train_config['num_epochs']}")
        
        epoch_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss
                
                # Scale loss for gradient accumulation
                loss = loss / train_config['gradient_accumulation_steps']
                
                # Backward pass
                loss.backward()
                
                # Update weights every gradient_accumulation_steps
                if (batch_idx + 1) % train_config['gradient_accumulation_steps'] == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(grit_params, 1.0)
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                # Update metrics
                epoch_loss += loss.item() * train_config['gradient_accumulation_steps']
                num_batches += 1
                
                # Update progress bar
                current_lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'loss': f"{loss.item() * train_config['gradient_accumulation_steps']:.4f}",
                    'lr': f"{current_lr:.2e}"
                })
                
                # Evaluation
                if global_step > 0 and global_step % train_config['eval_steps'] == 0:
                    print(f"\nEvaluating at step {global_step}...")
                    eval_loss = evaluate(model, val_loader, device)
                    print(f"Validation loss: {eval_loss:.4f}")
    
                    model.train()
                
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # End of epoch summary
        avg_loss = epoch_loss / max(1, num_batches)
        print(f"\nEpoch {epoch + 1} completed - Average loss: {avg_loss:.4f}")
    
    print("\n🎉 Training completed!")
    
    # Final evaluation
    print("\nFinal evaluation...")
    final_eval_loss = evaluate(model, val_loader, device)
    print(f"Final validation loss: {final_eval_loss:.4f}")
    
    # Save final model
    print("Saving final model...")
    save_grit_weights(model, "final_grit_weights.pt")
    print("All done!")


if __name__ == "__main__":
    main()