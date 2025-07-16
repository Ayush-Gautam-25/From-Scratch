import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from autoreg import AutoregressiveModel
from config.config import AutoregModelConfig

# Sample config for testing
config = AutoregModelConfig(
    vocab_size=2000,
    block_size=64,
    n_embeds=256,
    n_heads=4,
    n_layer=2,
    dropout=0.1
)

# Initialize model
model = AutoregressiveModel(config).cuda()
model.train()

# Create dummy input
B, T = 2, 32  # batch size and sequence length
dummy_input = torch.randint(0, config.vocab_size, (B, T), dtype=torch.long).cuda()

# Forward pass
output = model(dummy_input)
print("✅ Forward pass successful. Output shape:", output.shape)

# Create dummy target for backward pass
dummy_target = torch.randint(0, config.vocab_size, (B, T), dtype=torch.long).cuda()
loss_fn = torch.nn.CrossEntropyLoss()

# Reshape for CrossEntropyLoss (expects (B*T, vocab_size) and (B*T,))
loss = loss_fn(output.view(-1, config.vocab_size), dummy_target.view(-1))

# Backward pass
loss.backward()
print("✅ Backward pass successful. Gradients computed.")

# Optionally check a gradient exists
print("Gradient of token embedding:", model.token_emb.weight.grad.norm().item())
