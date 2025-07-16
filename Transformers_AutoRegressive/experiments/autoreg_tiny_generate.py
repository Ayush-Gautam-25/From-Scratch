import torch
import os
import json
import sys

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.autoreg import AutoregressiveModel
from config.config import autoreg_tiny_config
from tokenizer.bpe_tokenizer import BPE

# Load BPE tokenizer used during training
tokenizer_path = "../data/wikitext-103/bpe_tokenizer"
bpe = BPE()
bpe.load(tokenizer_path)

token_to_id = json.load(open(os.path.join(tokenizer_path, "token_to_id.json")))
autoreg_tiny_config.vocab_size = len(token_to_id)

# Load model and weights
model = AutoregressiveModel(autoreg_tiny_config)
model.load_state_dict(torch.load("../checkpoints/autoreg_tiny/autoreg_tiny.pt"))
model = model.cuda().eval()


# Encode a prompt
prompt = "Once upon a time"
prompt_ids = bpe.encode_text(prompt)  # adds [BOS] and [EOS]
input_ids = torch.tensor([prompt_ids], dtype=torch.long).cuda()

# Generate more tokens
with torch.no_grad():
    output = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    # eos_token_id=3
)


# Decode and print
decoded = bpe.decode_text(output[0].tolist())
print("Generated Text:\n", decoded)
