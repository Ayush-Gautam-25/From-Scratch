import torch
from torch import nn
import sys
import os
import math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.config import AutoregModelConfig
# from kernels.gelu import GELU
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, config: AutoregModelConfig) -> None:
        super().__init__()
        self.config = config
        self.Wk = nn.Linear(self.config.n_embeds, self.config.n_embeds)
        self.Wv = nn.Linear(self.config.n_embeds, self.config.n_embeds)
        self.Wq = nn.Linear(self.config.n_embeds, self.config.n_embeds)
        self.proj = nn.Linear(self.config.n_embeds, self.config.n_embeds)

        self.n_heads = config.n_heads
        self.head_dim = config.n_embeds//config.n_heads

        ## following dropouts to prevent overfitting
        self.attention_dropout = nn.Dropout(config.dropout)
        self.final_dropout = nn.Dropout(config.dropout)

        self.register_buffer('mask', torch.tril(torch.ones(config.block_size, config.block_size)).unsqueeze(0).unsqueeze(0)) # to create mask of size (1, 1, block_size, block_size), this ensures the causality, basically to prevent model from seeing future tokens which can cause data leaks and overfitting but not learning
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, embed_dim = x.size()
        assert embed_dim == self.config.n_embeds, "❌ Embedding Dimension of Input does not match the Model's Embedding Dimension!!"

        # At time of training, the seq_length == block_size, but at inference time, seq_length <= block_size


        k = self.Wk(x).view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.Wq(x).view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.Wv(x).view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)

        omega = (q @ k.transpose(-2, -1))
        normalized_omega = omega / math.sqrt(self.head_dim)
        normalized_omega = normalized_omega.masked_fill(self.mask[:, :, :seq_length, :seq_length] == 0, float("-inf"))
        soft = torch.softmax(normalized_omega, dim=-1)
        soft = self.attention_dropout(soft)

        output = soft @ v 
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
        return self.final_dropout(self.proj(output))
    
class TransformerBlock(nn.Module):
    def __init__(self, config: AutoregModelConfig) -> None:
        super().__init__()
        self.config = config
        self.attention = CausalSelfAttention(config)
        self.layernorm1 = nn.LayerNorm(config.n_embeds)
        self.layernorm2 = nn.LayerNorm(config.n_embeds)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embeds, 4 * config.n_embeds),
            nn.GELU(),
            nn.Linear(4 * config.n_embeds, config.n_embeds),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.layernorm1(x))
        x = x + self.mlp(self.layernorm2(x))
        return x
    

class AutoregressiveModel(nn.Module):
    def __init__(self, config: AutoregModelConfig):
        super().__init__()
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embeds)
        self.config = config
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embeds))
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embeds)
        self.head = nn.Linear(config.n_embeds, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.block_size, "Sequence too long"
        min_id, max_id = idx.min().item(), idx.max().item()
        vs = self.config.vocab_size
        assert 0 <= min_id < vs and 0 <= max_id < vs, \
            f"Token IDs out of range: [{min_id}, {max_id}], vocab_size={vs}"

        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb[:, :T, :]
        x = self.drop(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, eos_token_id=None):
        for i in range(max_new_tokens):
            print("Generating Token:", i + 1)
            idx_ = idx[:, -self.block_size:]
            logits = self(idx_) # model forward passes the idx_cond
            logits = logits[:, -1, :]
            logits = logits / temperature

            # Top-k filtering
            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                min_threshold = values[:, -1].unsqueeze(1)
                logits = torch.where(logits < min_threshold, torch.full_like(logits, float('-inf')), logits)

            # Top-p filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

                sorted_mask = cumulative_probs > top_p
                sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                sorted_mask[:, 0] = False

                logits.scatter_(1, sorted_indices, torch.where(sorted_mask, float('-inf'), sorted_logits))

            # Sampling
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

            # Early exit: check if eos_token was generated
            if eos_token_id is not None:
                if (next_token == eos_token_id).all():
                    print(f"Early stopping at step {i + 1} due to <eos> token.")
                    break

        return idx















    
