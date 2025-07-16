import torch
from torch import nn
from typing import Optional
import math
import torch.nn.functional as F
from models.moe_transformer_encoder import TransformerMoEBlock


class TransformerEncoderLayer(nn.Module):
    def __init__(self, n_heads: int, dim_model: int, dim_ffn: int, use_moe: bool=False, dropout: Optional[float]=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.dim_ffn = dim_ffn
        self.head_dim = dim_model//n_heads

        assert dim_model % n_heads == 0, "dim_model must be divisible by n_heads"

        self.q_proj = nn.Linear(dim_model, dim_model)
        self.k_proj = nn.Linear(dim_model, dim_model)
        self.v_proj = nn.Linear(dim_model, dim_model)
        self.out_proj = nn.Linear(dim_model, dim_model)

        if use_moe:
            self.ffn = TransformerMoEBlock(
                latent_dim=dim_model,
                ffn_dim=dim_ffn,
                n_experts=4,
                top_k=1
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(dim_model, dim_ffn),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_ffn, dim_model),
            )



        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(dropout)

    def attention(self, Q, K, V, mask=None):
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        out = attn @ V 
        return out

    def forward(self, x):
        B, T, D = x.shape

        Q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        K = self.k_proj(x).view(B, T, self.n_heads, self.head_dim)
        V = self.v_proj(x).view(B, T, self.n_heads, self.head_dim)

        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        attn_out = self.attention(Q, K, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.dropout(self.out_proj(attn_out))
        x = self.norm1(x)

        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm2(x)

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, num_layers=6, n_heads=8, dim_ffn=2048, dropout=0.1, max_len=1000, use_moe: bool=False):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(n_heads, d_model, dim_ffn, use_moe, dropout)
            for _ in range(num_layers)
        ])

        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        """
        x: [B, T, D]
        """
        B, T, D = x.shape
        device = x.device

        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T) 
        
        pos_embed = self.pos_embedding(pos_ids)
        x = x + pos_embed

        for layer in self.layers:
            x = layer(x)
        return x
