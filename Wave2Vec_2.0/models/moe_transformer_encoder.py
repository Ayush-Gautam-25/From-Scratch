import torch
from torch import nn
import torch.nn.functional as F

class TransformerMoEBlock(nn.Module):
    def __init__(self, latent_dim: int = 512, ffn_dim: int = 2048, n_experts: int = 4, top_k: int = 1, eps: float = 1e-2):
        super().__init__()
        self.gate = nn.Linear(latent_dim, n_experts)
        self.experts = nn.ModuleList([
                nn.Sequential(
                nn.Linear(latent_dim, ffn_dim),
                nn.ReLU(),
                nn.Linear(ffn_dim, latent_dim),
            ) for _ in range(n_experts)
        ])

        self.top_k = top_k
        self.eps = eps
        self.n_experts = n_experts


    def forward(self, x):
        B, T, D = x.size()  # [batch, seq_len, dim]
        x_flat = x.view(-1, D)  # [B*T, D]
        logits = self.gate(x)  # [B, T, n_experts]
        gates = F.softmax(logits, dim=-1)  # [B, T, n_experts]

        topk_vals, topk_idx = torch.topk(gates, self.top_k, dim=-1)  # [B, T, k], if k=1, then [B, T, 1]
        topk_vals = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + self.eps)

        # Flatten everything to 2D
        flat_topk_idx = topk_idx.view(-1, self.top_k)  # [B*T, k], if k=1, then [B*T, 1]
        flat_topk_vals = topk_vals.view(-1, self.top_k)  # [B*T, k], if k=1, then [B*T, 1]

        output = torch.zeros_like(x_flat)  # [B*T, D]

        for expert_id in range(self.n_experts):
            # Get the positions where this expert is chosen
            mask = (flat_topk_idx == expert_id)  # [B*T, k]
            if not mask.any():
                continue

            # Collect all inputs for this expert
            expert_mask = mask.float() * flat_topk_vals  # zero out unused
            expert_mask_sum = expert_mask.sum(dim=-1)  # [B*T]
            selected_indices = (expert_mask_sum > 0).nonzero(as_tuple=False).squeeze()

            if selected_indices.numel() == 0:
                continue

            expert_inputs = x_flat[selected_indices]  # [N, D]
            expert_outputs = self.experts[expert_id](expert_inputs)  # [N, D]

            # Scatter back to output
            output[selected_indices] += expert_outputs * expert_mask_sum[selected_indices].unsqueeze(-1)

        return output.view(B, T, D)
