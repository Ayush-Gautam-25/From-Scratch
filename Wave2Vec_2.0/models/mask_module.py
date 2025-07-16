from torch import nn
import torch
from typing import Optional

class MaskingModule(nn.Module):
    def __init__(self, emb_dim: int = 512, mask_prob: Optional[float] = 0.2, mask_span: Optional[int] = 10):
        super().__init__()
        self.mask_prob = mask_prob
        self.mask_span = mask_span
        self.mask_emb = nn.Parameter(torch.randn(emb_dim))

    def compute_mask_indices(self, B, T, device):
        mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        num_mask_spans = max(1, int(self.mask_prob*(T/self.mask_span)))
        # print(num_mask_spans)
        max_start = max(1, T - self.mask_span)


        for b in range(B):
            if max_start <= 1:
                start_indices = torch.tensor([0], device=device)
            else:
                start_indices = torch.randperm(max_start, device=device)[:num_mask_spans]
            for start in start_indices:
                end = min(T, start + self.mask_span)
                mask[b, start:end] = True


        return mask

    def forward(self, z_t):
        B, T, D = z_t.shape
        device = z_t.device

        mask_indices = self.compute_mask_indices(B, T, device)
        z_t_mask = z_t.clone()

        z_t_mask[mask_indices] = self.mask_emb.to(device=device)

        return z_t_mask, mask_indices
