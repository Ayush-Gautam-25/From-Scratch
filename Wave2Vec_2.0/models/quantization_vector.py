import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):

    def __init__(self, codebook_size: int, codebook_dim: int, temperature: float = 0.1, gumbel: bool = False):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.temperature = temperature
        self.use_gumbel = gumbel

        self.codebook = nn.Parameter(torch.randn(codebook_size, codebook_dim))

    def forward(self, z: torch.Tensor):
        B, T, D = z.shape
        assert D == self.codebook_dim, f"Input dim {D} != codebook dim {self.codebook_dim}"

        z_flat = z.view(-1, D)
        z_norm_sq = (z_flat ** 2).sum(dim=1, keepdim=True)
        c_norm_sq = (self.codebook ** 2).sum(dim=1)
        dot = torch.matmul(z_flat, self.codebook.t())
        distances = z_norm_sq + c_norm_sq - 2 * dot

        if self.use_gumbel:
            logits = -distances
            probs = F.gumbel_softmax(logits, tau=self.temperature, hard=True)
            q_flat = torch.matmul(probs, self.codebook)
            indices = probs.argmax(dim=-1)
        else:
            indices = distances.argmin(dim=-1)
            q_flat = self.codebook[indices]
            probs = None

        q_z = q_flat.view(B, T, D)
        indices = indices.view(B, T)

        return q_z, indices, probs
