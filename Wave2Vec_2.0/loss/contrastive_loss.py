import torch
from torch import nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = max(temperature, 1e-3)


    def forward(self, context: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        B, T, D = context.shape

        context_masked, target_masked = context[mask], target[mask]
        N = context_masked.size(0)
        # print("context_masked:", context_masked.shape)
        # print("targets_masked:", target_masked.shape)
        # print(context_masked, )
        # Normalize for cosine similarity
        context_norm = F.normalize(context_masked, dim=-1)
        targets_norm = F.normalize(target_masked, dim=-1)
        # print("context_norm mean:", context_norm.mean().item())
        # print("targets_norm mean:", targets_norm.mean().item())

        assert context_masked.shape == target_masked.shape, "Masked context and target shapes mismatch"
        assert N > 0, "No masked positions in batch!"



        logits = torch.matmul(context_norm, targets_norm.T) / self.temperature  # [N, N]

        # print("logits stats:", logits.min().item(), logits.max().item(), logits.mean().item())

        # Ground truth is diagonal
        labels = torch.arange(N, device=context.device)
        loss = F.cross_entropy(logits, labels)

        return loss

