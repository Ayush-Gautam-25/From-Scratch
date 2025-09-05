import torch
def _damped_inv(M: torch.Tensor, damping: float):
    """Safe damped inverse with fallback"""
    if M.numel() == 0 or M.size(0) == 0:
        return M
    device = M.device
    dtype = M.dtype
    eye = torch.eye(M.size(0), device=device, dtype=dtype)
    try:
        L = torch.linalg.cholesky(M + damping * eye)
        inv = torch.cholesky_inverse(L)
        if torch.isnan(inv).any() or torch.isinf(inv).any():
            raise RuntimeError("Invalid Cholesky result")
        return inv
    except Exception:
        try:
            inv = torch.linalg.pinv(M + damping * eye)
            return inv
        except Exception:
            return eye

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def freeze_non_grit_parameters(model):
    for name, param in model.named_parameters():
        if 'lora_A.weight' in name or 'lora_B.weight' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False