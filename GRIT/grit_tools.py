import torch
import torch
import math
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import _damped_inv

import torch
import math
from torch import nn
import torch.nn.functional as F
from utils import _damped_inv

class GritKFACFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lora_A_w, lora_B_w, scaling, dropout_p, k, module_ref):
        device, dtype = x.device, x.dtype
        ctx.module_ref = module_ref
        ctx.dropout_p = float(dropout_p)
        ctx.scaling = float(scaling)
        ctx.k = int(k)

        lora_A_w = lora_A_w.to(device, dtype=dtype)
        lora_B_w = lora_B_w.to(device, dtype=dtype)

        if ctx.dropout_p > 0 and module_ref.training:
            mask1 = (torch.rand_like(x, device=device) > ctx.dropout_p).to(x.dtype) / (1.0 - ctx.dropout_p)
            x_masked = x * mask1

            u = F.linear(x_masked, lora_A_w, bias=None)
            mask2 = (torch.rand_like(u, device=device) > ctx.dropout_p).to(u.dtype) / (1.0 - ctx.dropout_p)
            u_final = u * mask2

            ctx.save_for_backward(x, x_masked, u_final, lora_A_w, lora_B_w, mask1, mask2)
        else:
            x_masked = x
            mask1 = torch.ones_like(x, device=device)
            u_final = F.linear(x, lora_A_w, bias=None)
            mask2 = torch.ones_like(u_final, device=device)
            ctx.save_for_backward(x, x_masked, u_final, lora_A_w, lora_B_w, mask1, mask2)

        delta = F.linear(u_final, lora_B_w, bias=None) * ctx.scaling
        base_out = module_ref.base_layer(x)
        out = base_out + delta
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, x_masked, u_final, lora_A_w, lora_B_w, mask1, mask2 = ctx.saved_tensors
        module_ref = ctx.module_ref
        scaling = ctx.scaling
        k = ctx.k
        dropout_p = ctx.dropout_p

        device = grad_output.device
        dtype = grad_output.dtype
        g = grad_output.contiguous().to(device=device, dtype=dtype)

        in_features = x.shape[-1]
        out_features = g.shape[-1]
        r = u_final.shape[-1]

        g_2d = g.view(-1, out_features)
        x_masked_2d = x_masked.contiguous().view(-1, in_features)
        u_final_2d = u_final.contiguous().view(-1, r)

        mask1_2d = mask1.contiguous().view(-1, in_features)
        mask2_2d = mask2.contiguous().view(-1, r)

        N = max(1.0, float(g_2d.size(0)))
        grad_u_final_2d = g_2d.matmul(lora_B_w) * scaling
        if dropout_p > 0:
            grad_u_final_2d = grad_u_final_2d * mask2_2d

        # Gradients for LoRA weights
        grad_lora_A = grad_u_final_2d.t().matmul(x_masked_2d)
        grad_lora_B = g_2d.t().matmul(u_final_2d)
        grad_x_lora_2d = grad_u_final_2d.matmul(lora_A_w)
        if dropout_p > 0:
            grad_x_lora_2d = grad_x_lora_2d * mask1_2d

        if hasattr(module_ref.base_layer, "weight"):
            base_w = module_ref.base_layer.weight
            grad_x_base_2d = g_2d.matmul(base_w)
        else:
            grad_x_base_2d = torch.zeros_like(grad_x_lora_2d, device=device, dtype=dtype)

        grad_x_2d = grad_x_base_2d + grad_x_lora_2d
        grad_x = grad_x_2d.view(*x.shape)

        # --- K-FAC top-k preconditioning using _damped_inv (full-damped inverse of low-rank approx) ---
        if module_ref.training and k > 0:
            with torch.no_grad():
                # Update covariance EMA
                A_cov_new = (x_masked_2d.t() @ x_masked_2d) / N        # (in, in)
                G_cov_new = (g_2d.t() @ g_2d) / N                     # (out, out)
                module_ref.A_cov.mul_(module_ref.ema).add_(A_cov_new, alpha=(1.0 - module_ref.ema))
                module_ref.G_cov.mul_(module_ref.ema).add_(G_cov_new, alpha=(1.0 - module_ref.ema))

                # update step counter and decide whether to recompute cached inverses
                module_ref.kfac_step = getattr(module_ref, "kfac_step", 0) + 1
                update_freq = getattr(module_ref, "kfac_update_freq", 1)
                damping = getattr(module_ref, "kfac_damping", 1e-5)
                need_recompute = ((module_ref.kfac_step % update_freq) == 0) or (getattr(module_ref, "kfac_Ainv", None) is None)

                if need_recompute:
                    # Top-k eigen decomposition
                    eigvals_A, eigvecs_A = torch.linalg.eigh(module_ref.A_cov)
                    eigvals_G, eigvecs_G = torch.linalg.eigh(module_ref.G_cov)
                    idx_A = torch.argsort(eigvals_A, descending=True)[:k]
                    idx_G = torch.argsort(eigvals_G, descending=True)[:k]

                    U_A = eigvecs_A[:, idx_A] 
                    U_G = eigvecs_G[:, idx_G] 
                    sA = eigvals_A[idx_A]     
                    sG = eigvals_G[idx_G]     

                    # Build low-rank approximations M = U S U^T
                    U_A = U_A.to(device=device, dtype=dtype)
                    U_G = U_G.to(device=device, dtype=dtype)
                    S_A = torch.diag(sA.to(device=device, dtype=dtype))
                    S_G = torch.diag(sG.to(device=device, dtype=dtype))

                    M_A = U_A.matmul(S_A).matmul(U_A.t())
                    M_G = U_G.matmul(S_G).matmul(U_G.t())

                    A_inv = _damped_inv(M_A, damping=damping)
                    G_inv = _damped_inv(M_G, damping=damping)

                    module_ref.kfac_Ainv = A_inv
                    module_ref.kfac_Ginv = G_inv
                    module_ref.kfac_UA = U_A
                    module_ref.kfac_UG = U_G
                    module_ref.kfac_sA = sA.to(device=device, dtype=dtype)
                    module_ref.kfac_sG = sG.to(device=device, dtype=dtype)
                else:
                    A_inv = module_ref.kfac_Ainv.to(device=device, dtype=dtype)
                    G_inv = module_ref.kfac_Ginv.to(device=device, dtype=dtype)

                # Precondition LoRA gradients:
                grad_lora_A = grad_lora_A.matmul(A_inv)
                grad_lora_B = G_inv.matmul(grad_lora_B)

        return grad_x, grad_lora_A, grad_lora_B, None, None, None, None


# GritLinear: add kfac_update_freq and kfac_damping defaults + placeholders
class GritLinear(nn.Module):
    def __init__(self, module: nn.Linear, r=8, alpha=16, k=4,
                 dropout_p=0.05, enable_grit=True, ema=0.95,
                 kfac_update_freq: int = 1, kfac_damping: float = 1e-5):
        super().__init__()
        assert isinstance(module, nn.Linear), "GritLinear must wrap nn.Linear"
        self.base_layer = module
        self.enable_grit = enable_grit and (r > 0)
        self.r = r
        self.scaling = float(alpha) / float(r) if self.enable_grit else 1.0
        self.dropout_p = float(dropout_p) if self.enable_grit else 0.0
        self.k = max(1, min(k, r)) if self.enable_grit else 0
        self.ema = float(ema)

        # K-FAC params
        self.kfac_update_freq = int(kfac_update_freq)
        self.kfac_damping = float(kfac_damping)
        self.kfac_step = 0

        # freeze base layer parameters
        for p in self.base_layer.parameters():
            p.requires_grad = False

        if self.enable_grit:
            in_features = self.base_layer.in_features
            out_features = self.base_layer.out_features

            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
            param_dtype = next(self.base_layer.parameters()).dtype

            # covariance buffers (will move with .to(device))
            self.register_buffer("A_cov", torch.eye(in_features, dtype=param_dtype) * 1e-3)
            self.register_buffer("G_cov", torch.eye(out_features, dtype=param_dtype) * 1e-3)

            # placeholders / caches (not registered buffers)
            self.kfac_Ainv = None
            self.kfac_Ginv = None
            self.kfac_UA = None
            self.kfac_UG = None
            self.kfac_sA = None
            self.kfac_sG = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enable_grit:
            return self.base_layer(x)
        return GritKFACFunction.apply(
            x, self.lora_A.weight, self.lora_B.weight,
            self.scaling, self.dropout_p, self.k, self
        )


    def reproject_to_topk(self):
        """
        Explicitly enforce theta_new = U U^T theta for LoRA params using cached U_A / U_G.
        Call this after optimizer.step() (or periodically).
        """
        if not self.enable_grit:
            return

        if getattr(self, "kfac_UA", None) is None or getattr(self, "kfac_UG", None) is None:
            return

        device = self.lora_A.weight.device
        dtype = self.lora_A.weight.dtype

        U_A = self.kfac_UA.to(device=device, dtype=dtype) 
        U_G = self.kfac_UG.to(device=device, dtype=dtype)

        # compute projection matrices
        with torch.no_grad():
            P_A = U_A.matmul(U_A.t())
            self.lora_A.weight.copy_(self.lora_A.weight.matmul(P_A))

            P_G = U_G.matmul(U_G.t())
            self.lora_B.weight.copy_(P_G.matmul(self.lora_B.weight))


def apply_grit_to_model(model: nn.Module, target_modules=None, **grit_kwargs):
    """
    Replace specific nn.Linear modules in `model` with GritLinear wrappers.
    """
    if target_modules is None:
        target_modules = [
            "q_proj", 
            # "k_proj",
            # "v_proj", "o_proj",
            # "gate_proj", "up_proj", "down_proj",
            # "dense", "linear", "proj", "projection", "down_proj"
        ]

    # collect candidates first
    layers_to_replace = []
    for name, module in model.named_modules():
        for t in target_modules:
            if name.endswith(t) and isinstance(module, nn.Linear):
                # skip tiny layers and embedding / lm_head
                if module.in_features < 32 or module.out_features < 32:
                    continue
                if "embed" in name.lower() or "lm_head" in name.lower():
                    continue
                layers_to_replace.append((name, module))

    print(f"Found {len(layers_to_replace)} candidate layers to replace with GRIT")

    replaced_count = 0
    name2module = dict(model.named_modules())
    for name, orig_module in layers_to_replace:
        try:
            if "." in name:
                parent_name, attr_name = name.rsplit(".", 1)
                parent = name2module[parent_name]
            else:
                parent = model
                attr_name = name

            grit_layer = GritLinear(orig_module, **grit_kwargs)

            try:
                module_device = next(orig_module.parameters()).device
            except StopIteration:
                module_device = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu")

            grit_layer.to(module_device)

            setattr(parent, attr_name, grit_layer)
            replaced_count += 1
            print(f"Replaced {name} -> GRIT (in:{orig_module.in_features}, out:{orig_module.out_features})")
        except Exception as e:
            print(f"Failed to replace {name}: {e}")

    print(f"Successfully replaced {replaced_count} layers with GRIT")
    return model


def get_grit_parameters(model):
    grit_params = []
    for name, module in model.named_modules():
        if isinstance(module, GritLinear) and module.enable_grit:
            grit_params.append(module.lora_A.weight)
            grit_params.append(module.lora_B.weight)
    return grit_params


def reproject_grit_modules(model):
    """
    Call this AFTER optimizer.step() (or periodically)
    to enforce theta_new = U U^T theta_updated on all GritLinear modules.
    """
    for name, module in model.named_modules():
        if isinstance(module, GritLinear) and module.enable_grit:
            module.reproject_to_topk()


def print_grit_devices(model):
    for name, module in model.named_modules():
        if isinstance(module, GritLinear):
            a_dev = module.lora_A.weight.device if hasattr(module, "lora_A") else None
            b_dev = module.lora_B.weight.device if hasattr(module, "lora_B") else None
            a_cov = module.A_cov.device if "A_cov" in dict(module.named_buffers()) else None
            g_cov = module.G_cov.device if "G_cov" in dict(module.named_buffers()) else None
            print(name, "lora_A:", a_dev, "lora_B:", b_dev, "A_cov:", a_cov, "G_cov:", g_cov)


def evaluate(model, val_loader, device):
    """Evaluate the model on validation set"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            try:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = model(**batch)
                total_loss += outputs.loss.item()
                num_batches += 1
                
            except Exception as e:
                print(f"Evaluation error: {e}")
                continue
    
    return total_loss / max(1, num_batches)


def save_grit_weights(model, filename):
    """Save only GRIT weights"""
    grit_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, GritLinear) and module.enable_grit:
            grit_state_dict[f"{name}.lora_A.weight"] = module.lora_A.weight.clone()
            grit_state_dict[f"{name}.lora_B.weight"] = module.lora_B.weight.clone()
    
    torch.save(grit_state_dict, filename)
    print(f"Saved GRIT weights to {filename} ({len(grit_state_dict)} tensors)")


def load_grit_weights(model, filename):
    """Load GRIT weights"""
    grit_state_dict = torch.load(filename)
    
    loaded_count = 0
    for name, module in model.named_modules():
        if isinstance(module, GritLinear) and module.enable_grit:
            if f"{name}.lora_A.weight" in grit_state_dict:
                module.lora_A.weight.data = grit_state_dict[f"{name}.lora_A.weight"]
                loaded_count += 1
            if f"{name}.lora_B.weight" in grit_state_dict:
                module.lora_B.weight.data = grit_state_dict[f"{name}.lora_B.weight"]
                loaded_count += 1
    
    print(f"Loaded {loaded_count} GRIT weight tensors from {filename}")