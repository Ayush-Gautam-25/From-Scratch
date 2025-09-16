import torch
import torch
import math
from torch import nn
import torch.nn.functional as F

class GritLinear(nn.Module):
    def __init__(self, module: nn.Linear, r=8, alpha=16, k=4,
                 dropout_p=0.05, enable_grit=True, ema=0.95,
                 kfac_damping: float = 1e-5):
        super().__init__()
        assert isinstance(module, nn.Linear), "GritLinear must wrap nn.Linear"
        self.base_layer = module
        self.enable_grit = enable_grit and (r > 0)
        self.r = r
        self.scaling = float(alpha) / float(r) if self.enable_grit else 1.0
        self.dropout_p = float(dropout_p) if self.enable_grit else 0.0
        self.k = max(1, min(k, r)) if self.enable_grit else 0
        self.ema = float(ema)
        self.kfac_damping = float(kfac_damping)

        # Freeze base layer parameters
        for p in self.base_layer.parameters():
            p.requires_grad = False

        if self.enable_grit:
            in_features = self.base_layer.in_features
            out_features = self.base_layer.out_features

            # LoRA parameters
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

            # K-FAC covariance buffers
            self.register_buffer("A_cov", torch.eye(in_features, dtype=torch.float32) * 1e-3)
            self.register_buffer("G_cov", torch.eye(out_features, dtype=torch.float32) * 1e-3)

            self.kfac_UA = None  # Top-k eigenvectors of A_cov
            self.kfac_UG = None  # Top-k eigenvectors of G_cov
            
            # Store activations and gradients for K-FAC updates
            self.stored_activations = None
            self.stored_gradients = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enable_grit:
            return self.base_layer(x)

        # Store activations for K-FAC update
        if self.training:
            self.stored_activations = x.detach()

        # Base layer computation
        base_out = self.base_layer(x)

        # LoRA computation with dropout
        if self.dropout_p > 0 and self.training:
            x_dropped = F.dropout(x, p=self.dropout_p, training=self.training)
            lora_out = self.lora_B(self.lora_A(x_dropped))
        else:
            lora_out = self.lora_B(self.lora_A(x))

        return base_out + lora_out * self.scaling

    def update_kfac_statistics(self, grad_output):
        """Update K-FAC covariance matrices during backward pass"""
        if not self.training or self.stored_activations is None:
            return

        # Store gradient for potential use
        self.stored_gradients = grad_output.detach()

        # Flatten tensors for covariance computation
        x_flat = self.stored_activations.view(-1, self.stored_activations.shape[-1])
        g_flat = grad_output.view(-1, grad_output.shape[-1])
        
        batch_size = x_flat.shape[0]

        # Update covariance matrices with EMA
        with torch.no_grad():
            # Input covariance: A = E[xx^T]
            A_new = (x_flat.T @ x_flat) / batch_size
            self.A_cov.mul_(self.ema).add_(A_new, alpha=(1.0 - self.ema))

            # Gradient covariance: G = E[gg^T]  
            G_new = (g_flat.T @ g_flat) / batch_size
            self.G_cov.mul_(self.ema).add_(G_new, alpha=(1.0 - self.ema))

    def compute_subspace_bases(self):
        """Compute top-k eigenvector bases for neural reprojection"""
        if not self.training or self.k <= 0:
            return

        with torch.no_grad():
            # Eigendecomposition of covariance matrices
            eigvals_A, eigvecs_A = torch.linalg.eigh(self.A_cov)
            eigvals_G, eigvecs_G = torch.linalg.eigh(self.G_cov)

            # Get top-k eigenvectors (largest eigenvalues)
            idx_A = torch.argsort(eigvals_A, descending=True)[:self.k]
            idx_G = torch.argsort(eigvals_G, descending=True)[:self.k]

            self.kfac_UA = eigvecs_A[:, idx_A].detach()
            self.kfac_UG = eigvecs_G[:, idx_G].detach()

def grit_natural_gradient_step(model, optimizer):
    """
    Apply GRIT natural gradient update with K-FAC preconditioning.
    Call this INSTEAD of optimizer.step() when using GRIT.
    """
    # First collect gradients and compute natural gradients
    for name, module in model.named_modules():
        if isinstance(module, GritLinear) and module.enable_grit:
            if module.kfac_UA is not None and module.kfac_UG is not None:
                # Get current gradients
                grad_A = module.lora_A.weight.grad
                grad_B = module.lora_B.weight.grad
                
                if grad_A is not None and grad_B is not None:
                    device = grad_A.device
                    dtype = grad_A.dtype
                    
                    UA = module.kfac_UA.to(device=device, dtype=dtype)
                    UG = module.kfac_UG.to(device=device, dtype=dtype)
                    
                    # Get eigenvalues for damped inversion
                    eigvals_A = torch.linalg.eigvals(module.A_cov).real
                    eigvals_G = torch.linalg.eigvals(module.G_cov).real
                    
                    # Select top-k eigenvalues (same indices as used in compute_subspace_bases)
                    idx_A = torch.argsort(eigvals_A, descending=True)[:module.k]
                    idx_G = torch.argsort(eigvals_G, descending=True)[:module.k]
                    
                    lambda_A = eigvals_A[idx_A].to(device=device, dtype=dtype)
                    lambda_G = eigvals_G[idx_G].to(device=device, dtype=dtype)
                    
                    # Damped inversion in subspace
                    inv_lambda_A = 1.0 / (lambda_A + module.kfac_damping)
                    inv_lambda_G = 1.0 / (lambda_G + module.kfac_damping)
                    
                    # Natural gradient computation using efficient subspace operations
                    # For lora_A
                    grad_A_proj = grad_A @ UA  # (r, k)
                    grad_A_precond = grad_A_proj * inv_lambda_A.unsqueeze(0)  # (r, k)
                    natural_grad_A = grad_A_precond @ UA.T  # (r, in_features)
                    
                    # For lora_B
                    grad_B_proj = UG.T @ grad_B
                    grad_B_precond = grad_B_proj * inv_lambda_G.unsqueeze(1)
                    natural_grad_B = UG @ grad_B_precond
                    
                    # Replace gradients with natural gradients
                    module.lora_A.weight.grad = natural_grad_A
                    module.lora_B.weight.grad = natural_grad_B

    # Apply optimizer step with natural gradients
    optimizer.step()


def grit_neural_reprojection(model):
    """
    Apply neural reprojection: θ_new = U_k U_k^T θ_updated
    Call this AFTER the optimizer step.
    """
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, GritLinear) and module.enable_grit:
                if module.kfac_UA is not None and module.kfac_UG is not None:
                    device = module.lora_A.weight.device
                    dtype = module.lora_A.weight.dtype
                    
                    UA = module.kfac_UA.to(device=device, dtype=dtype)
                    UG = module.kfac_UG.to(device=device, dtype=dtype)
                    
                    # Neural reprojection: project weights onto top-k subspace
                    # For lora_A
                    proj_A = UA @ UA.T  # Projection matrix
                    module.lora_A.weight.data = module.lora_A.weight.data @ proj_A
                    
                    # For lora_B 
                    proj_G = UG @ UG.T  # Projection matrix
                    module.lora_B.weight.data = proj_G @ module.lora_B.weight.data

def apply_grit_to_model(model: nn.Module, target_modules=None, **grit_kwargs):
    """
    Replace specific nn.Linear modules in `model` with GritLinear wrappers.
    """
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            
            "gate_proj", "up_proj", "down_proj",
            "dense", "linear", "proj", "projection", "down_proj"
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

def save_grit_weights(model, filename):
    """Save only GRIT weights"""
    grit_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, GritLinear) and module.enable_grit:
            grit_state_dict[f"{name}.lora_A.weight"] = module.lora_A.weight.clone()
            grit_state_dict[f"{name}.lora_B.weight"] = module.lora_B.weight.clone()
    
    torch.save(grit_state_dict, filename)
    print(f"Saved GRIT weights to {filename} ({len(grit_state_dict)} tensors)")
