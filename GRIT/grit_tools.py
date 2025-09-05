import torch
import math
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

class GritKFACFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lora_A_w, lora_B_w, scaling, dropout_p, k, module_ref):
        """
        x: (B, in)
        lora_A_w: Parameter tensor shape (r, in)
        lora_B_w: Parameter tensor shape (out, r)
        """
        device = x.device
        ctx.module_ref = module_ref
        ctx.dropout_p = float(dropout_p)
        ctx.scaling = float(scaling)
        ctx.k = int(k)

        # ensure weights are on same device (they should be, but keep safe)
        lora_A_w = lora_A_w.to(device)
        lora_B_w = lora_B_w.to(device)

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

        # compute base output using the frozen base linear
        base_out = module_ref.base_layer(x)
        out = base_out + delta
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass that supports inputs with arbitrary leading dims
        (e.g. [B, S, in_features]). We flatten leading dims -> 2D for matmuls,
        then reshape back.
        """
        
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
        mask1_2d = mask1.contiguous().view(-1, in_features) if mask1 is not None else None
        mask2_2d = mask2.contiguous().view(-1, r) if mask2 is not None else None

        N = max(1.0, float(g_2d.size(0)))

        grad_u_final_2d = g_2d.matmul(lora_B_w) * scaling

        if (dropout_p > 0) and (mask2_2d is not None):
            grad_u_final_2d = grad_u_final_2d * mask2_2d

        # Gradients for LoRA weights (2D)
        grad_lora_A = grad_u_final_2d.t().matmul(x_masked_2d)
        grad_lora_B = g_2d.t().matmul(u_final_2d)
        grad_x_lora_2d = grad_u_final_2d.matmul(lora_A_w)

        if (dropout_p > 0) and (mask1_2d is not None):
            grad_x_lora_2d = grad_x_lora_2d * mask1_2d

        # Gradient w.r.t inputs from base linear:
        if hasattr(module_ref.base_layer, "weight"):
            base_w = module_ref.base_layer.weight
            grad_x_base_2d = g_2d.matmul(base_w)
        else:
            grad_x_base_2d = torch.zeros_like(grad_x_lora_2d, device=device, dtype=dtype)

        # Combine input gradients and reshape back to original x shape
        grad_x_2d = grad_x_base_2d + grad_x_lora_2d
        grad_x = grad_x_2d.view(*x.shape)

        # Update covariance EMA
        if module_ref.training and getattr(module_ref, "k", 0) > 0:
            with torch.no_grad():
                A_cov_new = (x_masked_2d.t().matmul(x_masked_2d)) / N 
                module_ref.A_cov.mul_(module_ref.ema).add_(A_cov_new, alpha=(1.0 - module_ref.ema))

                G_cov_new = (g_2d.t().matmul(g_2d)) / N
                module_ref.G_cov.mul_(module_ref.ema).add_(G_cov_new, alpha=(1.0 - module_ref.ema))

        # Return gradients for forward args:
        return grad_x, grad_lora_A, grad_lora_B, None, None, None, None



# -------------------------
# GritLinear wrapper
# -------------------------
class GritLinear(nn.Module):
    def __init__(self, module: nn.Linear, r=8, alpha=16, k=4,
                 dropout_p=0.05, enable_grit=True, ema=0.95):
        """
        Wrap an existing nn.Linear with a GRIT LoRA + K-FAC bookkeeping.
        - base layer is frozen (weights not trained)
        - lora_A: (in -> r), stored as nn.Linear(in, r)
        - lora_B: (r -> out), stored as nn.Linear(r, out)
        """
        super().__init__()
        assert isinstance(module, nn.Linear), "GritLinear must wrap nn.Linear"
        self.base_layer = module
        self.enable_grit = enable_grit and (r > 0)
        self.r = r
        self.scaling = float(alpha) / float(r) if self.enable_grit else 1.0
        self.dropout_p = float(dropout_p) if self.enable_grit else 0.0
        self.k = max(1, min(k, r)) if self.enable_grit else 0
        self.ema = float(ema)

        # freeze base layer parameters (they remain in model but are not trained)
        for p in self.base_layer.parameters():
            p.requires_grad = False

        if self.enable_grit:
            in_features = self.base_layer.in_features
            out_features = self.base_layer.out_features

            # define LoRA adapters:
            # lora_A.weight shape: (r, in)
            # lora_B.weight shape: (out, r)
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)

            # init
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

            # Ensure trainable
            self.lora_A.weight.requires_grad_(True)
            self.lora_B.weight.requires_grad_(True)

            # covariance buffers (will move with .to(device))
            self.register_buffer("A_cov", torch.eye(in_features, dtype=torch.float32) * 1e-3)
            self.register_buffer("G_cov", torch.eye(out_features, dtype=torch.float32) * 1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enable_grit:
            return self.base_layer(x)
        return GritKFACFunction.apply(
            x, self.lora_A.weight, self.lora_B.weight,
            self.scaling, self.dropout_p, self.k, self
        )
    

def apply_grit_to_model(model: nn.Module, target_modules=None, **grit_kwargs):
    """
    Replace specific nn.Linear modules in `model` with GritLinear wrappers.
    Ensures new wrapper is moved to the same device as the original module.
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
            # find parent
            if "." in name:
                parent_name, attr_name = name.rsplit(".", 1)
                parent = name2module[parent_name]
            else:
                parent = model
                attr_name = name

            # create wrapper
            grit_layer = GritLinear(orig_module, **grit_kwargs)

            # find device of original module (fallback to model device)
            try:
                module_device = next(orig_module.parameters()).device
            except StopIteration:
                module_device = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu")

            # move grit wrapper to the same device so buffers/params align
            grit_layer.to(module_device)

            # set attr on parent
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
            # append Parameter objects (weights)
            grit_params.append(module.lora_A.weight)
            grit_params.append(module.lora_B.weight)
    return grit_params


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