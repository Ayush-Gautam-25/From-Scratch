from transformers import AutoProcessor, AutoModelForVision2Seq, PaliGemmaForConditionalGeneration, TorchAoConfig, MllamaForConditionalGeneration
import torch
from torch import nn
import math
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import torch
from PIL import Image
from io import BytesIO
import requests
from tqdm import tqdm

def grit_training_step(model, batch, optimizer, lambda_k=0.2, lambda_r=0.1, accumulation_steps=1, step_count=0):

    outputs = model(**batch)
    task_loss = outputs.loss / accumulation_steps

    for module in model.modules():
        if isinstance(module, GRITLinear) and module.enable_grit:
            if module.stored_activations is not None and module.stored_gradients is not None:
                module.update_kfac_stats()

    curvature_loss = 0.0
    reprojection_loss = 0.0
    grit_params = []  

    for _, module in model.named_modules():
        if isinstance(module, GRITLinear) and module.enable_grit:
            A = module.lora_A.weight
            B = module.lora_B.weight
            AB = B @ A

            F = AB.numel()
            curvature_loss += torch.norm(AB / (F ** 0.5), p='fro') ** 2

            # Reprojection loss (if top-k directions exist)
            if getattr(module, "topk_A", None) is not None and getattr(module, "topk_G", None) is not None:
                UA = module.topk_A.to(AB.device, AB.dtype)
                UG = module.topk_G.to(AB.device, AB.dtype)
                
                reproj = UG @ (UG.T @ AB @ UA) @ UA.T
                reprojection_loss += torch.norm(AB - reproj, p='fro') ** 2

            grit_params.extend([A, B])

    total_loss = (
        task_loss
        + lambda_k * curvature_loss 
        + lambda_r * reprojection_loss
    )

    total_loss.backward()

    should_step = (step_count + 1) % accumulation_steps == 0
    if should_step:
        for module in model.modules():
            if isinstance(module, GRITLinear) and module.enable_grit:
                module.compute_topk_directions()

        torch.nn.utils.clip_grad_norm_(grit_params, 1.0)

        grit_natural_update_step(model, optimizer)  
        grit_neural_reprojection(model)
        
        optimizer.zero_grad()

    return {
        "task_loss": task_loss.item() * accumulation_steps,
        "curvature_loss": curvature_loss.item() if isinstance(curvature_loss, torch.Tensor) else curvature_loss,
        "reprojection_loss": reprojection_loss.item() if isinstance(reprojection_loss, torch.Tensor) else reprojection_loss,
        "total_loss": total_loss.item() * accumulation_steps
    }

class GRITLinear(nn.Module):
    def __init__(self, module: nn.Module, rank: int, k: int, alpha: float, ema: float=0.99, enable_grit: bool=True, eps: float=1e-4):
        super().__init__()

        self.module = module
        self.scale = alpha / rank
        self.k = k
        self.enable_grit = enable_grit
        self.rank = rank
        self.dtype = self.module.weight.dtype
        self.ema = ema
        self.eps = eps
        
        for param in module.parameters():
            param.requires_grad = False

        if enable_grit:
            in_dim = self.module.in_features
            out_dim = self.module.out_features

            self.lora_A = nn.Linear(in_dim, rank, bias=False).to(dtype=self.dtype, device=DEVICE)
            self.lora_B = nn.Linear(rank, out_dim, bias=False).to(dtype=self.dtype, device=DEVICE)

            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

            self.register_buffer("A_KFAC", torch.eye(in_dim, dtype=self.dtype, device=DEVICE) * self.eps)
            self.register_buffer("G_KFAC", torch.eye(out_dim, dtype=self.dtype, device=DEVICE) * self.eps)

            self.module.register_forward_hook(self._save_activation_forward)
            self.module.register_full_backward_hook(self._save_gradient_backward)

            self.stored_activations = None
            self.stored_gradients = None

            self.topk_A = None
            self.topk_G = None
            self.topk_lambda_A = None
            self.topk_lambda_G = None

    def _save_activation_forward(self, module, input, output):
        """Save activations for K-FAC statistics"""
        if len(input) > 0:
            self.stored_activations = input[0].detach().clone()

    def _save_gradient_backward(self, module, grad_input, grad_output):
        """Save gradients for K-FAC statistics"""
        if len(grad_output) > 0 and grad_output[0] is not None:
            self.stored_gradients = grad_output[0].detach().clone()

    def update_kfac_stats(self):
        """Update K-FAC statistics using EMA"""
        if self.stored_activations is None or self.stored_gradients is None:
            return
            
        # Reshape to (batch*seq, features)
        A = self.stored_activations.view(-1, self.lora_A.in_features)
        G = self.stored_gradients.view(-1, self.lora_B.out_features)

        with torch.no_grad():
            A_cov = torch.mm(A.T, A) / A.size(0)
            G_cov = torch.mm(G.T, G) / G.size(0)
            
            # EMA update
            self.A_KFAC.mul_(self.ema).add_(A_cov, alpha=(1 - self.ema))
            self.G_KFAC.mul_(self.ema).add_(G_cov, alpha=(1 - self.ema))

    def compute_topk_directions(self):
        """Compute top-k eigenvectors from K-FAC statistics"""
        with torch.no_grad():
            try:
                eigvals_A, eigvecs_A = torch.linalg.eigh(self.A_KFAC)
                eigvals_G, eigvecs_G = torch.linalg.eigh(self.G_KFAC)

                idx_A = torch.argsort(eigvals_A, descending=True)[:self.k]
                idx_G = torch.argsort(eigvals_G, descending=True)[:self.k]

                self.topk_A = eigvecs_A[:, idx_A].detach()
                self.topk_G = eigvecs_G[:, idx_G].detach()
                
                self.topk_lambda_A = eigvals_A[idx_A].detach().clamp(min=self.eps)
                self.topk_lambda_G = eigvals_G[idx_G].detach().clamp(min=self.eps)
                
            except Exception as e:
                print(f"Warning: Failed to compute top-k directions: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enable_grit:
            return self.module(x)
        
        base_out = self.module(x)
        lora_out = self.lora_B(self.lora_A(x)) * self.scale
        return base_out + lora_out


def grit_natural_update_step(model, optimizer, damping=1e-3):
    for module in model.modules():
        if isinstance(module, GRITLinear) and module.enable_grit:
            if module.lora_A.weight.grad is None or module.lora_B.weight.grad is None:
                continue
                
            if module.topk_A is None or module.topk_G is None:
                continue

            grad_A = module.lora_A.weight.grad
            grad_B = module.lora_B.weight.grad

            UA = module.topk_A.to(grad_A.device, dtype=grad_A.dtype)
            UG = module.topk_G.to(grad_B.device, dtype=grad_B.dtype)
            
            lambda_A = module.topk_lambda_A.to(grad_A.device, dtype=grad_A.dtype)
            lambda_G = module.topk_lambda_G.to(grad_B.device, dtype=grad_B.dtype)

            grad_A_proj = torch.mm(grad_A, UA)
            
            inv_lambda_A = 1.0 / (lambda_A + damping)
            grad_A_precond = grad_A_proj * inv_lambda_A.unsqueeze(0)
            
            natural_grad_A = torch.mm(grad_A_precond, UA.T)

            grad_B_proj = torch.mm(UG.T, grad_B)
            
            inv_lambda_G = 1.0 / (lambda_G + damping)
            grad_B_precond = grad_B_proj * inv_lambda_G.unsqueeze(1)
            
            natural_grad_B = torch.mm(UG, grad_B_precond)

            module.lora_A.weight.grad = natural_grad_A
            module.lora_B.weight.grad = natural_grad_B

    # Apply optimizer step
    optimizer.step()


def grit_neural_reprojection(model):
    """Project LoRA weights back to top-k subspace"""
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, GRITLinear) and module.enable_grit:
                if module.topk_A is not None and module.topk_G is not None:
                    device, dtype = module.lora_A.weight.device, module.lora_A.weight.dtype

                    UA = module.topk_A.to(device=device, dtype=dtype)
                    UG = module.topk_G.to(device=device, dtype=dtype)

                    proj_A = torch.mm(UA, UA.T)
                    module.lora_A.weight.data = torch.mm(module.lora_A.weight.data, proj_A)
  
                    proj_G = torch.mm(UG, UG.T)
                    module.lora_B.weight.data = torch.mm(proj_G, module.lora_B.weight.data)


def apply_grit_to_model(model: nn.Module, target_modules=None, **grit_kwargs):
    """Replace specific nn.Linear modules in model with GRITLinear wrappers"""
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"] 

    layers_to_replace = []
    for name, module in model.named_modules():
        if any(name.endswith(t) for t in target_modules) and isinstance(module, nn.Linear):
            if module.in_features < 32 or module.out_features < 32:
                continue
            if "embed" in name.lower() or "lm_head" in name.lower():
                continue
            layers_to_replace.append((name, module))

    print(f"Found {len(layers_to_replace)} candidate layers to replace with GRIT")

    name2module = dict(model.named_modules())
    replaced_count = 0

    for name, orig_module in layers_to_replace:
        try:
            parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = name2module[parent_name] if parent_name else model

            grit_layer = GRITLinear(orig_module, **grit_kwargs)

            try:
                module_device = next(orig_module.parameters()).device
            except StopIteration:
                module_device = next(model.parameters()).device
            grit_layer.to(module_device)

            setattr(parent, attr_name, grit_layer)
            replaced_count += 1
            print(f"Replaced {name} -> GRIT (in:{orig_module.in_features}, out:{orig_module.out_features})")
        except Exception as e:
            print(f"Failed to replace {name}: {e}")

    print(f"Successfully replaced {replaced_count} layers with GRIT")
    return model


def get_grit_parameters(model):
    """Get all GRIT trainable parameters"""
    grit_params = []
    for module in model.modules():
        if isinstance(module, GRITLinear) and module.enable_grit:
            grit_params.append(module.lora_A.weight)
            grit_params.append(module.lora_B.weight)
    return grit_params


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
                total_loss += outputs.loss
                num_batches += 1
                
            except Exception as e:
                print(f"Evaluation error: {e}")
                continue
    
    return total_loss / max(1, num_batches)

class OKQACollator:
    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        texts = [f"Question: {ex['question']}\nAnswer: {ex['answers'][0] if isinstance(ex['answers'], list) else ex['answer']}" 
                 for ex in batch]

        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        labels = encodings["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        encodings["labels"] = labels
        return encodings


class UniversalVLMGritCollator:
    def __init__(self, processor, max_length: int = 128, fallback_image_size=(224, 224)):
        self.processor = processor
        self.max_length = max_length
        self.fallback_image_size = fallback_image_size

    def _robust_load(self, img_field):
        try:
            if isinstance(img_field, Image.Image):
                return img_field.convert("RGB")

            if isinstance(img_field, (bytes, bytearray)):
                return Image.open(BytesIO(img_field)).convert("RGB")

            if isinstance(img_field, dict):
                for k in ("image", "image_url", "url", "file_name", "path", "bytes", "coco_url"):
                    if k in img_field and img_field[k] is not None:
                        return self._robust_load(img_field[k])

            if isinstance(img_field, str):
                s = img_field.strip()
                if s.startswith("http://") or s.startswith("https://"):
                    try:
                        from transformers.image_utils import load_image as hf_load_image
                        pil = hf_load_image(s)
                        return pil.convert("RGB")
                    except Exception:
                        resp = requests.get(s, timeout=10)
                        resp.raise_for_status()
                        return Image.open(BytesIO(resp.content)).convert("RGB")
                else:
                    return Image.open(s).convert("RGB")

            raise ValueError(f"Unsupported image type: {type(img_field)}")

        except Exception as e:
            print(f"[Warning] Failed to load image ({e}); using blank fallback.")
            return Image.new("RGB", self.fallback_image_size, color=(127, 127, 127))

    def __call__(self, batch):
        images, prompts, labels = [], [], []

        for ex in batch:
            if "messages" in ex:
                msg = ex["messages"]
            else:
                q = ex.get("question") or ex.get("caption") or ""
                msg = [{"role": "user",
                        "content": [{"type": "image"},
                                    {"type": "text", "text": q}]}]

            prompt = self.processor.apply_chat_template(msg, add_generation_prompt=True)
            prompts.append(prompt)

            img_field = None
            for key in ("image", "image_url", "img", "file_name",
                        "filepath", "path", "coco_url"):
                if key in ex and ex[key] is not None:
                    img_field = ex[key]
                    break
            if img_field is None:
                img_field = ex
            images.append(self._robust_load(img_field))

            ans = None
            if "answers" in ex and isinstance(ex["answers"], list) and len(ex["answers"]) > 0:
                ans = ex["answers"][0]
            elif "answer" in ex:
                ans = ex["answer"]
            labels.append(ans)

        enc = self.processor(
            text=prompts,
            images=images,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        if any(lab is not None for lab in labels):
            label_texts = [lab if lab is not None else "" for lab in labels]
            label_enc = self.processor(
                text=label_texts,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            ).input_ids
            label_enc[label_enc == self.processor.tokenizer.pad_token_id] = -100
            enc["labels"] = label_enc

        return enc


def load_model(model_id: str, tdtype=torch.bfloat16):
    """Load model and processor based on model ID"""
    if "paligemma" in model_id:
        quantization_config = TorchAoConfig("int4_weight_only", group_size=128)
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=tdtype, quantization_config=quantization_config, device_map="auto"
        ).to(DEVICE)
        processor = AutoProcessor.from_pretrained(model_id)
    elif "llama" in model_id:
        model = MllamaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=tdtype, device_map="auto"
        ).to(DEVICE)
        processor = AutoProcessor.from_pretrained(model_id)
    else:
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForVision2Seq.from_pretrained(
            model_id, 
            torch_dtype=tdtype, 
            _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager"
        ).to(DEVICE)
    return model, processor


def save_grit_weights(model, filename):
    """Save only GRIT weights"""
    grit_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, GRITLinear) and module.enable_grit:
            grit_state_dict[f"{name}.lora_A.weight"] = module.lora_A.weight.detach().cpu().clone()
            grit_state_dict[f"{name}.lora_B.weight"] = module.lora_B.weight.detach().cpu().clone()

    torch.save(grit_state_dict, filename)
    print(f"Saved GRIT weights to {filename} ({len(grit_state_dict)} tensors)")