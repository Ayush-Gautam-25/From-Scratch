import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from preprocess.utils import load_data
from models.feature_encoder import CNNFeatureEncoder
from models.mask_module import MaskingModule
from models.transformer_encoder import TransformerEncoder
from models.quantization_vector import VectorQuantizer
from loss.contrastive_loss import ContrastiveLoss
from tqdm import tqdm
import torch.nn.functional as F

LATENT_DIM = 512
NUM_EPOCHS = 10

train_loader = load_data("test", batch_size=10)
val_loader = load_data("val", batch_size=10)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda"
cnn = CNNFeatureEncoder(conv_dim=LATENT_DIM).to(device)
masker = MaskingModule(mask_prob=0.35, mask_span=10, emb_dim=LATENT_DIM).to(device)
transformer = TransformerEncoder(d_model=LATENT_DIM, use_moe=True).to(device)
quantizer = VectorQuantizer(
        codebook_size=320,     # number of discrete tokens (K)
        codebook_dim=512,      # same as CNN output / transformer dim (D)
        temperature=0.03,
        gumbel=True
    ).to(device)
all_params = list(cnn.parameters()) + \
             list(quantizer.parameters()) + \
             list(masker.parameters()) + \
             list(transformer.parameters())

optimizer = torch.optim.AdamW(all_params, lr=1e-4)

loss_fn = ContrastiveLoss()

for epoch in range(1, NUM_EPOCHS + 1):
    transformer.train(); cnn.train(); quantizer.train(); masker.train()
    running_loss = 0.0

    loop = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    for batch in loop:
        audio = batch["audio"][:, :9000].to(device)            # [B, T]
        z = cnn(audio)                                # [B, T', D]
        q_z, _, _ = quantizer(z)                      # quantized target
        z_masked, mask = masker(z)                    # masking
        # print("Masked positions:", mask.sum().item())  # Should be > 0

        c = transformer(z_masked)                     # context
        # print("context:", c.shape)
        # print("quantized target:", q_z.shape)
        # print("mask sum:", mask.sum().item())
        # print("masked context:", c[mask].shape)
        # print("masked quantized:", q_z[mask].shape)

        loss = loss_fn(c, q_z, mask)

        optimizer.zero_grad()
        # print("Loss before backward:", loss.item())  # If this prints `nan`, debug further below

        # with torch.autograd.detect_anomaly():
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch} Loss: {running_loss / len(train_loader):.4f}")

    # =======================
    #       VALIDATION
    # =======================
    transformer.eval(); cnn.eval(); quantizer.eval(); masker.eval()
    running_val_loss = 0.0
    running_val_cossim = 0.0
    total_val_masked = 0

    with torch.no_grad():
        val_loop = tqdm(val_loader, desc=f"[Val] Epoch {epoch}", leave=False)
        for batch in val_loop:
            audio = batch["audio"][:, :9000].to(device)

            z = cnn(audio)
            q_z, _, _ = quantizer(z)
            z_masked, mask = masker(z)
            c = transformer(z_masked)

            val_loss = loss_fn(c, q_z, mask)

            context_masked = F.normalize(c[mask], dim=-1)
            targets_masked = F.normalize(q_z[mask], dim=-1)
            val_cos_sim = (context_masked * targets_masked).sum(dim=-1).mean().item()

            running_val_loss += val_loss.item()
            running_val_cossim += val_cos_sim
            total_val_masked += mask.sum().item()

            val_loop.set_postfix(loss=val_loss.item(), cos_sim=f"{val_cos_sim:.3f}", masked=mask.sum().item())

    avg_val_loss = running_val_loss / len(val_loader)
    avg_val_cos = running_val_cossim / len(val_loader)

    print(f"[✓] Epoch {epoch} Val   Loss: {avg_val_loss:.4f} | CosSim: {avg_val_cos:.4f} | Avg Masked: {total_val_masked // len(val_loader)}")

