import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

from grit_tools import apply_grit_to_model, get_grit_parameters, save_grit_weights, evaluate, reproject_grit_modules
from utils import count_parameters, freeze_non_grit_parameters

# ------------------------------
# Multimodal Collator for OK-VQA (bfloat16 compatible)
# ------------------------------
import requests
from io import BytesIO
from PIL import Image
from transformers.image_utils import load_image as hf_load_image

class OKVQAGritCollator:
    def __init__(self, processor, max_length=128, fallback_image_size=(224, 224)):
        self.processor = processor
        self.max_length = max_length
        self.fallback_image_size = fallback_image_size

    def _robust_load(self, img_field):
        """
        Accepts many forms:
        - PIL.Image.Image -> return .convert("RGB")
        - str (URL or local path) -> load
        - bytes/bytearray -> BytesIO -> Image
        - dict containing 'url'/'image_url'/'file_name'/'path'/'bytes' -> recurse
        """
        try:
            # already a PIL image
            if isinstance(img_field, Image.Image):
                return img_field.convert("RGB")

            # bytes or bytearray
            if isinstance(img_field, (bytes, bytearray)):
                return Image.open(BytesIO(img_field)).convert("RGB")

            # dict-like container
            if isinstance(img_field, dict):
                # try common keys
                for k in ("image", "image_url", "url", "file_name", "path", "bytes", "coco_url"):
                    if k in img_field and img_field[k] is not None:
                        return self._robust_load(img_field[k])
                # sometimes datasets return {"bytes": b"..."} etc.
                # fallthrough to error

            # string: could be an http URL or local path
            if isinstance(img_field, str):
                s = img_field.strip()
                if s.startswith("http://") or s.startswith("https://"):
                    # try huggingface helper first
                    try:
                        pil = hf_load_image(s)  # returns PIL.Image
                        return pil.convert("RGB")
                    except Exception:
                        # fallback to requests
                        resp = requests.get(s, timeout=10)
                        resp.raise_for_status()
                        return Image.open(BytesIO(resp.content)).convert("RGB")
                else:
                    # local path - let PIL handle file path
                    return Image.open(s).convert("RGB")

            # If we get here, unsupported type - throw to fallback
            raise ValueError(f"Unsupported image field type: {type(img_field)}")
        except Exception as e:
            # log a short warning and return a tiny blank RGB image (won't crash training)
            print(f"[Warning] failed to load image ({e}); using blank fallback.")
            return Image.new("RGB", self.fallback_image_size, color=(127, 127, 127))

    def __call__(self, batch):
        input_texts = []
        batch_images = []

        for ex in batch:
            # Question and first answer
            question = ex.get("question", "")
            answer = ex.get("answers")
            if isinstance(answer, list) and len(answer) > 0:
                answer = answer[0]
            else:
                answer = ex.get("answer", "N/A")

            message = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
            ]
            input_texts.append((message, answer))

            # Robustly find the image field
            img_field = None
            # Common field names used by different datasets
            for key in ("image", "image_url", "image_id", "img", "img_url", "file_name", "filepath", "coco_url"):
                if key in ex and ex[key] is not None:
                    img_field = ex[key]
                    break
            # Some datasets put the image under 'image' as a dict with 'path' or 'url'
            if img_field is None and "image" in ex:
                img_field = ex["image"]

            # As last resort, try ex itself (some HF dataset rows are Image objects)
            if img_field is None:
                img_field = ex

            img = self._robust_load(img_field)
            batch_images.append(img)

        # Build prompts
        prompts = [self.processor.apply_chat_template(msg, add_generation_prompt=True) for msg, _ in input_texts]
        answers = [ans for _, ans in input_texts]

        # Encode inputs (processor expects list of PIL images)
        encodings = self.processor(
            text=prompts,
            images=batch_images,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        # Labels for causal LM (direct tokenization)
        labels_enc = self.processor(
            text=answers,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        labels = labels_enc.input_ids
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        encodings["labels"] = labels

        return encodings


# ------------------------------
# Training Loop
# ------------------------------
def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    model_name = "HuggingFaceTB/SmolVLM-Base"
    grit_config = {"r": 4, "alpha": 16, "k": 2, "dropout_p": 0.1, "ema": 0.95}
    train_config = {
        "batch_size": 1,
        "learning_rate": 2e-4,
        "num_epochs": 3,
        "max_samples": 100,
        "gradient_accumulation_steps": 2,
        "eval_steps": 20,
    }

    print("Loading processor and model...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
    ).to(DEVICE)

    print("Applying GRIT...")
    model = apply_grit_to_model(model, **grit_config)
    freeze_non_grit_parameters(model)

    total_params, trainable_params = count_parameters(model)
    print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

    grit_params = get_grit_parameters(model)
    if not grit_params:
        print("❌ No GRIT params found!")
        return

    print("Loading OK-VQA dataset...")
    dataset = load_dataset("lmms-lab/OK-VQA")
    train_data = dataset["val2014"].select(range(train_config["max_samples"] // 4))

    collator = OKVQAGritCollator(processor)
    train_loader = DataLoader(train_data, batch_size=train_config["batch_size"], shuffle=True, collate_fn=collator)

    optimizer = torch.optim.AdamW(grit_params, lr=train_config["learning_rate"])
    total_steps = len(train_loader) * train_config["num_epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    print("🚦 Training...")
    model.train()
    global_step = 0
    for epoch in range(train_config["num_epochs"]):
        print(f"Epoch {epoch+1}/{train_config['num_epochs']}")
        progress = tqdm(train_loader)
        for batch in progress:
            # Move tensors to device
            batch = {k: v.to(DEVICE) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            outputs = model(**batch)
            loss = outputs.loss / train_config["gradient_accumulation_steps"]
            loss.backward()

            if (global_step + 1) % train_config["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(grit_params, 1.0)
                optimizer.step()
                scheduler.step()
                reproject_grit_modules(model)
                optimizer.zero_grad()

            progress.set_postfix({"loss": loss.item() * train_config["gradient_accumulation_steps"]})
            global_step += 1

            if global_step % train_config["eval_steps"] == 0:
                val_loss = evaluate(model, train_loader, DEVICE)
                print(f"Step {global_step} - Val loss: {val_loss:.4f}")
                model.train()

    print("🎉 Training complete!")
    save_grit_weights(model, "okvqa_smolvlm_grit.pt")


if __name__ == "__main__":
    main()
