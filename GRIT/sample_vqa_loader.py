from torch.utils.data import Dataset, DataLoader
from PIL import Image

class SimpleVQADataset(Dataset):
    def __init__(self, split='train', max_samples=100):
        self.split = split
        self.max_samples = max_samples
        self.create_synthetic_data()
        print(f"Created {len(self.data)} synthetic VQA samples")

    def create_synthetic_data(self):
        questions = [
            "What color is this object?",
            "How many items are in the image?",
            "What is the main subject of this image?",
            "Is this indoors or outdoors?",
            "What shape is the object?",
        ]
        answers = [
            "red", "blue", "green", "yellow", "white", "black",
            "one", "two", "three", "four", "five",
            "person", "animal", "car", "building", "tree",
            "indoors", "outdoors",
            "round", "square", "rectangular", "triangular"
        ]
        self.data = []
        import random
        for i in range(self.max_samples):
            color = random.choice([(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,255,255)])
            image = Image.new('RGB', (224,224), color=color)
            question = random.choice(questions)
            answer = random.choice(answers)
            self.data.append({'image': image, 'question': question, 'answer': answer})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
class SimpleVQACollator:
    """
    Collator for the synthetic VQA dataset.
    Produces:
      - input_ids, attention_mask, labels
    Labels mask prompt tokens with -100 so loss ignores them.
    """
    def __init__(self, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, batch):
        # batch: list of dicts {'image': PIL.Image, 'question': str, 'answer': str}
        questions = [item['question'] for item in batch]
        answers = [item['answer'] for item in batch]

        prompts = [f"Question: {q}\nAnswer:" for q in questions]
        full_texts = [p + " " + a for p, a in zip(prompts, answers)]

        # tokenize prompts (inputs to model) and full texts (for labels)
        # We tokenize full_texts as the input to the causal LM (prompt + answer),
        # then set label tokens corresponding to the prompt to -100.
        full_inputs = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # create labels from the full inputs
        labels = full_inputs["input_ids"].clone()

        # for each example, find how many tokens the prompt occupies and set them to -100
        for i, prompt in enumerate(prompts):
            # tokenize prompt only to get token count
            prompt_tokens = self.tokenizer(prompt, truncation=True, max_length=self.max_length, return_tensors="pt")
            prompt_len = prompt_tokens["input_ids"].size(1)
            # if prompt longer than sequence length, we skip masking (edge case)
            seq_len = labels.size(1)
            if prompt_len >= seq_len:
                # mask everything except the last token (so training still has something)
                labels[i, :] = -100
            else:
                labels[i, :prompt_len] = -100

        return {
            "input_ids": full_inputs["input_ids"],
            "attention_mask": full_inputs["attention_mask"],
            "labels": labels,
        }
