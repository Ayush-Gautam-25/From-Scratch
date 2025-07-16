import os
import torch
from datasets import load_dataset
from typing import List
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from tokenizer.bpe_tokenizer import BPE


SAVE_DIR = "../data/wikitext-103"
VOCAB_SIZE = 2000

def save_tokenized_split(split_name: str, text_list:List[str], bpe: BPE):
    print(f"\nTokenizing {split_name} split...")
    all_ids = []
    i=0
    for line in tqdm(text_list, desc=f"Processing {split_name}", unit="line"):
        if line.strip() == "":
            continue
        ids = bpe.encode_text(line.strip())
        all_ids.extend(ids)
    print(f"Total tokens in {split_name} split: {len(all_ids)}")
    tensor = torch.tensor(all_ids, dtype=torch.long)
    torch.save(tensor, os.path.join(SAVE_DIR, f"{split_name}.pt"))

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print('Loading WikiText-103 Dataset...')
    dataset = load_dataset("wikitext", "wikitext-103-v1")

    bpe = BPE(vocab_size = VOCAB_SIZE)
    bpe_path = os.path.join(SAVE_DIR, "bpe_tokenizer")

    if not os.path.exists(os.path.join(bpe_path, "token_to_id.json")):
        with open(os.path.join(SAVE_DIR, "train_raw.txt"), "w", encoding="utf-8") as f:
            for line in dataset["train"]["text"][:10000]:
                line = line.strip()
                if line:
                    f.write(line + "\n")

        print("Training BPE tokenizer on WikiText-103 train split...")
        bpe.train(os.path.join(SAVE_DIR, "train_raw.txt"), show_merges=False)
        bpe.save(bpe_path)
    else:
        print("Loading existing BPE tokenizer...")
        bpe.load(bpe_path)

    save_tokenized_split("train", dataset["train"]["text"][:1000000], bpe) # done for first million
    save_tokenized_split("val", dataset["validation"]["text"], bpe)
    save_tokenized_split("test", dataset["test"]["text"], bpe)

    print("\nSuccess. Tokenized data saved to:", SAVE_DIR)

if __name__=='__main__':
    main()
