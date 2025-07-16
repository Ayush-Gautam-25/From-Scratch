import re
from collections import defaultdict
from typing import List, DefaultDict, Tuple, Dict
import json
import os
from tqdm import trange  

class BPE:
    def __init__(self, vocab_size: int =5000) -> None:
        self.vocab_size = vocab_size
        self.merges = []
        self.vocab = {}
        self.token_to_id = {}
        self.id_to_token = {}

    def _get_vocab(self, corpus: List[List[str]]) -> dict:
        vocab = defaultdict(int)
        for word in corpus:
            word = tuple(word) + ('</w>',) # this marks end of a word
            vocab[word] += 1
        return vocab

    
    def _get_stats(self, vocab: DefaultDict[Tuple[str, ...], int]) -> DefaultDict[Tuple[str, ...], int]:
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            for i in range(len(word)-1):
                pairs[(word[i], word[i+1])] += freq
        return pairs
    
    def _get_bpe_corpus(self, text_path: str, lower_case_all=False) -> List[List[str]]:
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()

        if lower_case_all:
            # Lowercase and split into words
            words = text.lower().split()
        else:
            words = text.split()
        
        # Each word becomes a list of characters
        corpus = [list(word) for word in words]
        return corpus


    def _merge_vocab(self, pair: Tuple, vocab: DefaultDict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
        new_vocab = {}
        pattern = re.escape(" ".join(pair))
        replacement = "".join(pair)
        for word, freq in vocab.items():
            word_str = " ".join(word)
            new_word = tuple(re.sub(pattern, replacement, word_str).split())
            new_vocab[new_word] = freq
        return new_vocab
    
    def _build_token_ids(self,) -> None:
        special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]

        self.token_to_id = {tok: i for i, tok in enumerate(special_tokens)}
        self.id_to_token = {i: tok for i, tok in enumerate(special_tokens)}

        bpe_tokens = set()
        for word in self.vocab:
            for token in word:
                bpe_tokens.add(token)
        bpe_tokens = sorted(bpe_tokens)
        next_id = len(self.token_to_id)

        for token in bpe_tokens:
            self.token_to_id[token] = next_id
            self.id_to_token[next_id] = token
            next_id += 1


    def train(self, text_path: str, lower_case_all: bool = False, show_merges=False):
        corpus = self._get_bpe_corpus(text_path=text_path, lower_case_all=lower_case_all)

        vocab = self._get_vocab(corpus)
        self.vocab = vocab

        for i in trange(self.vocab_size, desc="Training BPE merges"):
            pairs = self._get_stats(vocab=vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = self._merge_vocab(best, vocab)
            self.merges.append(best)
            if show_merges:
                print(f"Merge {i+1}: {best}")
        

        self.vocab = vocab  # final vocab
        self._build_token_ids()

    def encode_ids(self, word: str) -> List[int]:
        tokens = self.encode(word)
        return [self.token_to_id[token] for token in tokens]

    def decode_ids(self, ids: List[int]) -> str:
        tokens = [self.id_to_token[i] for i in ids]
        return self.decode(tokens)


    def encode(self, word: str) -> str:
        word = list(word) + ["</w>"]
        while True:
            pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
            mergeable = [pair for pair in pairs if pair in self.merges]
            if not mergeable:
                break
            # Pick first match in merge list order
            for merge in self.merges:
                if merge in pairs:
                    i = pairs.index(merge)
                    word = word[:i] + ["".join(merge)] + word[i+2:]
                    break
        return word
    
    def decode(self, tokens: str) -> str:
        return "".join(token.replace("</w>", "") for token in tokens)
    
    def encode_text(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ids = []
        # Adding beginning of sequence token [BOS]
        if add_special_tokens and '[BOS]' in self.token_to_id:
            ids.append(self.token_to_id['[BOS]'])

        words = text.strip().split()
        for word in words:
            try:
                token_ids = self.encode_ids(word)
            except KeyError:
                # fallback to [UNK] for OOV tokens
                token_ids = [self.token_to_id.get("[UNK]", 1)]
            ids.extend(token_ids)

        # Adding end of sequence token [EOS]
        if add_special_tokens and "[EOS]" in self.token_to_id:
            ids.append(self.token_to_id["[EOS]"])

        return ids
    
    def decode_text(self, ids: List[int]) -> str:
        ignore_tokens = {"[BOS]", "[EOS]", "[PAD]"}
        tokens = [self.id_to_token.get(i, "[UNK]") for i in ids]
        tokens = [t for t in tokens if t not in ignore_tokens]
        text = "".join(tokens)
        words = text.replace("</w>", " ").strip()
        return words

    def pad_batch(self, batch: List[List[int]], pad_id: int = 0, max_len: int = None) -> List[List[int]]:
        if not batch:
            return []

        if max_len is None:
            max_len = max(len(seq) for seq in batch)

        padded_batch = [
            seq + [pad_id] * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len]
            for seq in batch
        ]
        return padded_batch

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "merges.json"), "w", encoding="utf-8") as f:
            json.dump(self.merges, f)
        with open(os.path.join(path, "token_to_id.json"), "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f)
        with open(os.path.join(path, "id_to_token.json"), "w", encoding="utf-8") as f:
            json.dump(self.id_to_token, f)

    def load(self, path: str) -> None:
        with open(os.path.join(path, "merges.json"), "r", encoding="utf-8") as f:
            self.merges = [tuple(pair) for pair in json.load(f)]
        with open(os.path.join(path, "token_to_id.json"), "r", encoding="utf-8") as f:
            self.token_to_id = json.load(f)
        with open(os.path.join(path, "id_to_token.json"), "r", encoding="utf-8") as f:
            self.id_to_token = {int(k): v for k, v in json.load(f).items()}

    
if __name__=='__main__':
    bpe = BPE(vocab_size=100)
    bpe.train("../data/tinyshakespeare/tinyshakespeare.txt")

    # bpe.encode_text("to be or not to be")


    # encoded = bpe.encode("shakespeare")
    # print("Encoded:", encoded)

    # decoded = bpe.decode(encoded)
    # print("Decoded:", decoded)

    # ids = bpe.encode_ids("shakespeare")
    # print("Token IDs:", ids)

    # decoded = bpe.decode_ids(ids)
    # print("Decoded:", decoded)


    # words = ["love", "hate", "heart", "death", "life", "you", "thou", "dearest"]

    # for word in words:
    #     enc = bpe.encode(word)
    #     dec = bpe.decode(enc)
    #     assert dec == word, f"Failed on {word}"
    #     print(f"{word} → {enc} → {dec}")

    # text = "to be or not to be"
    # ids = bpe.encode_text(text)
    # decoded = bpe.decode_text(ids)
    # print("Original:", text)
    # print("Token IDs:", ids)
    # print("Decoded:", decoded)


    batch = [bpe.encode_text(s) for s in ["to be", "or not", "to be or not to be"]]
    padded = bpe.pad_batch(batch, pad_id=bpe.token_to_id["[PAD]"]) # since id for [PAD] is 0, the padding in the sequence will be of 0
    print("Padded:", padded)

    # Save and load
    bpe.save("bpe_tokenizer")
    bpe2 = BPE()
    bpe2.load("bpe_tokenizer")

    assert bpe2.token_to_id == bpe.token_to_id
    assert bpe2.decode_text(padded[0]) == bpe.decode_text(padded[0])

    