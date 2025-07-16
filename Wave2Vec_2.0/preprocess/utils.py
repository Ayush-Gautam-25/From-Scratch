from preprocess.preprocessor import AudioPreprocessor
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from preprocess.dataset import LibriSpeechDataset
import os
import sys
from typing import List, Dict, Any
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    audio_tensors = [item["audio"] if isinstance(item, dict) else item for item in batch]
    audio_tensors = [a if a.dim() == 1 else a.squeeze(0) for a in audio_tensors]
    padded = pad_sequence(audio_tensors, batch_first=True)  # (B, T)
    texts = [item["text"] for item in batch]
    return {"audio": padded, "text": texts}

def load_data(dtype: str = "train", batch_size: int = 128, describe=True) -> DataLoader:
    """
    Loads LibriSpeech data with preprocessing and batching.

    Args:
        dtype (str): One of 'train', 'val', 'test'
        batch_size (int): Batch size for DataLoader

    Returns:
        torch.utils.data.DataLoader
    """

    split_map = {
        "train": "train-clean-100",
        "val": "dev-clean",
        "test": "test-clean"
    }

    if dtype not in split_map:
        raise ValueError(f"Invalid dtype '{dtype}'. Must be one of {list(split_map.keys())}.")

    root = f"../data/librispeech/LibriSpeech/{split_map[dtype]}"
    pre = AudioPreprocessor(target_sr=16000, normalize=True, max_duration=5.0)
    dataset = LibriSpeechDataset(root_dir=root, preprocessor=pre, ext=".flac")

    if len(dataset) == 0:
        raise RuntimeError(f"Dataset is empty at path: {root}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(dtype == "train"), collate_fn=collate_fn)

    print(f"✅ {dtype} data loaded...")
    if describe:
        print(f"Number of samples in {dtype} data: {len(dataset)}")
        sample = dataset[0]
        print(f"Audio shape (first sample): {sample['audio'].shape}")
        print(f"DataLoader info: batch_size={batch_size}, shuffle={dtype == 'train'}, num_batches={len(loader)}")
    return loader
