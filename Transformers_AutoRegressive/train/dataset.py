import torch
from torch.utils.data import Dataset

class ModelDataset(Dataset):
    def __init__(self, data_tensor: torch.Tensor, block_size: int):
        self.data = data_tensor
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int):
        # the whole corpus/dataset is taken as a single stream of data
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        return x, y
