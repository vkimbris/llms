import torch

from torch.utils.data import Dataset


class LanguageModelingDataset(Dataset):

    def __init__(self, tokens: torch.Tensor) -> None:
        super().__init__()

        self.tokens = tokens

        self.inputs = tokens[:, :-1]
        self.labels = tokens[:, 1:]

    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, index):
        return {"inputs": self.inputs[index], "labels": self.labels[index]}
