import numpy as np
import mlx.core as mx
import mlx.data as dx


class LanguageModelingDataset:

    def __init__(self, tokens: mx.array) -> None:
        self.tokens = tokens

        self.inputs = tokens[:, :-1]
        self.labels = tokens[:, 1:]

        elements = []
        for _input, _label in zip(self.inputs, self.labels):
            elements.append({
                "inputs": _input,
                "labels": _label
            })

    def __len__(self):
        return len(self.inputs)

    def to_dataloader(self, batch_size: int = 32):
        perm = mx.array(np.random.permutation(self.labels.shape[0]))
        for s in range(0, self.labels.shape[0], batch_size):
            ids = perm[s : s + batch_size]
            yield self.inputs[ids], self.labels[ids]