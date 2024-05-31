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

        self.buffer = dx.buffer_from_vector(elements)

    def to_dataloader(self, batch_size: int = 32):
        return self.buffer.shuffle().to_stream().batch(batch_size)