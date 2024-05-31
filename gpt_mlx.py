import mlx.core as mx
import mlx.nn as nn

from dataclasses import dataclass


def generate_attention_mask(seq_length: int) -> mx.array:
    return mx.triu(mx.full((seq_length, seq_length), float('-inf')), k=1)


class Attention(nn.Module):

    def __init__(self, input_dim: int, query_dim: int, value_dim: int) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.query_dim = query_dim
        self.value_dim = value_dim

        self.query_weights = nn.Linear(self.input_dim, self.query_dim)
        self.value_weights = nn.Linear(self.input_dim, self.value_dim)
        
        self.key_weights = nn.Linear(self.input_dim, self.query_dim)

    def __call__(self, x, attention_mask):
        q, k, v = self.query_weights(x), self.key_weights(x), self.value_weights(x)

        attentions = (q @ k.transpose(0, 2, 1)) / self.query_dim ** 2
        attentions = attentions + attention_mask
        
        attentions = mx.softmax(attentions, axis=-1)
        
        return attentions, attentions @ v
    

class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads: int, input_dim: int, query_dim: int, value_dim: int) -> None:
        super().__init__()
        
        self.heads = [
            Attention(input_dim, query_dim, value_dim) for _ in range(n_heads)
        ]

        self.output_weights = nn.Linear(n_heads * value_dim, input_dim)

    def __call__(self, x, attention_maks):
        attentions, outputs = [], []

        for head in self.heads:
            attention, output = head(x, attention_maks)

            attentions.append(attention)
            outputs.append(output)
        
        outputs = mx.concatenate(outputs, axis=-1)
        attentions = mx.concatenate(attentions)

        return attentions, self.output_weights(outputs)
    

class FeedForward(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def __call__(self, x):
        return self.network(x)
    

class DecoderBlock(nn.Module):

    def __init__(self, 
                 multihead_attention: MultiHeadAttention,
                 feed_forward: FeedForward,
                 first_layer_norm: nn.LayerNorm,
                 second_layer_norm: nn.LayerNorm) -> None:
        
        super().__init__()

        self.multihead_attention = multihead_attention
        self.feed_forward = feed_forward

        self.first_layer_norm = first_layer_norm
        self.second_layer_norm = second_layer_norm

    def __call__(self, x, attention_mask):
        _, y = self.multihead_attention(x, attention_mask)

        y = y + x

        y = self.first_layer_norm(y)

        y = self.feed_forward(y) + y

        y = self.second_layer_norm(y)

        return y
    

@dataclass
class GPTParams:
    vocab_size: int
    context_size: int
    
    input_dim: int
    query_dim: int
    value_dim: int

    feed_forward_hidden_dim: int

    n_heads: int = 8
    n_decoder_blocks: int = 12

    device: str = "cpu"
    

class GPT(nn.Module):

    def __init__(self, params: GPTParams) -> None:
        super().__init__()

        self.params = params

        self.embeddings = nn.Embedding(
            num_embeddings=params.vocab_size, dims=params.input_dim
        )

        self.positional_embeddings = mx.random.normal((params.context_size, params.input_dim), scale=0.02)

        self.decoders_blocks = [
            DecoderBlock(
                multihead_attention=MultiHeadAttention(
                    n_heads=params.n_heads,
                    input_dim=params.input_dim,
                    query_dim=params.query_dim,
                    value_dim=params.value_dim,
                ),

                feed_forward=FeedForward(
                    input_dim=params.input_dim,
                    hidden_dim=params.feed_forward_hidden_dim,
                    output_dim=params.input_dim,
                ),

                first_layer_norm=nn.LayerNorm(
                    dims=params.input_dim
                ),

                second_layer_norm=nn.LayerNorm(
                    dims=params.input_dim
                ),
            )

            for _ in range(params.n_decoder_blocks)
        ]

        self.output_layer = nn.Linear(input_dims=params.input_dim, output_dims=params.vocab_size)

    def __call__(self, x, attention_mask):
        x = self.embeddings(x) + self.positional_embeddings[:x.shape[1]]

        for decoder in self.decoders_blocks:
            x = decoder(x, attention_mask)

        return self.output_layer(x)
