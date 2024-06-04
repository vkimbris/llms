import os
import torch
import json
import torch.nn as nn

import torch.nn.functional as F

from dataclasses import dataclass


def generate_window_attention_mask(seq_length: int, window_size: int) -> torch.Tensor:
    upper = torch.tril(torch.ones(window_size, seq_length), diagonal=0)

    lower = torch.tensor([1] * window_size + [0] * (seq_length - window_size))
    
    if seq_length > window_size:
        lower = torch.vstack([lower.roll(shifts=k) for k in range(1, seq_length - window_size + 1)])

        return torch.cat([upper, lower])
    
    elif seq_length < window_size:
        return torch.tril(torch.ones(seq_length, seq_length), diagonal=0)

    else:
        return upper
    

def generate_attention_mask(seq_length: int, device: str = "cpu") -> torch.Tensor:
    return torch.triu(torch.full((seq_length, seq_length), float('-inf')), diagonal=1).to(device)


class Attention(nn.Module):

    def __init__(self, input_dim: int, query_dim: int, value_dim: int, device: str = "cpu") -> None:
        super().__init__()

        self.input_dim = input_dim
        self.query_dim = query_dim
        self.value_dim = value_dim

        self.query_weights = nn.Linear(self.input_dim, self.query_dim, device=device)
        self.value_weights = nn.Linear(self.input_dim, self.value_dim, device=device)
        
        self.key_weights = nn.Linear(self.input_dim, self.query_dim, device=device)

    def forward(self, x, attention_mask):
        q, k, v = self.query_weights(x), self.key_weights(x), self.value_weights(x)

        attentions = (q @ k.transpose(1, 2)) / self.query_dim ** 2
        attentions = attentions + attention_mask
        
        attentions = F.softmax(attentions, dim=-1)
        
        return attentions, attentions @ v

    

class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads: int, input_dim: int, query_dim: int, value_dim: int, device: str = "cpu") -> None:
        super().__init__()
        
        self.heads = [
            Attention(input_dim, query_dim, value_dim, device) for _ in range(n_heads)
        ]

        self.output_weights = nn.Linear(n_heads * value_dim, input_dim, device=device)

    def forward(self, x, attention_maks):
        attentions, outputs = [], []

        for head in self.heads:
            attention, output = head(x, attention_maks)

            attentions.append(attention)
            outputs.append(output)
        
        outputs = torch.cat(outputs, dim=2)
        attentions = torch.cat(attentions)

        return attentions, self.output_weights(outputs)


class FeedForward(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, device: str = "cpu") -> None:
        
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, device=device),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim, device=device)
        )

    def forward(self, x):
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

    def forward(self, x, attention_mask):
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
            num_embeddings=params.vocab_size, embedding_dim=params.input_dim, device=params.device
        )

        self.positional_embeddings = nn.Parameter(
            data=torch.empty(params.context_size, params.input_dim)
        ).to(params.device)

        nn.init.xavier_uniform_(self.positional_embeddings)

        self.decoders_blocks = [
            DecoderBlock(
                multihead_attention=MultiHeadAttention(
                    n_heads=params.n_heads,
                    input_dim=params.input_dim,
                    query_dim=params.query_dim,
                    value_dim=params.value_dim,
                    device=params.device
                ),

                feed_forward=FeedForward(
                    input_dim=params.input_dim,
                    hidden_dim=params.feed_forward_hidden_dim,
                    output_dim=params.input_dim,
                    device=params.device
                ),

                first_layer_norm=nn.LayerNorm(
                    normalized_shape=params.input_dim, device=params.device
                ),

                second_layer_norm=nn.LayerNorm(
                    normalized_shape=params.input_dim, device=params.device
                ),
            )

            for _ in range(params.n_decoder_blocks)
        ]

        self.output_layer = nn.Linear(in_features=params.input_dim, out_features=params.vocab_size, device=params.device)

    def forward(self, x, attention_mask):
        x = self.embeddings(x) + self.positional_embeddings[:x.size(1)]

        for decoder in self.decoders_blocks:
            x = decoder(x, attention_mask)

        return self.output_layer(x)
    
    def save(self, path_to_folder: str):
        self.save_weights(path_to_folder)
        self.save_config(path_to_folder)

    def save_weights(self, path_to_folder: str):
        if not os.path.isdir(path_to_folder):
            os.mkdir(path_to_folder)

        state_dict = self.state_dict()
        for key, value in state_dict.items():
            state_dict[key] = value.to("cpu")

        torch.save(state_dict, path_to_folder + "/" + "model.pt")

    def save_config(self, path_to_folder: str):
        with open(path_to_folder + "/" + "model_config.json", "w") as f:
            json.dump(self.params.__dict__, f)

    @staticmethod
    def from_pretrained(path_to_folder: str):
        with open(path_to_folder + "/model_config.json", "r") as f:
            params = json.load(f)

        model = GPT(GPTParams(**params))
        model.load_state_dict(torch.load(path_to_folder + "/model.pt"))

        return model
