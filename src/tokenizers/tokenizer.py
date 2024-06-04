import os
import torch
import json

from typing import Dict, List, Tuple
from dataclasses import dataclass

from src.tokenizers.bpe import BytePairEncoder

from typing import Any

#TODO: rewrite logic from torch to numpy or lists


@dataclass
class TokenizerOutput:
    input_ids: torch.Tensor | List[int] | List[List[int]]
    attention_mask: torch.Tensor | List[int] | List[List[int]]


class Tokenizer:

    def __init__(self, bpe: BytePairEncoder) -> None:
        self.bpe = bpe

        self.vocab = {}
        self.reverse_vocab = {}

        for k, token in enumerate(bpe.vocab):
            self.vocab[token] = k
            self.reverse_vocab[k] = token

        self.pad_token_id = self.vocab[bpe.end_of_string_token]

    def __call__(self, texts: List[str] | str, return_tensors: str | None = None) -> Any:
        if isinstance(texts, str):
            output = self.tokenize(texts)

        elif isinstance(texts, list):            
            output = self.tokenize_many(texts)
        
        else:
            raise Exception("Unprocessable input type.")
        
        return TokenizerOutput(input_ids=Tokenizer._transform_to_tensor(output, return_tensors), attention_mask=None)

    def tokenize(self, text: str) -> List[int]:
        tokenized_text = map(lambda token: self.vocab[token], self.bpe.encode(text))
        tokenized_text = list(tokenized_text)
        
        return tokenized_text
    
    def tokenize_many(self, texts: List[str]) -> List[List[int]]:
        tokenized_texts = [self.tokenize(text) for text in texts]

        return tokenized_texts
    
    def decode(self, ids: List[int]) -> str:
        decoded_string = ""

        for i in ids:
            decoded_string += self.reverse_vocab[i]

        return decoded_string
    
    def save(self, path_to_folder) -> None:
        if not os.path.isdir(path_to_folder):
            os.mkdir(path_to_folder)
        
        merges = self.bpe.merges
        merges = [{"left": pair[0], "right": pair[1], "merge": merge} for pair, merge in merges.items()]
        
        Tokenizer.__save_json(merges, path_to_folder + "/" + "merges.json")
        
        vocab = self.bpe.vocab
        vocab = {value: index for index, value in enumerate(vocab)}

        Tokenizer.__save_json(vocab, path_to_folder + "/" + "vocab.json") 

    @staticmethod
    def from_pretrained(path_to_folder: str):
        return Tokenizer(BytePairEncoder.from_pretrained(path_to_folder))
    
    @staticmethod
    def _transform_to_tensor(output: List[int] | List[List[int]], tensor_type: str | None):
        if tensor_type is None:
            return output
        
        if tensor_type == "pt":
            return torch.tensor(output)
    
    @staticmethod
    def __save_json(json_dict: dict, path: str):
        with open(path, "w") as f:
            json.dump(json_dict, f)
