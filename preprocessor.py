import torch

from typing import List
from tqdm.notebook import tqdm

from tokenizer import Tokenizer

class Preprocessor:

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer

    def preprocess(self, corpus: List[str], context_size: int, batch_size: int):
        tokenized_corpus = self._tokenize_corpus(corpus, batch_size)
        tokenized_corpus = self._align_tokenized_corpus(tokenized_corpus)

        tokenized_corpus = self._split_aligned_tokenized_corpus(tokenized_corpus, context_size)

        return tokenized_corpus

    def _tokenize_corpus(self, corpus: List[str], batch_size: int) -> List[List[int]]:
        tokenized_corpus = []

        for k in tqdm(range(0, len(corpus), batch_size)):
            tokenized_corpus.extend(
                self.tokenizer(corpus[k : k + batch_size]).input_ids
            )
        
        return tokenized_corpus

    def _align_tokenized_corpus(self, tokenized_corpus: List[List[int]]):
        return [i for ids in tokenized_corpus for i in ids + [self.tokenizer.pad_token_id]]
    
    def _split_aligned_tokenized_corpus(self, aligned_tokenized_corpus: List[int], context_size: int) -> List[List[int]]:
        splits = []

        for k in range(0, len(aligned_tokenized_corpus), context_size):
            splits.append(aligned_tokenized_corpus[k : k + context_size])

        return splits[:-1]