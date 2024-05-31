import json
import os
import regex as re

from typing import Dict, List, Tuple
from dataclasses import dataclass

from collections import defaultdict

from tqdm import tqdm


class BytePairEncoder:

    SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self, alphabet: List[str] | None = None, end_of_string_token: str = "<|endoftext|>") -> None:
        self.alphabet = alphabet
        self.end_of_string_token = end_of_string_token

        self.pattern = re.compile(BytePairEncoder.SPLIT_PATTERN)

        self.vocab: List[str] = None

        self.word_freqs: defaultdict = defaultdict(int)
        self.merges: Dict[Tuple[str, str], str] = {}

        self.cache: Dict[str, List[str]] = {}

    def train(self, corpus: List[str], max_vocab_size: int = None, max_merges: int = None, verbose: bool = False):
        total_iterations, stopping_type = BytePairEncoder._validate_train_params(max_vocab_size, max_merges)

        progress_bar = tqdm(total=total_iterations)
        
        self._compute_words_frequencies(corpus)

        self.vocab = self._compute_base_vocabulary()

        if stopping_type == "vocab":
            progress_bar.update(len(self.vocab))

        word_characters = {word: [character for character in word] for word in self.word_freqs.keys()}
        while True:
            pair_freqs = self._compute_pair_frequencies(word_characters)

            first_letter, second_letter, _ = BytePairEncoder._get_most_frequent_pair(pair_freqs)

            word_characters = self._merge_pair(first_letter, second_letter, word_characters)

            self.merges[(first_letter, second_letter)] = first_letter + second_letter
            self.vocab.append(first_letter + second_letter)

            if stopping_type == "vocab":
                if len(self.vocab) >= total_iterations:
                    break

            if stopping_type == "merge":
                if len(self.merges) >= total_iterations:
                    break

            progress_bar.update(1)

        progress_bar.update(1)
        progress_bar.close()

    def encode(self, text: str) -> List[str]:
        words = self._split_text(text)

        bpe_tokens = []
        for word in words:
            if word in self.cache:
                bpe_tokens.extend(self.cache[word])
            
            else:
                split = list(word)
                
                for pair, merge in self.merges.items():
                    i = 0
                    while i < len(split) - 1:
                        if split[i] == pair[0] and split[i + 1] == pair[1]:
                            split = split[:i] + [merge] + split[i + 2 :]
                        else:
                            i += 1

                bpe_tokens.extend(split)
                
                self.cache[word] = split

        return bpe_tokens
    
    def save(self, path_to_folder: str):
        os.mkdir(path_to_folder)

        merges = [{"left": pair[0], "right": pair[1], "merge": merge} for pair, merge in self.merges.items()]
        vocab = {token: k for k, token in enumerate(self.vocab)}

        with open(path_to_folder + "/merges.json", "w") as f:
            json.dump(merges, f)

        with open(path_to_folder + "/vocab.json", "w") as f:
            json.dump(vocab, f)

    def _compute_words_frequencies(self, corpus: List[str]):
        for text in corpus:
            words = self._split_text(text)
            for word in words:
                self.word_freqs[word] += 1
    
    def _compute_pair_frequencies(self, word_characters: Dict[str, List[str]]) -> defaultdict:
        pair_freqs = defaultdict(int)
        
        for word, freq in self.word_freqs.items():
            characters = word_characters[word]

            if len(characters) != 1:
                for k in range(len(characters) - 1):
                    pair = (characters[k], characters[k + 1])

                    pair_freqs[pair] += freq

        return pair_freqs
    
    def _split_text(self, text: str) -> List[str]:        
        splitted_text = re.findall(self.pattern, text)

        return splitted_text
    
    def _merge_pair(self, first_letter: str, second_letter: str, word_characters: Dict[str, List[str]]):
        for word in word_characters.keys():
            split = word_characters[word]
            if len(split) == 1:
                continue

            i = 0
            while i < len(split) - 1:
                if split[i] == first_letter and split[i + 1] == second_letter:
                    split = split[:i] + [first_letter + second_letter] + split[i + 2 :]
                else:
                    i += 1
            word_characters[word] = split

        return word_characters
    
    def _compute_base_vocabulary(self) -> List[str]:
        self.alphabet = self.alphabet or []
        
        for word in self.word_freqs.keys():
            for letter in word:
                if letter not in self.alphabet:
                    self.alphabet.append(letter)
        
        self.alphabet.sort()

        self.vocab = [self.end_of_string_token] + self.alphabet

        return self.vocab
    
    @staticmethod
    def from_pretrained(path_to_folder: str):
        with open(path_to_folder + "/merges.json", "r") as f:
            merges = json.load(f)
        merges = {(item["left"], item["right"]): item["merge"] for item in merges}

        with open(path_to_folder + "/vocab.json", "r") as f:
            vocab = json.load(f)

        vocab = list(vocab.keys())

        bpe = BytePairEncoder()
        bpe.merges = merges
        bpe.vocab = vocab

        return bpe
    
    @staticmethod
    def _get_most_frequent_pair(pair_freqs: defaultdict) -> Tuple[str, str, int]:
        (first_letter, second_letter), freq = max(pair_freqs.items(), key=lambda item: item[1])

        return first_letter, second_letter, freq
    
    @staticmethod
    def _validate_train_params(max_vocab_size: int = None, max_merges: int = None):
        if max_vocab_size is None and max_merges is None:
            raise Exception('Either the "max_vocab_size" or "max_merges" has to be specified.')
        
        if max_vocab_size is not None and max_merges is not None:
            raise Exception('You have to specifiy either "max_vocab_size" or "max_merges".')
        
        if max_vocab_size is not None and max_merges is None:
            return max_vocab_size, "vocab"
        
        if max_vocab_size is None and max_merges is not None:
            return max_merges, "merge"