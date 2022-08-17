import itertools
import json
from typing import Optional
from attrs import define
from abc import ABC, abstractmethod
from transformers import GPT2Tokenizer, PreTrainedTokenizerBase
import os
from pathlib import Path

import torch

from utils import flatten_activations


class WikiDataset:
    def __init__(self, model_word: str, data_folder: str = "wikitext"):
        self.tokenizer: PreTrainedTokenizerBase = GPT2Tokenizer.from_pretrained(model_word)
        self.folder = data_folder
        self.text = {}
        for mode in ["valid", "test"]:
            with (Path("data") / data_folder / f"wiki.{mode}.raw").open(encoding="utf-8") as f:
                if mode == "valid":
                    mode = "val"  # Stay consistent with other ds
                self.text[
                    mode
                ] = (
                    f.readlines()
                )  # In practice, each line is short enough to fit in the prompt, but reasonably long too (max 500 words, average of 55). We use one line as the unit of text. Not super scientific but that will be enough. Anyway, the encoding is a little messed up too...

    def get_all_strs(self, mode: str = "val"):
        return self.text[mode]

    def get_all_tokens(self, mode: str = "val"):
        text = self.text[mode]
        return [self.tokenizer(t, return_tensors="pt") for t in text]
