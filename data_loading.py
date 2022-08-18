import itertools
import json
from typing import Optional
from attrs import define
from abc import ABC, abstractmethod
from transformers import GPT2Tokenizer, PreTrainedTokenizerBase
import os
from pathlib import Path

from utils import flatten_activations

import pandas as pd

from abc import ABC, abstractmethod


class VariationDataset(ABC):
    @abstractmethod
    def get_strs_by_category(self, mode: str = "train"):
        """Returns a dict of list of strings, keys are categories"""
        pass

    @abstractmethod
    def get_tokens_by_category(self, mode: str = "train"):
        """Returns a dict of list of model inputs (toks + attn mask), keys are categories"""
        pass

    @abstractmethod
    def get_tests_by_category(self, mode: str = "train"):
        """Returns a list of dict of a list of slightly different model inputs (toks + attn mask), keys are categories"""
        pass


class PromptDataset(VariationDataset):
    def __init__(self, model_name: str, data_folder: str):
        self.tokenizer: PreTrainedTokenizerBase = GPT2Tokenizer.from_pretrained(model_name)
        self.folder = data_folder
        self.questions = {}
        for mode in ["train", "val", "control"]:
            with (Path("data") / data_folder / f"{mode}.json").open() as f:
                self.questions[mode] = json.load(f)
        with (Path("data") / data_folder / "replacements.json").open() as f:
            self.replacements = json.load(f)
        with (Path("data") / data_folder / "settings.json").open() as f:
            self.settings = json.load(f)
        self.positive_answers = dict(
            (positive_answer, self.tokenizer(positive_answer)["input_ids"][0])
            for positive_answer in self.settings["positive_answers"]
        )
        self.negative_answers = dict(
            (negative_answer, self.tokenizer(negative_answer)["input_ids"][0])
            for negative_answer in self.settings["negative_answers"]
        )

    @property
    def answers(self):
        return {**self.positive_answers, **self.negative_answers}

    def transform_prompt(self, prompt, word):
        if isinstance(word, str):
            return prompt.replace("_", word)
        elif isinstance(word, list):
            p = prompt
            for i, r in enumerate(word):
                p = p.replace(f"_{i}", r)
            return p

    def transform_questions(self, questions):
        return dict(
            (
                category,
                list(
                    set([self.transform_prompt(prompt, word) for word, prompt in itertools.product(words, questions)])
                ),
            )
            for category, words in self.replacements.items()
        )

    def get_strs_by_category(self, mode: str = "train"):
        questions = self.questions[mode]
        return self.transform_questions(questions)

    def get_tokens_by_category(self, mode: str = "train"):
        questions = self.questions[mode]
        prompts_per_category = self.transform_questions(questions)
        tokenized_prompts = dict(
            (
                category,
                [self.tokenizer(prompt, return_tensors="pt") for prompt in prompts],
            )
            for category, prompts in prompts_per_category.items()
        )
        return tokenized_prompts

    def get_tests_by_category(self, mode: str = "val"):
        prompts = self.questions[mode]
        preprompt = self.settings["preprompt"]
        question = self.settings["question"]
        tests = []
        for prompt in prompts:
            d = {}
            replacements = {"control": [""]} if mode == "control" else self.replacements
            for category, words in replacements.items():
                transformed_prompts = [self.transform_prompt(prompt, word) for word in words]
                d[category] = [
                    self.tokenizer(preprompt + p + question, return_tensors="pt") for p in transformed_prompts
                ]
            tests.append((d, prompt))
        return tests


class StringsDataset(ABC):
    @abstractmethod
    def get_all_strs(self, mode: str = "train"):
        """Returns a list of strings"""
        pass

    @abstractmethod
    def get_all_tokens(self, mode: str = "train"):
        """Returns a list of tokens"""
        pass


class RedditDataset(StringsDataset):
    def __init__(self, model_name: str, strs: list, val_prop: float = 0.3):
        self.tokenizer: PreTrainedTokenizerBase = GPT2Tokenizer.from_pretrained(model_name)
        self.text = {}
        nb_train = int(len(strs) * (1 - val_prop))
        self.text["train"] = strs[:nb_train]
        self.text["val"] = strs[nb_train:]

    @classmethod
    def from_file(cls, model_name: str, file_path: str = "reddit_by_subreddit/books.csv", val_prop: float = 0.3):
        condition = lambda s: len(s.split()) < 500
        strs = [s for s in pd.read_csv(Path("data") / file_path).selftext.dropna() if condition(s)]
        return RedditDataset(model_name, strs, val_prop)

    @classmethod
    def from_folder(cls, model_name: str, folder_path: str = "reddit_by_subreddit", val_prop: float = 0.3):
        condition = lambda s: len(s.split()) < 500
        strs = []
        for path in (Path("data") / folder_path).glob("*.csv"):
            strs += [s for s in pd.read_csv(path).selftext.dropna() if condition(s)]
        return RedditDataset(model_name, strs, val_prop)

    def get_all_strs(self, mode: str = "train"):
        return self.text[mode]

    def get_all_tokens(self, mode: str = "train"):
        text = self.text[mode]
        return [self.tokenizer(t, return_tensors="pt") for t in text]


class WikiDataset(StringsDataset):
    def __init__(self, model_name: str, data_folder: str = "wikitext"):
        self.tokenizer: PreTrainedTokenizerBase = GPT2Tokenizer.from_pretrained(model_name)
        self.folder = data_folder
        self.text = {}
        for mode in ["valid", "test"]:
            with (Path("data") / data_folder / f"wiki.{mode}.raw").open(encoding="utf-8") as f:
                mode = {"valid": "train", "test": "val"}[mode]
                self.text[mode] = f.readlines()
                # In practice, each line is short enough to fit in the prompt, but reasonably long too (max 500 words, average of 55). We use one line as the unit of text. Not super scientific but that will be enough. Anyway, the encoding is a little messed up too...

    def get_all_strs(self, mode: str = "train"):
        return self.text[mode]

    def get_all_tokens(self, mode: str = "train"):
        text = self.text[mode]
        return [self.tokenizer(t, return_tensors="pt") for t in text]
