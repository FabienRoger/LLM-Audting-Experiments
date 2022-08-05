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


class PromptDataset:
    def __init__(self, model_name: str, data_folder: str):
        self.tokenizer: PreTrainedTokenizerBase = GPT2Tokenizer.from_pretrained(
            model_name
        )
        self.folder = data_folder
        self.questions = {}
        for mode in ["train", "val", "control"]:
            with (Path("data") / data_folder / f"{mode}.json").open() as f:
                self.questions[mode] = json.load(f)
        with (Path("data") / data_folder / "replacements.json").open() as f:
            self.replacements = json.load(f)
        with (Path("data") / data_folder / "settings.json").open() as f:
            self.settings = json.load(f)
        self.answers = dict(
            (answer, self.tokenizer(answer)["input_ids"][0])
            for answer in self.settings["answers"]
        )

    def transform_prompt(self, prompt, name):
        return prompt.replace("_", name)

    def transform_questions(self, questions):
        return dict(
            (
                category,
                list(
                    set(
                        [
                            self.transform_prompt(prompt, name)
                            for name, prompt in itertools.product(words, questions)
                        ]
                    )
                ),
            )
            for category, words in self.replacements.items()
        )

    def get_all_strs(self, mode: str = "train"):
        questions = self.questions[mode]
        return self.transform_questions(questions)

    def get_all_tokens(self, mode: str = "train"):
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

    def get_all_tests(self, mode: str = "val"):
        prompts = self.questions[mode]
        preprompt = self.settings["preprompt"]
        question = self.settings["question"]
        tests = []
        for prompt in prompts:
            d = {}
            replacements = {"control": [""]} if mode == "control" else self.replacements
            for category, words in replacements.items():
                transformed_prompts = [
                    self.transform_prompt(prompt, name) for name in words
                ]
                d[category] = [
                    self.tokenizer(preprompt + p + question, return_tensors="pt")
                    for p in transformed_prompts
                ]
            tests.append((d, prompt))
        return tests


class ActivationsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ):
        super().__init__()
        self.x_data = x
        self.y_data = y

    @classmethod
    def from_data(cls, data, layers: list, device: str = "cpu"):
        data_x = []
        data_y = []
        for i, category in enumerate(data.keys()):
            x = torch.cat(
                tuple(flatten_activations(data[category][layer]) for layer in layers)
            )
            data_x.append(x)
            y = torch.zeros(x.shape[0], len(data.keys()))
            y[:, i] = 1
            data_y.append(y)
        return cls(torch.cat(data_x).to(device), torch.cat(data_y).to(device))

    def project(self, dir: torch.Tensor):
        dir_norm = dir / torch.linalg.norm(dir)
        new_x_data = self.x_data - torch.outer((self.x_data @ dir_norm), dir_norm)
        return ActivationsDataset(new_x_data, self.y_data)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx, :]
        y = self.y_data[idx, :]
        sample = (x, y)
        return sample