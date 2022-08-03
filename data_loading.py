import itertools
import json
from typing import Literal
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
        self.answers_tokens = [
            self.tokenizer(answer)["input_ids"][0]
            for answer in self.settings["answers"]
        ]

    def transform_questions(self, questions):
        return dict(
            (
                word_class,
                list(
                    set(
                        [
                            prompt.replace("_", name)
                            for name, prompt in itertools.product(words, questions)
                        ]
                    )
                ),
            )
            for word_class, words in self.replacements.items()
        )

    def get_all_tokens(
        self, mode: Literal["val", "train", "control"] = "train"
    ) -> dict[str, list]:
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

    def get_all_tests(
        self, mode: Literal["val", "train", "control"] = "val"
    ) -> dict[str, list]:
        questions = self.questions[mode]
        prompts_per_category = self.transform_questions(questions)

        preprompt = self.settings["preprompt"]
        question = self.settings["question"]
        tokenized_prompts = dict(
            (
                category,
                [
                    self.tokenizer(preprompt + prompt + question, return_tensors="pt")
                    for prompt in prompts
                ],
            )
            for category, prompts in prompts_per_category.items()
        )
        return tokenized_prompts


class ActivationsDataset(torch.utils.data.Dataset):
    def __init__(self, data: dict[str, dict], layers: list, device: str = "cpu"):
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
        self.x_data = torch.cat(data_x).to(device)
        self.y_data = torch.cat(data_y).to(device)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx, :]
        y = self.y_data[idx, :]
        sample = (x, y)
        return sample
