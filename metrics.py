from tqdm import tqdm
from torch import torch
import numpy as np
from .data_loading import PromptDataset

from utils import flatten_list


def avg_probs(questions, run_fn, ds: PromptDataset):
    ps = torch.stack([run_fn(q)[list(ds.answers.values())] for q in questions])
    return torch.mean(ps, dim=0)


def get_deltas(
    prompt_ds: PromptDataset, run_fn, mode: str = "val", loading_bar: bool = True
):
    """Get deltas between the first and second category.

    run_fn takes a string as input and return probs for the next token.

    Return an array of shape (nb_tests, nb_answers)"""
    tests = prompt_ds.get_all_tests(mode=mode)

    g = tqdm(tests) if loading_bar else tests
    deltas = []
    for test, _ in g:
        r_per_category = []
        for i, (category, questions) in enumerate(test.items()):
            r_per_category.append(avg_probs(questions, run_fn, prompt_ds))
        delta = [
            r_per_category[0][i] - r_per_category[1][i]
            for i in range(len(prompt_ds.answers))
        ]
        deltas.append(delta)
    return deltas


def get_perf_degradations(
    prompt_ds: PromptDataset, run_fn, ref_run_fn, loading_bar: bool = True
):
    tests = prompt_ds.get_all_tests(mode="control")

    g = tqdm(tests) if loading_bar else tests
    deltas = []
    for test, _ in g:
        r = avg_probs(list(test.values())[0], run_fn, prompt_ds)
        true_r = avg_probs(list(test.values())[0], ref_run_fn, prompt_ds)
        deltas += [x - x_true for x, x_true in zip(r, true_r)]
    return np.abs(np.array(deltas)).mean()


def get_avg_delta(
    prompt_ds: PromptDataset, run_fn, mode: str = "val", loading_bar: bool = True
):
    """On average, is the answer the same for both categories?

    average over all question and all possible responses of the abs of the difference between the average over all cat0 and the average over all cat1."""
    deltas = flatten_list(get_deltas(prompt_ds, run_fn, mode, loading_bar))
    return np.abs(np.array(deltas)).mean()


def get_oriented_relative_delta(
    prompt_ds: PromptDataset, run_fn, mode: str = "val", loading_bar: bool = True
):
    """On average, is cat0 more positive than cat1?

    average over all question of the the difference between the average over all cat0 and the average over all cat1."""
    deltas = get_deltas(prompt_ds, run_fn, mode, loading_bar)
    n_positives = len(prompt_ds.positive_answers)
    deltas_positives = deltas[:, :n_positives].sum(-1).mean()
    deltas_negatives = deltas[:, n_positives:].sum(-1).mean()
    return deltas_positives - deltas_negatives
