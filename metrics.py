from tqdm import tqdm
from torch import torch
import numpy as np

from utils import flatten_list


def avg_probs(questions, run_fn, ds):
    ps = torch.stack([run_fn(q)[list(ds.answers.values())] for q in questions])
    return torch.mean(ps, dim=0)


def get_deltas(prompt_ds, run_fn, mode: str = "val", loading_bar: bool = True):
    """Get deltas between the first and second category.

    run_fn takes a string as input and return probs for the next token."""
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


def get_perf_degradations(prompt_ds, run_fn, ref_run_fn, loading_bar: bool = True):
    tests = prompt_ds.get_all_tests(mode="control")

    g = tqdm(tests) if loading_bar else tests
    deltas = []
    for test, _ in g:
        r = avg_probs(list(test.values())[0], run_fn, prompt_ds)
        true_r = avg_probs(list(test.values())[0], ref_run_fn, prompt_ds)
        deltas += [x - x_true for x, x_true in zip(r, true_r)]
    return np.abs(np.array(deltas)).mean()


def get_avg_delta(prompt_ds, run_fn, mode: str = "val", loading_bar: bool = True):
    deltas = flatten_list(get_deltas(prompt_ds, run_fn, mode, loading_bar))
    return np.abs(np.array(deltas)).mean()
