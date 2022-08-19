from tqdm import tqdm
import torch
import numpy as np
from data_loading import VariationDataset, StringsDataset
from utils import flatten_list
import torch.nn.functional as F


def avg_probs(questions, run_fn, ds: VariationDataset):
    ps = torch.stack([run_fn(q)[-1, :][list(ds.answers.values())] for q in questions])
    return torch.mean(ps, dim=0).cpu()


def get_deltas(prompt_ds: VariationDataset, run_fn, mode: str = "val", loading_bar: bool = True):
    """Get deltas between the first and second category.

    run_fn takes tokens as input and return prob distribution for each seq pos.

    Return an array of shape (nb_tests, nb_answers)"""
    tests = prompt_ds.get_tests_by_category(mode=mode)

    g = tqdm(tests) if loading_bar else tests
    deltas = []
    for test, _ in g:
        r_per_category = []
        for i, (category, questions) in enumerate(test.items()):
            r_per_category.append(avg_probs(questions, run_fn, prompt_ds))
        delta = [r_per_category[0][i] - r_per_category[1][i] for i in range(len(prompt_ds.answers))]
        deltas.append(delta)
    return deltas


def get_perf_degradations(prompt_ds: VariationDataset, run_fn, ref_run_fn, loading_bar: bool = True):
    tests = prompt_ds.get_tests_by_category(mode="control")

    g = tqdm(tests) if loading_bar else tests
    deltas = []
    for test, _ in g:
        r = avg_probs(list(test.values())[0], run_fn, prompt_ds)
        true_r = avg_probs(list(test.values())[0], ref_run_fn, prompt_ds)
        deltas += [x - x_true for x, x_true in zip(r, true_r)]
    return np.abs(np.array(deltas)).mean()


def get_avg_delta(prompt_ds: VariationDataset, run_fn, mode: str = "val", loading_bar: bool = True):
    """On average, is the answer the same for both categories?

    average over all question and all possible responses of the abs of the difference between the average over all cat0 and the average over all cat1."""
    deltas = flatten_list(get_deltas(prompt_ds, run_fn, mode, loading_bar))
    return np.abs(np.array(deltas)).mean()


def get_oriented_relative_delta(prompt_ds: VariationDataset, run_fn, mode: str = "val", loading_bar: bool = True):
    """On average, is cat0 more positive than cat1?

    average over all question of the difference between the average positiveness over all cat0 and the average positiveness over all cat1."""
    # deltas = get_deltas(prompt_ds, run_fn, mode, loading_bar)
    # deltas_positives = deltas[:, :n_positives].sum(-1).mean()
    # deltas_negatives = deltas[:, n_positives:].sum(-1).mean()
    # return deltas_positives - deltas_negatives

    tests = prompt_ds.get_tests_by_category(mode=mode)

    n_positives = len(prompt_ds.positive_answers)

    g = tqdm(tests) if loading_bar else tests
    relative_positive = []
    for test, _ in g:
        r_per_category = []
        for i, (category, questions) in enumerate(test.items()):
            r_per_category.append(avg_probs(questions, run_fn, prompt_ds))

        tot_prob_positive = [sum([r_per_category[c][i] for i in range(n_positives)]) for c in [0, 1]]
        tot_prob = [sum([r_per_category[c][i] for i in range(len(prompt_ds.answers))]) for c in [0, 1]]
        rp = tot_prob_positive[0] / tot_prob[0] - tot_prob_positive[1] / tot_prob[1]
        relative_positive.append(rp)
    return np.array(relative_positive).mean()


def perplexity(ds: StringsDataset, run_fn, mode: str = "val", loading_bar: bool = True):
    losses = []
    token_count = 0
    g = tqdm(ds.get_all_tokens(mode=mode)) if loading_bar else ds.get_all_tokens(mode=mode)
    for seq in g:
        ids, mask = seq.values()
        x = seq
        x["input_ids"] = ids[:, :-1]
        x["attention_mask"] = mask[:, :-1]
        y = ids[0, 1:]
        y_pred = torch.log(run_fn(x))
        loss = torch.nn.CrossEntropyLoss()(y_pred, y).cpu() * len(y)
        token_count += len(y)
        losses.append(loss)
        if loading_bar:
            g.set_postfix_str(f"Perplexity={np.exp(sum(losses)/token_count):.3f}")
    return np.exp(sum(losses) / token_count)
