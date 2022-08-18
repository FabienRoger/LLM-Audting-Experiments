from typing import Optional
import torch

from data_loading import StringsDataset, VariationDataset


def get_mlp_layers(model, layer_numbers: Optional[list]):
    layer_numbers = list(range(len(model.transformer.h))) if layer_numbers is None else layer_numbers
    return [l.mlp for i, l in enumerate(model.transformer.h) if i in layer_numbers]


def get_res_layers(model, layer_numbers: Optional[list]):
    layer_numbers = list(range(len(model.transformer.h))) if layer_numbers is None else layer_numbers
    return [l for i, l in enumerate(model.transformer.h) if i in layer_numbers]


def get_all_activations(ds: VariationDataset, model, layers, mode: str = "val"):
    prompts = ds.get_tokens_by_category(mode)
    activations = {}
    for category, l in prompts.items():
        activations[category] = {}
        for i, inps in enumerate(l):
            acts = get_activations(inps, model, layers)
            for layer, act in acts.items():
                if i == 0:
                    activations[category][layer] = []
                activations[category][layer].append(act)
    return activations


def get_corresponding_activations(datasets, model, layers, mode: str = "val"):
    """datasets is a dict where keys are categories & values are StringDatasets."""
    activations = {}
    for category, ds in datasets.items():
        ds: StringsDataset = ds
        prompts = ds.get_all_tokens(mode)
        activations[category] = {}
        for i, inps in enumerate(prompts):
            acts = get_activations(inps, model, layers)
            for layer, act in acts.items():
                if i == 0:
                    activations[category][layer] = []
                activations[category][layer].append(act)
    return activations


def get_activations(tokens, model, layers, operation=lambda x: x):
    handles = []
    activations = {}

    def hook_fn(module, inp, out):
        activations[module] = operation(out[0].detach())

    for layer in layers:
        handles.append(layer.register_forward_hook(hook_fn))
    try:
        model(**tokens.to(model.device))
    except Exception as e:
        raise e
    finally:
        for handle in handles:
            handle.remove()
    return activations


def run_and_modify(tokens, model, modification_fns: dict = {}):
    handles = []
    for layer, f in modification_fns.items():
        handles.append(layer.register_forward_hook(f))
    try:
        out = model(**tokens.to(model.device))
        return out
    except Exception as e:
        raise e
    finally:
        for handle in handles:
            handle.remove()
