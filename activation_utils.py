import torch

from data_loading import PromptDataset


def get_all_activations(ds: PromptDataset, model, mode: str = "val"):
    prompts = ds.get_all_tokens(mode)
    activations = {}
    for category, l in prompts.items():
        activations[category] = {}
        for i, inps in enumerate(l):
            acts = get_activations(inps, model)
            for layer, act in acts.items():
                if i == 0:
                    activations[category][layer] = []
                activations[category][layer].append(act)
    return activations


def get_activations(tokens, model, operation=lambda x: x):
    handles = []
    activations = {}

    def hook_fn(module, inp, out):
        activations[module] = operation(out[0].detach())

    for layer in model.transformer.h:
        handles.append(layer.register_forward_hook(hook_fn))
    try:
        model(**tokens.to(model.device))
    except Exception as e:
        print(e)
        pass
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
        for handle in handles:
            handle.remove()
        return out
    except Exception as e:
        for handle in handles:
            handle.remove()
        print(e)
        return None
