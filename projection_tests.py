#%%
%load_ext autoreload
%autoreload 2
#%%
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from activation_utils import (
    get_activations,
    get_all_activations,
    get_res_layers,
    run_and_modify,
    get_mlp_layers,
)

from data_loading import PromptDataset, WikiDataset
from activation_ds import ActivationsDataset
from inlp import inlp
from linear import get_linear_cut, get_multi_lin_cut
from logit_lense import print_logit_lense
from metrics import get_avg_delta, get_perf_degradations, perplexity
from utils import make_projections, orthonormalize
from tqdm import tqdm
import pandas as pd
#%%
df = pd.read_csv("data/reddit_by_subreddit/AskReddit.csv")
#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
print(model.device)
#%%
ds = PromptDataset("gpt2", "men_v_momen_double_bind")
#%%
wikids = WikiDataset("gpt2")
#%%
def default_run_fn(inps):
    with torch.no_grad():
        return torch.softmax(model(**inps).logits[0, :, :], -1)
perplexity(wikids, default_run_fn, loading_bar=True)
# %%

stud_layers = get_mlp_layers(model, [5, 7, 8])
activations = get_all_activations(ds, model, stud_layers)
# %%

act_ds = ActivationsDataset.from_data(activations, stud_layers, device)

# %%
#%%
# dirs = inlp(act_ds, 16, max_iters=10_000, use_torch=False)
dirs = inlp(act_ds, 16, max_iters=200, use_torch=True)
#%%
dirs, acc = get_multi_lin_cut(act_ds, 8, epochs=500)
print(acc)
# %%
for v in dirs:
    print_logit_lense(model, ds.tokenizer, v)
# %%
modifications_fns = {"default": {}}
# modifications_fns["rdm32"] = dict(
#     [
#         (layer, make_projections(torch.eye(act_ds.x_data.shape[-1])[:32, :]))
#         for layer in stud_layers
#     ]
# )
for i in [1, 4, 8]:
    modifications_fns[f"proj{i}"] = dict([(layer, make_projections(dirs[:i], is_rest=False)) for layer in stud_layers])

# %%


def to_probs(output, ds: PromptDataset):
    return torch.softmax(output.logits[0, -1], -1)[list(ds.answers.values())].detach()


def avg_probs(questions, run_fn, ds):
    ps = torch.cat([to_probs(run_fn(q), ds)[None, :] for q in questions])
    return torch.mean(ps, dim=0)


for mode in ["train", "val"]:
    # for mode in ["control", "train", "val"]:
    print(mode)
    tests = ds.get_tests_by_category(mode=mode)
    for test, prompt in tests:
        print(prompt)
        for name, modif in modifications_fns.items():
            print(name)
            r = {}
            for category, questions in test.items():
                ps = avg_probs(questions, lambda x: run_and_modify(x, model, modif), ds)
                r[category] = ps

            if mode != "control":
                delta_yes = r["male"][0] - r["female"][0]
                delta_no = r["male"][1] - r["female"][1]
                print(
                    " ".join(f"{c} yes:{ps[0]:.3f} no:{ps[1]:.3f}" for c, ps in r.items()),
                    f"delta yes:{delta_yes:.5f}",
                    f"delta no:{delta_no:.5f}",
                )

# %%
run_fns = dict(
    [
        (
            name,
            lambda x, modif=modif: torch.softmax(run_and_modify(x, model, modif).logits[0, :, :].detach(), -1),
        )
        for name, modif in modifications_fns.items()
    ]
)
for name, run_fn in run_fns.items():
    print(
        name,
        get_avg_delta(ds, run_fn),
        get_perf_degradations(ds, run_fn, run_fns["default"]),
    )

# %%
