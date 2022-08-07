#%%
# %load_ext autoreload
# %autoreload 2
#%%
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from activation_utils import get_activations, get_all_activations, run_and_modify

from data_loading import ActivationsDataset, PromptDataset
from inlp import inlp
from linear import get_linear_cut
from logit_lense import print_logit_lense
from utils import make_projections, orthonormalize
from tqdm import tqdm

#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
print(model.device)
#%%
ds = PromptDataset("gpt2", "men_v_women")
# %%
toks = ds.get_all_tokens()
# %%

activations = get_all_activations(ds, model)
# %%

stud_layers = [model.transformer.h[l] for l in [4, 5, 6, 7, 8]]
act_ds = ActivationsDataset.from_data(activations, stud_layers, device)

# %%
#%%
dirs = inlp(act_ds, 16, max_iters=10_000)
# %%
for v in dirs:
    print_logit_lense(model, ds.tokenizer, v)
# %%
modifications_fns = {"default": {}}
modifications_fns["rdm32"] = dict(
    [
        (layer, make_projections(torch.eye(act.ds.shape[-1])[:32]))
        for layer in stud_layers
    ]
)
for i in [1, 4, 8]:
    modifications_fns[f"proj{i}"] = dict(
        [(layer, make_projections(dirs[:i])) for layer in stud_layers]
    )

# %%


def to_probs(output, ds: PromptDataset):
    return torch.softmax(output.logits[0, -1], -1)[list(ds.answers.values())].detach()


def avg_probs(questions, run_fn, ds):
    ps = torch.cat([to_probs(run_fn(q), ds)[None, :] for q in questions])
    return torch.mean(ps, dim=0)


for mode in ["train", "val"]:
    # for mode in ["control", "train", "val"]:
    print(mode)
    tests = ds.get_all_tests(mode=mode)
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
                    " ".join(
                        f"{c} yes:{ps[0]:.3f} no:{ps[1]:.3f}" for c, ps in r.items()
                    ),
                    f"delta yes:{delta_yes:.5f}",
                    f"delta no:{delta_no:.5f}",
                )

# %%
