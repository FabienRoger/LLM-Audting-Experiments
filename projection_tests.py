#%%
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from activation_utils import get_activations, get_all_activations

from data_loading import ActivationsDataset, PromptDataset

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

stud_layers = [model.transformer.h[l] for l in [6, 7, 8]]
act_ds = ActivationsDataset(activations, stud_layers)

# %%
