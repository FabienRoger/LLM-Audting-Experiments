#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
df = pd.read_csv("results/comparison_3ds_all_methods.csv", index_col=None)
# %%
studied_layers_posibilities = list(df.studied_layer.unique())
for sl in studied_layers_posibilities:
    selected = (df.studied_layer == sl) & (df.proj_strength != "default")
    bias = df[selected].avg_delta.mean()
    quality_deg = df[selected].irrelevant_modifications.mean()
    print(f"{sl} {bias=:.3f}, {quality_deg=:.3f}")
# %%
inlp_posibilities = list(df.proj_method.unique())
for pm in inlp_posibilities:
    selected = (
        (df.proj_method == pm)
        & (df.proj_strength != "default")
        & ~((df.proj_method == "mlp") & (df.proj_strength != "proj8"))
    )
    bias = df[selected].avg_delta.mean()
    quality_deg = df[selected].irrelevant_modifications.mean()
    print(f"{pm} {bias=:.3f}, {quality_deg=:.3f}")
# %%
proj_strength_posibilities = list(df.proj_strength.unique())
for ps in proj_strength_posibilities:
    selected = df.proj_strength == ps
    bias = df[selected].avg_delta.mean()
    quality_deg = df[selected].irrelevant_modifications.mean()
    print(f"{ps} {bias=:.3f}, {quality_deg=:.3f}")
# %%

proj_strength_posibilities = list(df.proj_strength.unique())
for ps in proj_strength_posibilities:
    selected = df.proj_strength == ps
    bias = df[selected].avg_delta.mean()
    quality_deg = df[selected].irrelevant_modifications.mean()
    print(f"{ps} {bias=:.3f}, {quality_deg=:.3f}")
#%%
r_shape = (len(proj_strength_posibilities), len(inlp_posibilities))
r_bias = np.zeros(r_shape)
r_degrad = np.zeros(r_shape)
for i, ps in enumerate(proj_strength_posibilities):
    for j, pm in enumerate(inlp_posibilities):
        selected = (df.proj_strength == ps) & (df.proj_method == pm)
        r_bias[i, j] = df[selected].avg_delta.mean()
        r_degrad[i, j] = df[selected].irrelevant_modifications.mean()
plt.imshow(r_bias)
plt.xticks(ticks=np.arange(r_shape[1]), labels=inlp_posibilities)
plt.yticks(ticks=np.arange(r_shape[0]), labels=proj_strength_posibilities)
plt.colorbar()
plt.title("bias")
plt.show()
plt.imshow(r_degrad)
plt.xticks(ticks=np.arange(r_shape[1]), labels=inlp_posibilities)
plt.yticks(ticks=np.arange(r_shape[0]), labels=proj_strength_posibilities)
plt.title("change in unrelated responses")
plt.colorbar()
# %%
r_shape = (len(proj_strength_posibilities), len(studied_layers_posibilities))
r_bias = np.zeros(r_shape)
r_degrad = np.zeros(r_shape)
for i, ps in enumerate(proj_strength_posibilities):
    for j, sl in enumerate(studied_layers_posibilities):
        selected = (df.proj_strength == ps) & (df.studied_layer == sl)
        r_bias[i, j] = df[selected].avg_delta.mean()
        r_degrad[i, j] = df[selected].irrelevant_modifications.mean()
plt.imshow(r_bias)
plt.xticks(ticks=np.arange(r_shape[1]), labels=studied_layers_posibilities)
plt.yticks(ticks=np.arange(r_shape[0]), labels=proj_strength_posibilities)
plt.colorbar()
plt.title("bias")
plt.show()
plt.imshow(r_degrad)
plt.xticks(ticks=np.arange(r_shape[1]), labels=studied_layers_posibilities)
plt.yticks(ticks=np.arange(r_shape[0]), labels=proj_strength_posibilities)
plt.title("change in unrelated responses")
plt.colorbar()
# %%
