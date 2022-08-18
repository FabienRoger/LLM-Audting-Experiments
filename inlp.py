from activation_ds import ActivationsDataset

from tqdm import tqdm

from linear import get_linear_cut
from utils import orthonormalize


def inlp(
    ds: ActivationsDataset,
    n: int,
    loading_bar: bool = True,
    max_iters: int = 1_000,
    use_torch: bool = False,
):
    working_ds = ds
    dirs = []

    if use_torch:
        loading_bar = False

    g = tqdm(range(n)) if (loading_bar and not use_torch) else range(n)
    for i in g:
        d, acc = get_linear_cut(working_ds, max_iters, use_torch, loading_bar)
        if i == 0:
            working_ds = working_ds.project(d)
        else:
            working_ds.project_(d)
        dirs.append(d)

        if loading_bar and not use_torch:
            g.set_description(f"Acc = {acc:.2f}")
        else:
            print(f"Acc = {acc:.2f}")
    return orthonormalize(dirs)
