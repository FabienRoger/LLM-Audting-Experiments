from data_loading import ActivationsDataset

from tqdm import tqdm

from linear import get_linear_cut
from utils import orthonormalize


def inlp(
    ds: ActivationsDataset, n: int, loading_bar: bool = True, max_iters: int = 1_000
):
    working_ds = ds
    dirs = []

    g = tqdm(range(n)) if loading_bar else range(n)
    for i in g:
        d, acc = get_linear_cut(working_ds, max_iters)
        if i == 0:
            working_ds = working_ds.project(d)
        else:
            working_ds.project_(d)
        dirs.append(d)

        if loading_bar:
            g.set_description(f"Acc = {acc:.2f}")
    return orthonormalize(dirs)
