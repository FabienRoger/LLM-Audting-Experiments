from data_loading import ActivationsDataset

from tqdm import tqdm

from linear import get_linear_cut
from utils import orthonormalize


def inlp(ds: ActivationsDataset, n: int, loading_bar: bool = True):
    working_ds = ds
    dirs = []

    g = tqdm(range(n)) if loading_bar else range(n)
    for i in g:
        d = get_linear_cut(working_ds)
        working_ds = working_ds.project(d)
        dirs.append(d)
    return orthonormalize(dirs)
