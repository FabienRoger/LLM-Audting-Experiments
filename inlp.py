from data_loading import ActivationsDataset

from tqdm import tqdm

from linear import get_linear_cut
from utils import orthonormalize


def inlp(ds: ActivationsDataset, n: int, loading_bar: bool = True):
    working_ds = ds
    n_directions = 32
    dirs = []

    g = tqdm(range(n_directions)) if loading_bar else range(n_directions)
    for i in g:
        d = get_linear_cut(working_ds)
        working_ds = working_ds.project(d)
        dirs.append(d)
    return orthonormalize(dirs)
