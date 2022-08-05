from sklearn.svm import LinearSVC
import torch
import numpy as np

from data_loading import ActivationsDataset


def get_linear_cut(ds: ActivationsDataset) -> torch.Tensor:
    classifier = LinearSVC(loss="hinge", random_state=0)
    if ds.y_data.shape[1] == 2:
        classifier.fit(ds.x_data, ds.y_data[:, 1])
        return torch.Tensor(np.array(classifier.coef_))[0]
    else:
        raise NotImplementedError(ds.y_data.shape[1])
