from typing import Tuple
from sklearn.svm import LinearSVC
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from torchmetrics import Accuracy
from data_loading import ActivationsDataset


def get_linear_cut(
    ds: ActivationsDataset, max_iters: int = 1_000
) -> Tuple[torch.Tensor, float]:
    return get_linear_cut_svc(ds, max_iters)


def fit_model(m, ds, max_iters):
    loss_obj = torch.nn.BCELoss()  # binary cross entropy
    optimizer = torch.optim.Adam(m.parameters(), lr=3e-4, weight_decay=1e-4)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)

    for epoch in range(max_iters):
        epoch_loss = 0.0  # sum of avg loss/item/batch

        for (batch_idx, (X, y)) in enumerate(dataloader):
            optimizer.zero_grad()
            oupt = m(X)

            loss_val = loss_obj(oupt, y[:, 0:1].float())  # a tensor
            epoch_loss += loss_val.item()  # accumulate
            loss_val.backward()  # compute all gradients
            optimizer.step()  # update all wts, biases
        print(epoch_loss, end=" ")


class SimpleModel(torch.nn.Module):
    def __init__(self, input_dimension):
        super().__init__()
        self.linear = torch.nn.Linear(input_dimension, 1)
        self.act = torch.nn.Sigmoid()

    def forward(self, input_dimension):
        return self.act(self.linear(input_dimension))


def get_linear_cut_torch(
    ds: ActivationsDataset, max_iters: int
) -> Tuple[torch.Tensor, float]:
    # Doesn't work yet?
    m = SimpleModel(ds.x_data.shape[-1]).to(ds.x_data.device)
    fit_model(m, ds, max_iters)
    preds = m(ds.x_data) > 0.5

    return (m.linear.weight.detach()[0], torch.mean((preds == ds.y_data[:, 1]).float()))


def get_linear_cut_svc(
    ds: ActivationsDataset, max_iters: int
) -> Tuple[torch.Tensor, float]:
    classifier = LinearSVC(
        penalty="l2",
        C=0.01,
        fit_intercept=True,
        class_weight=None,
        dual=False,
        max_iter=max_iters,
    )
    if ds.y_data.shape[1] == 2:
        classifier.fit(ds.x_data, ds.y_data[:, 1])
        acc = accuracy_score(classifier.predict(ds.x_data), ds.y_data[:, 1])
        return torch.Tensor(np.array(classifier.coef_))[0], acc
    else:
        raise NotImplementedError(ds.y_data.shape[1])
