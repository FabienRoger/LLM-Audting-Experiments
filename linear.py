from typing import Tuple
from sklearn.svm import LinearSVC
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from data_loading import ActivationsDataset
from tqdm import tqdm


def get_linear_cut(
    ds: ActivationsDataset,
    max_iters: int = 1_000,
    use_torch: bool = False,
    loading_bar=True,
) -> Tuple[torch.Tensor, float]:
    return (get_linear_cut_torch if use_torch else get_linear_cut_svc)(
        ds, max_iters, loading_bar
    )


def fit_model(m, ds, max_iters, progress_bar: bool = True):
    loss_obj = torch.nn.BCELoss()  # binary cross entropy
    optimizer = torch.optim.Adam(m.parameters(), lr=1e-4, weight_decay=1e-4)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)

    tepoch = tqdm(range(max_iters), unit="epoch") if progress_bar else range(max_iters)

    for _ in tepoch:
        epoch_loss = 0.0  # sum of avg loss/item/batch

        for (batch_idx, (X, y)) in enumerate(dataloader):
            optimizer.zero_grad()
            oupt = m(X)

            loss_val = loss_obj(oupt, y[:, 1:2].float())  # a tensor
            epoch_loss += loss_val.item()  # accumulate
            loss_val.backward()  # compute all gradients
            optimizer.step()  # update all wts, biases

        if progress_bar:
            tepoch.set_postfix(loss=epoch_loss)


class MultiVecModel(torch.nn.Module):
    def __init__(self, input_dimension, n_vec, n_hid):
        super().__init__()
        self.linears = torch.nn.Linear(input_dimension, n_vec)
        self.agregator1 = torch.nn.Linear(n_vec, n_hid)
        self.agregator2 = torch.nn.Linear(n_hid, 1)
        self.act = torch.nn.ReLU()
        self.final_act = torch.nn.Sigmoid()

    def forward(self, x):
        h = self.linears(x)
        h = self.agregator1(h)
        h = self.act(h)
        h = self.agregator2(h)
        return self.final_act(h)


def get_multi_lin_cut(
    ds: ActivationsDataset,
    n_dirs: int,
    n_hid: int = 64,
    epochs: int = 300,
    progress_bar: bool = True,
):
    m = MultiVecModel(ds.x_data.shape[-1], n_dirs, n_hid).to(ds.x_data.device)
    fit_model(m, ds, epochs, progress_bar)
    preds = m(ds.x_data) > 0.5
    return m.linears.weight.detach(), torch.mean(
        (preds[:, 0] == ds.y_data[:, 1]).float()
    )


class SimpleModel(torch.nn.Module):
    def __init__(self, input_dimension):
        super().__init__()
        self.linear = torch.nn.Linear(input_dimension, 1)
        self.act = torch.nn.Sigmoid()

    def forward(self, input_dimension):
        return self.act(self.linear(input_dimension))


def get_linear_cut_torch(
    ds: ActivationsDataset, max_iters: int, loading_bar=True
) -> Tuple[torch.Tensor, float]:
    # Doesn't work yet?
    m = SimpleModel(ds.x_data.shape[-1]).to(ds.x_data.device)
    fit_model(m, ds, max_iters, loading_bar)
    preds = m(ds.x_data) > 0.5

    return m.linear.weight.detach()[0], torch.mean(
        (preds[:, 0] == ds.y_data[:, 1]).float()
    )


def get_linear_cut_svc(
    ds: ActivationsDataset,
    max_iters: int,
    loading_bar=False,
) -> Tuple[torch.Tensor, float]:
    x, y = ds.x_data.cpu(), ds.y_data[:, 1].cpu()
    classifier = LinearSVC(
        penalty="l2",
        C=0.01,
        fit_intercept=True,
        class_weight=None,
        dual=False,
        max_iter=max_iters,
    )
    if ds.y_data.shape[1] == 2:
        classifier.fit(x, y)
        acc = accuracy_score(classifier.predict(x), y)
        return torch.Tensor(np.array(classifier.coef_))[0], acc
    else:
        raise NotImplementedError(ds.y_data.shape[1])
