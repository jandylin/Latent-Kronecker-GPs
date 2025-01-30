import torch
import torch.distributions as dist
from torch import Tensor


def _rmse(Y_pred: Tensor, Y_true: Tensor):
    return torch.sqrt(((Y_pred - Y_true) ** 2).mean())


def _nll(Y_pred_mean: Tensor, Y_pred_std: Tensor, Y_true: Tensor):
    return -torch.mean(dist.Normal(Y_pred_mean, Y_pred_std).log_prob(Y_true))


def _r2_score(Y_pred: Tensor, Y_true: Tensor):
    ss_res = ((Y_true - Y_pred) ** 2).sum()
    ss_tot = ((Y_true - Y_true.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot


def calculate_metrics(Y_pred_mean: Tensor, Y_pred_std: Tensor, Y_true: Tensor):
    rmse = _rmse(Y_pred_mean, Y_true)
    nll = _nll(Y_pred_mean, Y_pred_std, Y_true)
    r2_score = _r2_score(Y_pred_mean, Y_true)
    return rmse, nll, r2_score
