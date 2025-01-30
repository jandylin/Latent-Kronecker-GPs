import numpy as np
import torch
import scipy.io as sio
from config import TrainConfig
from torch import Tensor


def _mask_valid_uniform(missing_ratio: float, shape: tuple[int]):
    # Create mask with `missing_ratio` of values missing uniformly at random.
    n_observed = np.prod(shape)
    n_missing = int(missing_ratio * n_observed)
    idx_missing = np.random.choice(n_observed, n_missing, replace=False)
    mask_valid = np.ones(shape, dtype=bool)
    mask_valid.flat[idx_missing] = False
    return mask_valid


def _mask_valid_trailing(missing_ratio: float, shape: tuple[int]):
    # Create mask with (1 - `missing_ratio`) fully observed time series
    # and `missing_ratio` time series with missing values at the end.
    # Number of missing values per time series is sampled uniformly at random.
    n_missing = int(missing_ratio * shape[0])
    idx_missing = np.random.choice(shape[0], n_missing, replace=False)
    mask_valid = np.ones(shape, dtype=bool)
    for i in idx_missing:
        j = np.random.randint(shape[1])
        mask_valid[i, j:] = False
    return mask_valid


def _load_sarcos(missing_ratio: float, N: int = 5000):
    # data: 44484 x 28, subsample to 5000 x 28
    # 21 inputs: 7 positions, 7 velocities, 7 accelerations
    # 7 outputs: 7 torques
    data: np.ndarray = sio.loadmat("./data/sarcos_inv.mat")["sarcos_inv"]
    idx = np.random.permutation(data.shape[0])[:N]
    d = 21
    X, Y = data[idx, :d], data[idx, d:]
    T = np.arange(Y.shape[-1])
    mask_valid = _mask_valid_uniform(missing_ratio, Y.shape)
    return X, Y, T, mask_valid


def _load_lcbench(missing_ratio: float, dataset_name: str):
    # X: 2000 x 7, hyperparameter configurations
    # Y: 2000 x 52, validation loss learning curves
    data = np.load(f"./data/LCBench/{dataset_name}.npz")
    X: np.ndarray = data["X"]
    Y: np.ndarray = data["Y"]
    T = np.log(np.arange(1, Y.shape[-1] + 1)) / np.log(Y.shape[-1] + 1)
    mask_valid = _mask_valid_trailing(missing_ratio, Y.shape)
    return X, Y, T, mask_valid


def _load_ngcd(missing_ratio: float, data_type: str):
    # X: 5000 x 2, latitude and longitude of spatial locations
    # Y: 5000 x 1000, temperature or precipitation over 1000 days
    data = np.load(f"./data/NGCD_{data_type}.npz")
    X: np.ndarray = data["X"]
    Y: np.ndarray = data["Y"]
    T = np.arange(Y.shape[-1])
    mask_valid = _mask_valid_uniform(missing_ratio, Y.shape)
    return X, Y, T, mask_valid


def load_data(
    dataset_name: str,
    dataset_split: int,
    missing_ratio: float,
    seed: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
    standardize: bool = True,
):
    np.random.seed(seed + dataset_split)
    torch.manual_seed(seed + dataset_split)
    # load data into numpy arrays
    if dataset_name == "sarcos":
        X, Y, T, mask_valid = _load_sarcos(missing_ratio)
    elif dataset_name.startswith("lcbench_"):
        X, Y, T, mask_valid = _load_lcbench(missing_ratio, dataset_name[8:])
    elif dataset_name == "ngcd_tg":
        X, Y, T, mask_valid = _load_ngcd(missing_ratio, "TG")
    elif dataset_name == "ngcd_rr":
        X, Y, T, mask_valid = _load_ngcd(missing_ratio, "RR")
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    # convert to torch tensors
    X = torch.from_numpy(X).to(dtype=dtype, device=device)
    Y = torch.from_numpy(Y).to(dtype=dtype, device=device)
    T = torch.from_numpy(T).to(dtype=dtype, device=device)
    mask_valid = torch.from_numpy(mask_valid).to(device=device)

    # calculate z-zscores and standardize
    if standardize:
        std_X, mean_X = torch.std_mean(X, dim=0)
        if dataset_name.startswith("lcbench_"):
            # standardize using observed final values of time series
            std_Y, mean_Y = torch.std_mean(Y[mask_valid[:, -1], -1])
        else:
            # standardize using all observed values
            std_Y, mean_Y = torch.std_mean(Y[mask_valid])
        X = (X - mean_X) / std_X
        Y = (Y - mean_Y) / std_Y

    return X, Y, T, mask_valid


def get_train_data_loader(
    X_train: Tensor,
    Y_train: Tensor,
    T: Tensor,
    cfg: TrainConfig,
    shuffle: bool = True,
):
    idx_valid = ~torch.isnan(Y_train.view(-1))
    if cfg.model_name in {"exact", "lkgp"}:
        # return the whole dataset without missing values as single batch
        return ((X_train, Y_train.view(-1)[idx_valid]),)
    elif cfg.model_name in {"svgp", "vnngp"}:
        # return mini-batches
        X_batches = factors_to_product(X_train, T)[idx_valid]
        Y_batches = Y_train.view(-1)[idx_valid]
        if shuffle:
            idx_shuffle = torch.randperm(idx_valid.sum())
            X_batches, Y_batches = X_batches[idx_shuffle], Y_batches[idx_shuffle]
        X_batches = X_batches.split(cfg.batch_size)
        Y_batches = Y_batches.split(cfg.batch_size)
        return list(zip(X_batches, Y_batches, strict=True))
    elif cfg.model_name == "cagp":
        return (
            (factors_to_product(X_train, T)[idx_valid], Y_train.view(-1)[idx_valid]),
        )
    else:
        raise ValueError(f"Unknown model name: {cfg.model_name}")


def get_test_data_loader(X_test: Tensor, T: Tensor, cfg: TrainConfig):
    if cfg.model_name in {"exact", "lkgp"}:
        # return the whole dataset as single batch
        return (X_test,)
    elif cfg.model_name in {"svgp", "vnngp"}:
        # return the dataset in mini-batches
        return factors_to_product(X_test, T).split(cfg.batch_size)
    elif cfg.model_name == "cagp":
        return (factors_to_product(X_test, T),)
    else:
        raise ValueError(f"Unknown model name: {cfg.model_name}")


def factors_to_product(X: Tensor, T: Tensor):
    """
    Args: X, T: torch.Tensor of shape (n, d), (m,)
    Returns: X_hat: torch.Tensor of shape (n * m, d + 1)
    """
    X_hat = torch.cat(
        [
            X.repeat_interleave(T.shape[0], dim=0),
            T.repeat(X.shape[0]).view(-1, 1),
        ],
        dim=-1,
    )
    return X_hat
