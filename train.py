from config import TrainConfig
from data import load_data, get_train_data_loader, get_test_data_loader
from models import get_model
from metrics import calculate_metrics
from linear_operator import settings
import hydra
import omegaconf
import os
import time
import datetime
import torch
import wandb


def wandb_name(cfg: TrainConfig):
    wandb_name = (
        f"{cfg.dataset_name}_{cfg.dataset_split}_{int(cfg.missing_ratio * 100)}%"
    )
    wandb_name += f"_{cfg.model_name}"

    return wandb_name


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: omegaconf.DictConfig):
    os.environ["WANDB_API_KEY"] = cfg.wandb_api_key

    with wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        settings=wandb.Settings(start_method="thread"),
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),
        mode="online" if cfg.wandb_logging else "disabled",
        name=wandb_name(cfg),
        resume="allow",
    ):
        wandb_cfg = omegaconf.OmegaConf.create(wandb.config.as_dict())
        cfg = omegaconf.OmegaConf.merge(cfg, wandb_cfg)
        _train(cfg)


def _train(cfg: TrainConfig):
    settings.verbose_linalg._default = cfg.verbose_linalg
    dtype = getattr(torch, cfg.dtype)
    device = torch.device(cfg.device)

    X, Y, T, mask_valid = load_data(
        cfg.dataset_name,
        cfg.dataset_split,
        cfg.missing_ratio,
        cfg.seed,
        dtype=dtype,
        device=device,
        standardize=cfg.standardize,
    )
    valid_ratio = mask_valid.sum() / mask_valid.numel()
    print(f"X: {X.shape}, Y: {Y.shape}, T: {T.shape}")
    print(f"valid_ratio: {100 * valid_ratio:.1f}%")
    print(f"missing_ratio: {100 * cfg.missing_ratio:.1f}%")
    print()
    Y_train = Y.clone()
    Y_train[~mask_valid] = torch.nan
    model, mll = get_model(X, Y_train, T, cfg)
    model.to(dtype=dtype, device=device)
    print(f"model name: {cfg.model_name}")
    print()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    model.train()
    with model.context_manager():
        train_time = 0.0
        for i in range(cfg.n_epochs):
            train_data_loader = get_train_data_loader(X, Y_train, T, cfg)
            for j, (X_batch, Y_batch) in enumerate(train_data_loader):
                print(i, j, X_batch.shape, Y_batch.shape)
                start_time_train = time.monotonic()
                optimizer.zero_grad()
                output = model(X_batch)
                loss = -mll(output, Y_batch)
                loss.backward()
                optimizer.step()
                end_time_train = time.monotonic()
                train_time += end_time_train - start_time_train

                print(
                    f"Epoch {i + 1}/{cfg.n_epochs}, "
                    f"Iter {j + 1}/{len(train_data_loader)} - "
                    f"Loss: {loss.item():.4f}, "
                    f"Train Time: {datetime.timedelta(seconds=train_time)}"
                )
                if cfg.wandb_logging:
                    wandb.log(
                        {
                            "train/loss": loss.item(),
                            "train/train_time": train_time,
                        }
                    )
    train_memory = torch.cuda.max_memory_allocated() if cfg.device == "cuda" else 0
    if cfg.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print()
    test_data_loader = get_test_data_loader(X, T, cfg)
    model.eval()
    predict_time = time.monotonic()
    Y_pred_mean, Y_pred_stddev = [], []
    for X_batch in test_data_loader:
        with model.context_manager(), torch.no_grad():
            Y_pred_mean_batch, Y_pred_stddev_batch = model.predict(X_batch)
        Y_pred_mean.append(Y_pred_mean_batch)
        Y_pred_stddev.append(Y_pred_stddev_batch)
    Y_pred_mean = torch.cat(Y_pred_mean)
    Y_pred_stddev = torch.cat(Y_pred_stddev)
    predict_time = time.monotonic() - predict_time
    predict_memory = torch.cuda.max_memory_allocated() if cfg.device == "cuda" else 0

    train_rmse, train_nll, train_r2_score = calculate_metrics(
        Y_pred_mean[mask_valid.view(-1)],
        Y_pred_stddev[mask_valid.view(-1)],
        Y[mask_valid],
    )
    test_rmse, test_nll, test_r2_score = calculate_metrics(
        Y_pred_mean[~mask_valid.view(-1)],
        Y_pred_stddev[~mask_valid.view(-1)],
        Y[~mask_valid],
    )
    print(f"Train RMSE: {train_rmse.item()}")
    print(f"Train NLL : {train_nll.item()}")
    print(f"Train R2  : {train_r2_score.item()}")
    print()
    print(f"Test  RMSE: {test_rmse.item()}")
    print(f"Test  NLL : {test_nll.item()}")
    print(f"Test  R2  : {test_r2_score.item()}")
    print()
    print(f"Train   Time: {datetime.timedelta(seconds=train_time)}")
    print(f"Predict Time: {datetime.timedelta(seconds=predict_time)}")
    print()
    print(f"Train   Memory: {train_memory} bytes")
    print(f"Predict Memory: {predict_memory} bytes")
    if cfg.wandb_logging:
        wandb.log(
            {
                "final/train_rmse": train_rmse.item(),
                "final/train_nll": train_nll.item(),
                "final/train_r2": train_r2_score.item(),
                "final/test_rmse": test_rmse.item(),
                "final/test_nll": test_nll.item(),
                "final/test_r2": test_r2_score.item(),
                "final/train_time": train_time,
                "final/predict_time": predict_time,
                "final/train_memory": train_memory,
                "final/predict_memory": predict_memory,
            }
        )

    for name, param in model.named_parameters():
        print(f"\n{name}:\n{param.data}")


if __name__ == "__main__":
    main()
