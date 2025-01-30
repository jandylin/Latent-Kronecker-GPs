from typing import NamedTuple


class TrainConfig(NamedTuple):
    seed: int
    dtype: str
    device: str

    dataset_name: str
    dataset_split: int
    missing_ratio: float
    standardize: bool

    model_name: str
    learning_rate: float
    n_epochs: int

    svgp_n_inducing_points: int

    vnngp_k: int
    vnngp_max_n_inducing_points: int

    cagp_projection_dim: int

    batch_size: int
    verbose_linalg: bool

    wandb_logging: bool
    wandb_project: str
    wandb_entity: str
    wandb_api_key: str
