from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import (
    ExactMarginalLogLikelihood,
    VariationalELBO,
)
from gpytorch.models import ApproximateGP, ExactGP
from gpytorch.means import ZeroMean, MultitaskMean
from gpytorch.kernels import (
    Kernel,
    ScaleKernel,
    RBFKernel,
    IndexKernel,
    PeriodicKernel,
    ProductKernel,
)
from gpytorch.distributions import MultivariateNormal
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    MeanFieldVariationalDistribution,
    VariationalStrategy,
    NNVariationalStrategy,
)
from linear_operator import settings
from linear_operator.operators import (
    ConstantDiagLinearOperator,
    DenseLinearOperator,
    MaskedLinearOperator,
    KroneckerProductLinearOperator,
)
from computation_aware_gp import (
    ComputationAwareGP,
    ComputationAwareELBO,
    CaGPKernelLCBench,
    CaGPKernelNGCD,
)
import gpytorch.settings as gpytorch_settings
import contextlib
import torch
from torch import Tensor
from config import TrainConfig
from data import factors_to_product


def _context_manager(
    tol: float = 0.01,
    max_iter: int = 10000,
    covar_root_decomposition: bool = False,
    log_prob: bool = False,
    solves: bool = False,
    fast_pred_var: bool = False,
    fast_pred_samples: bool = False,
    precond_size: int = 100,
):
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            settings.fast_computations(
                covar_root_decomposition=covar_root_decomposition,
                log_prob=log_prob,
                solves=solves,
            )
        )
        stack.enter_context(settings.cg_tolerance(tol))
        stack.enter_context(settings.max_cg_iterations(max_iter))
        stack.enter_context(gpytorch_settings.fast_pred_var(fast_pred_var))
        stack.enter_context(gpytorch_settings.fast_pred_samples(fast_pred_samples))
        stack.enter_context(gpytorch_settings.max_preconditioner_size(precond_size))
        return stack.pop_all()


class Exact(ExactGP):
    def __init__(
        self,
        X_train: Tensor,
        Y_train: Tensor,
        T: Tensor,
        data_covar_module: Kernel,
        task_covar_module: Kernel,
        likelihood: GaussianLikelihood,
    ):
        self.idx_valid = ~torch.isnan(Y_train.view(-1))
        super(Exact, self).__init__(
            X_train,
            Y_train.view(-1)[self.idx_valid],
            likelihood,
        )
        self.T = T
        self.mean_module = ZeroMean()
        data_covar_module.active_dims = torch.tensor(
            tuple(range(X_train.shape[-1])), dtype=torch.long
        )
        task_covar_module.active_dims = torch.tensor(
            (X_train.shape[-1],), dtype=torch.long
        )
        self.covar_module = ProductKernel(data_covar_module, task_covar_module)

    def forward(self, x: Tensor):
        if self.training:
            idx = self.idx_valid
        else:
            idx = torch.ones(
                x.shape[-2] * self.T.shape[-1],
                dtype=bool,
                device=x.device,
            )
            idx[: self.idx_valid.shape[-1]] = self.idx_valid

        x = factors_to_product(x, self.T)[idx]
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)

    def predict(self, x: Tensor):
        posterior = self.likelihood(self(x))
        return posterior.mean, posterior.stddev

    def context_manager(self):
        return _context_manager(log_prob=True, solves=True)


class LKGP(ExactGP):
    def __init__(
        self,
        X_train: Tensor,
        Y_train: Tensor,
        T: Tensor,
        data_covar_module: Kernel,
        task_covar_module: Kernel,
        likelihood: GaussianLikelihood,
    ):
        self.idx_valid = ~torch.isnan(Y_train.view(-1))
        super(LKGP, self).__init__(
            X_train,
            Y_train.view(-1)[self.idx_valid],
            likelihood,
        )
        self.T = T
        self.mean_module = MultitaskMean(ZeroMean(), num_tasks=Y_train.shape[-1])
        self.data_covar_module = data_covar_module
        self.task_covar_module = task_covar_module

    def forward(self, x: Tensor):
        if self.training:
            idx = self.idx_valid
        else:
            idx = torch.ones(
                x.shape[-2] * self.T.shape[-1],
                dtype=bool,
                device=x.device,
            )
            idx[: self.idx_valid.shape[-1]] = self.idx_valid

        mean = self.mean_module(x).view(-1)[idx]
        data_covar = self.data_covar_module(x)
        task_covar = self.task_covar_module(self.T)
        covar = KroneckerProductLinearOperator(data_covar, task_covar)
        covar = MaskedLinearOperator(covar, row_mask=idx, col_mask=idx)
        return MultivariateNormal(mean, covar)

    def predict(self, x: Tensor):
        samples = self._sample_posterior(x)
        f_pred_var, Y_pred_mean = torch.var_mean(samples, dim=0)
        Y_pred_std = torch.sqrt(f_pred_var + self.likelihood.noise_covar.noise)
        return Y_pred_mean, Y_pred_std

    def context_manager(self):
        return _context_manager(log_prob=True, solves=True)

    def _sample_posterior(self, x: Tensor, n_samples: int = 64):
        X_train: Tensor = self.train_inputs[0]
        Y_train: Tensor = self.train_targets
        n_train_full = X_train.shape[-2] * self.T.shape[-1]
        n_train = Y_train.shape[-1]
        n_test = x.shape[-2] * self.T.shape[-1]

        eps_base = torch.randn(n_samples, n_train, dtype=x.dtype, device=x.device)
        w_train = torch.randn(n_samples, n_train_full, dtype=x.dtype, device=x.device)
        w_test = torch.randn(n_samples, n_test, dtype=x.dtype, device=x.device)

        eps = torch.sqrt(self.likelihood.noise_covar.noise) * eps_base

        K_T = self.task_covar_module(self.T.unsqueeze(-1))

        # Evaluate prior mean at training data
        m_train = self.mean_module(X_train).view(-1)[self.idx_valid]

        # Calculate prior sample
        K_train_train_X = self.data_covar_module(X_train)
        L_train_train_X = K_train_train_X.cholesky(upper=False)
        L_T = K_T.cholesky(upper=False)

        L_train_train = KroneckerProductLinearOperator(L_train_train_X, L_T)

        f_prior_train = L_train_train @ w_train.unsqueeze(-1)
        f_prior_train = m_train + f_prior_train.squeeze(-1)[..., self.idx_valid]

        K_train_train = KroneckerProductLinearOperator(K_train_train_X, K_T)
        K_train_train = MaskedLinearOperator(
            K_train_train,
            row_mask=self.idx_valid,
            col_mask=self.idx_valid,
        )
        noise_covar = ConstantDiagLinearOperator(
            torch.tensor(
                [self.likelihood.noise_covar.noise],
                dtype=x.dtype,
                device=x.device,
            ),
            diag_shape=n_train,
        )
        H = K_train_train + noise_covar

        v: Tensor = self.train_targets - (f_prior_train + eps)
        # Expand once here to avoid repeated expansion by MaskedLinearOperator later
        H_inv_v = torch.zeros(n_samples, n_train_full, dtype=x.dtype, device=x.device)
        H_inv_v[..., self.idx_valid] = H.solve(v.unsqueeze(-1)).squeeze(-1)

        # Evaluate prior mean at test data
        m_test = self.mean_module(x).view(-1)

        K_train_test_X: DenseLinearOperator = self.data_covar_module(
            X_train, x
        ).evaluate_kernel()
        K_test_test_X: DenseLinearOperator = self.data_covar_module(x).evaluate_kernel()

        L_train_test_X = L_train_train_X.solve_triangular(
            K_train_test_X.tensor, upper=False
        )
        L_test_test_X = (
            K_test_test_X - L_train_test_X.transpose(-2, -1) @ L_train_test_X
        ).cholesky(upper=False)

        L_test_train = KroneckerProductLinearOperator(
            L_train_test_X.transpose(-2, -1), L_T
        )

        L_test_test = KroneckerProductLinearOperator(L_test_test_X, L_T)

        f_prior_test = L_test_train @ w_train.unsqueeze(-1)
        f_prior_test = f_prior_test + L_test_test @ w_test.unsqueeze(-1)
        f_prior_test = m_test + f_prior_test.squeeze(-1)

        K_train_test = KroneckerProductLinearOperator(K_train_test_X, K_T)
        # no MaskedLinearOperator here because H_inv_v is already expanded
        samples = K_train_test.transpose(-2, -1) @ H_inv_v.unsqueeze(-1)
        samples = samples + f_prior_test.unsqueeze(-1)
        return samples.squeeze(-1)


class SVGP(ApproximateGP):
    def __init__(
        self,
        X_train: Tensor,
        Y_train: Tensor,
        T: Tensor,
        data_covar_module: Kernel,
        task_covar_module: Kernel,
        likelihood: GaussianLikelihood,
        n_inducing_points: int,
    ):
        self.n_inducing_points = n_inducing_points
        self.idx_valid = ~torch.isnan(Y_train.view(-1))
        self.n_valid = self.idx_valid.sum()
        variational_distribution = CholeskyVariationalDistribution(n_inducing_points)
        if n_inducing_points >= self.n_valid:
            raise ValueError(
                f"Number of inducing points {n_inducing_points} exceeds "
                f"number of valid training examples {self.n_valid}."
            )
        idx = torch.randperm(self.n_valid)[:n_inducing_points]
        inducing_points = factors_to_product(X_train, T)[self.idx_valid][idx]
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super(SVGP, self).__init__(variational_strategy)
        self.T = T
        self.mean_module = ZeroMean()
        data_covar_module.active_dims = torch.tensor(
            tuple(range(X_train.shape[-1])), dtype=torch.long
        )
        task_covar_module.active_dims = torch.tensor(
            (X_train.shape[-1],), dtype=torch.long
        )
        self.covar_module = ProductKernel(data_covar_module, task_covar_module)
        self.likelihood = likelihood

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)

    def predict(self, x: Tensor):
        posterior = self.likelihood(self(x))
        return posterior.mean, posterior.stddev

    def context_manager(self):
        return _context_manager()


class VNNGP(ApproximateGP):
    def __init__(
        self,
        X_train: Tensor,
        Y_train: Tensor,
        T: Tensor,
        data_covar_module: Kernel,
        task_covar_module: Kernel,
        likelihood: GaussianLikelihood,
        k: int,
        max_n_inducing_points: int,
        batch_size: int,
    ):
        self.idx_valid = ~torch.isnan(Y_train.view(-1))
        self.n_valid = self.idx_valid.sum()
        # set inducing points to be the training data
        inducing_points = factors_to_product(X_train, T)[self.idx_valid]
        if inducing_points.shape[-2] > max_n_inducing_points:
            idx = torch.randperm(inducing_points.shape[-2])[:max_n_inducing_points]
            inducing_points = inducing_points[..., idx, :]

        self.m = inducing_points.shape[-2]
        self.k = k

        variational_distribution = MeanFieldVariationalDistribution(self.m)
        variational_strategy = NNVariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            k=k,
            training_batch_size=batch_size,
        )
        super(VNNGP, self).__init__(variational_strategy)
        self.T = T
        self.mean_module = ZeroMean()
        data_covar_module.active_dims = torch.tensor(
            tuple(range(X_train.shape[-1])), dtype=torch.long
        )
        task_covar_module.active_dims = torch.tensor(
            (X_train.shape[-1],), dtype=torch.long
        )
        self.covar_module = ProductKernel(data_covar_module, task_covar_module)
        self.likelihood = likelihood

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)

    def predict(self, x: Tensor):
        posterior = self.likelihood(self(x))
        return posterior.mean, posterior.stddev

    def context_manager(self):
        return _context_manager()


class CaGP(ComputationAwareGP):
    def __init__(
        self,
        X_train: Tensor,
        Y_train: Tensor,
        T: Tensor,
        custom_covar_module: Kernel,
        likelihood: GaussianLikelihood,
        projection_dim: int,
    ):
        self.idx_valid = ~torch.isnan(Y_train.view(-1))
        self.T = T

        train_inputs = factors_to_product(X_train, T)[self.idx_valid]
        train_targets = Y_train.view(-1)[self.idx_valid]

        super().__init__(
            train_inputs,
            train_targets,
            ZeroMean(),
            custom_covar_module,
            likelihood,
            projection_dim,
        )

    def predict(self, x: Tensor):
        posterior = self.likelihood(self(x))
        return posterior.mean, posterior.stddev

    def context_manager(self):
        return _context_manager()


def get_model(X_train: Tensor, Y_train: Tensor, T: Tensor, cfg: TrainConfig):
    data_covar_module, task_covar_module = _get_covar_modules(cfg, X_train, Y_train)
    likelihood = GaussianLikelihood()
    if cfg.model_name == "exact":
        model = Exact(
            X_train,
            Y_train,
            T,
            data_covar_module,
            task_covar_module,
            likelihood,
        )
        mll = ExactMarginalLogLikelihood(likelihood, model)
    elif cfg.model_name == "lkgp":
        model = LKGP(
            X_train,
            Y_train,
            T,
            data_covar_module,
            task_covar_module,
            likelihood,
        )
        mll = ExactMarginalLogLikelihood(likelihood, model)
    elif cfg.model_name == "svgp":
        model = SVGP(
            X_train,
            Y_train,
            T,
            data_covar_module,
            task_covar_module,
            likelihood,
            cfg.svgp_n_inducing_points,
        )
        mll = VariationalELBO(likelihood, model, num_data=model.n_valid)
    elif cfg.model_name == "vnngp":
        model = VNNGP(
            X_train,
            Y_train,
            T,
            data_covar_module,
            task_covar_module,
            likelihood,
            cfg.vnngp_k,
            cfg.vnngp_max_n_inducing_points,
            cfg.batch_size,
        )
        mll = VariationalELBO(likelihood, model, num_data=model.n_valid)
    elif cfg.model_name == "cagp":
        model = CaGP(
            X_train,
            Y_train,
            T,
            data_covar_module,
            likelihood,
            cfg.cagp_projection_dim,
        )
        mll = ComputationAwareELBO(likelihood, model)
    else:
        raise ValueError(f"Unknown model name: {cfg.model_name}")

    return model, mll


def _get_covar_modules(cfg: TrainConfig, X_train: Tensor, Y_train: Tensor):
    if cfg.dataset_name == "sarcos":
        if cfg.model_name == "cagp":
            raise ValueError("CaGP is not supported for SARCOS.")
        else:
            num_tasks = Y_train.shape[-1]
            data_covar_module = ScaleKernel(RBFKernel(ard_num_dims=X_train.shape[-1]))
            task_covar_module = IndexKernel(num_tasks=num_tasks, rank=num_tasks)
    elif cfg.dataset_name.startswith("lcbench_"):
        if cfg.model_name == "cagp":
            data_covar_module = CaGPKernelLCBench(ard_num_dims=X_train.shape[-1])
            task_covar_module = None
        else:
            data_covar_module = ScaleKernel(RBFKernel(ard_num_dims=X_train.shape[-1]))
            task_covar_module = RBFKernel(ard_num_dims=1)
    elif cfg.dataset_name in {"ngcd_tg", "ngcd_rr"}:
        if cfg.model_name == "cagp":
            data_covar_module = CaGPKernelNGCD(ard_num_dims=X_train.shape[-1])
            task_covar_module = None
        else:
            data_covar_module = ScaleKernel(RBFKernel(ard_num_dims=X_train.shape[-1]))
            task_covar_module = RBFKernel(ard_num_dims=1) * PeriodicKernel()
    else:
        raise ValueError(f"Unknown dataset name: {cfg.dataset_name}")
    return data_covar_module, task_covar_module
