# adapted from
# https://github.com/cornellius-gp/linear_operator/tree/sparsity
# https://github.com/cornellius-gp/gpytorch/tree/computation-aware-gps-v2
import math
import torch
from typing import Union
from jaxtyping import Float
from torch import Tensor
from linear_operator.operators import (
    LinearOperator,
    AddedDiagLinearOperator,
    DiagLinearOperator,
)
from linear_operator import operators, utils as linop_utils
from gpytorch import kernels, likelihoods, means, settings
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import _GaussianLikelihoodBase, GaussianLikelihood
from gpytorch.mlls import MarginalLogLikelihood
from gpytorch.models import ExactGP
from pykeops.torch import LazyTensor


def _sq_dist_keops(
    x1: Float[Tensor, "*batch M D"], x2: Float[Tensor, "*batch N D"]
) -> Float[Tensor, "*batch M N"]:
    x1_ = LazyTensor(x1[..., :, None, :])
    x2_ = LazyTensor(x2[..., None, :, :])
    sq_dist = (x1_ - x2_).square().sum(-1)
    return sq_dist


class CaGPKernelLCBench(kernels.Kernel):
    def __init__(self, ard_num_dims: int):
        super().__init__()
        self.data_covar_module = kernels.ScaleKernel(
            kernels.RBFKernel(ard_num_dims=ard_num_dims)
        )
        self.task_covar_module = kernels.RBFKernel(ard_num_dims=1)

    def forward(self, x1: Tensor, x2: Tensor, **kwargs):
        data_covar = self.data_covar_module(x1[..., :-1], x2[..., :-1], **kwargs)
        task_covar = self.task_covar_module(x1[..., [-1]], x2[..., [-1]], **kwargs)
        return data_covar * task_covar

    def forward_keops(self, x1: Tensor, x2: Tensor):
        # RBFKernel
        data_covar_sq_dist = _sq_dist_keops(
            x1[..., :-1] / self.data_covar_module.base_kernel.lengthscale,
            x2[..., :-1] / self.data_covar_module.base_kernel.lengthscale,
        )
        data_covar_rbf = (data_covar_sq_dist / -2).exp()
        # ScaleKernel
        data_covar = self.data_covar_module.outputscale * data_covar_rbf
        # RBFKernel
        task_covar_sq_dist = _sq_dist_keops(
            x1[..., [-1]] / self.task_covar_module.lengthscale,
            x2[..., [-1]] / self.task_covar_module.lengthscale,
        )
        task_covar = (task_covar_sq_dist / -2).exp()
        return data_covar * task_covar


class CaGPKernelNGCD(kernels.Kernel):
    def __init__(self, ard_num_dims: int):
        super().__init__()
        self.data_covar_module = kernels.ScaleKernel(
            kernels.RBFKernel(ard_num_dims=ard_num_dims)
        )
        self.task_covar_module = (
            kernels.RBFKernel(ard_num_dims=1) * kernels.PeriodicKernel()
        )

    def forward(self, x1: Tensor, x2: Tensor, **kwargs):
        data_covar = self.data_covar_module(x1[..., :-1], x2[..., :-1], **kwargs)
        task_covar = self.task_covar_module(x1[..., [-1]], x2[..., [-1]], **kwargs)
        return data_covar * task_covar

    def forward_keops(self, x1: Tensor, x2: Tensor):
        # RBFKernel
        data_covar_sq_dist = _sq_dist_keops(
            x1[..., :-1] / self.data_covar_module.base_kernel.lengthscale,
            x2[..., :-1] / self.data_covar_module.base_kernel.lengthscale,
        )
        data_covar_rbf = (data_covar_sq_dist / -2).exp()
        # ScaleKernel
        data_covar = self.data_covar_module.outputscale * data_covar_rbf
        # RBFKernel
        task_covar_rbf_sq_dist = _sq_dist_keops(
            x1[..., [-1]] / self.task_covar_module.kernels[0].lengthscale,
            x2[..., [-1]] / self.task_covar_module.kernels[0].lengthscale,
        )
        task_covar_rbf = (task_covar_rbf_sq_dist / -2).exp()
        # PeriodicKernel
        task_covar_periodic_sq_dist = _sq_dist_keops(
            x1[..., [-1]] / (self.task_covar_module.kernels[1].period_length / math.pi),
            x2[..., [-1]] / (self.task_covar_module.kernels[1].period_length / math.pi),
        )
        task_covar_periodic_dist = (task_covar_periodic_sq_dist + 1e-20).sqrt()
        task_covar_periodic = (
            -2
            * task_covar_periodic_dist.sin() ** 2
            / self.task_covar_module.kernels[1].lengthscale[0, 0]
        ).exp()
        # ProductKernel
        task_covar = task_covar_rbf * task_covar_periodic
        return data_covar * task_covar


class BlockDiagonalSparseLinearOperator(LinearOperator):
    """A sparse linear operator (which when reordered) has dense blocks on its diagonal.

    Linear operator with a matrix representation that has sparse rows, with an equal number of
    non-zero entries per row. The non-zero entries are stored in a tensor of size M x NNZ, where M is
    the number of rows and NNZ is the number of non-zero entries per row. When appropriately re-ordering
    the columns of the matrix, it is a block-diagonal matrix.

    Note:
        This currently only supports equally sized blocks of size 1 x NNZ.

    :param non_zero_idcs: Tensor of non-zero indices.
    :param blocks: Tensor of non-zero entries.
    :param size_input_dim: Size of the (sparse) input dimension, equivalently the number of columns.
    """

    def __init__(
        self,
        non_zero_idcs: Float[torch.Tensor, "M NNZ"],
        blocks: Float[torch.Tensor, "M NNZ"],
        size_input_dim: int,
    ):
        super().__init__(non_zero_idcs, blocks, size_input_dim=size_input_dim)
        self.non_zero_idcs = torch.atleast_2d(non_zero_idcs)
        self.non_zero_idcs.requires_grad = False  # Ensure indices cannot be optimized
        self.blocks = torch.atleast_2d(blocks)
        self.size_input_dim = size_input_dim

    def _matmul(
        self: Float[LinearOperator, "*batch M N"],
        rhs: Float[torch.Tensor, "*batch2 N C"],
    ) -> Union[Float[torch.Tensor, "... M C"], Float[torch.Tensor, "... M"]]:
        # Workarounds for (Added)DiagLinearOperator
        # There seems to be a bug in DiagLinearOperator, which doesn't allow subsetting the way we do here.
        if isinstance(rhs, AddedDiagLinearOperator):
            return self._matmul(rhs._linear_op) + self._matmul(rhs._diag_tensor)

        if isinstance(rhs, DiagLinearOperator):
            return BlockDiagonalSparseLinearOperator(
                non_zero_idcs=self.non_zero_idcs,
                blocks=rhs.diag()[self.non_zero_idcs] * self.blocks,
                size_input_dim=self.size_input_dim,
            ).to_dense()

        # Subset rhs via index tensor
        rhs_non_zero = rhs[..., self.non_zero_idcs, :]

        if rhs.ndim == 2 and rhs.shape[-1] == 1:
            # Multiply and sum on sparse dimension
            return (self.blocks.unsqueeze(-1) * rhs_non_zero).sum(dim=-2)

        # Multiply on sparse dimension
        return (self.blocks.unsqueeze(-2) @ rhs_non_zero).squeeze(-2)

    def _size(self) -> torch.Size:
        return torch.Size((self.non_zero_idcs.shape[0], self.size_input_dim))

    def to_dense(self: LinearOperator) -> Tensor:
        if self.size() == self.blocks.shape:
            return self.blocks
        return torch.zeros(
            (self.blocks.shape[0], self.size_input_dim),
            dtype=self.blocks.dtype,
            device=self.blocks.device,
        ).scatter_(src=self.blocks, index=self.non_zero_idcs, dim=1)


class ComputationAwareGP(ExactGP):
    """Computation-aware Gaussian process."""

    def __init__(
        self,
        train_inputs: torch.Tensor,
        train_targets: torch.Tensor,
        mean_module: "means.Mean",
        covar_module: "kernels.Kernel",
        likelihood: "likelihoods.GaussianLikelihood",
        projection_dim: int,
        initialization: str = "random",
    ):
        # Set number of non-zero action entries such that num_non_zero * projection_dim = num_train_targets
        num_non_zero = train_targets.size(-1) // projection_dim

        super().__init__(
            # Training data is subset to satisfy the requirement: num_non_zero * projection_dim = num_train_targets
            train_inputs[0 : num_non_zero * projection_dim],
            train_targets[0 : num_non_zero * projection_dim],
            likelihood,
        )
        self.mean_module = mean_module
        self.covar_module = covar_module
        self.projection_dim = projection_dim
        self.num_non_zero = num_non_zero
        self.cholfac_gram_SKhatS = None

        non_zero_idcs = torch.arange(
            self.num_non_zero * projection_dim,
            device=train_inputs.device,
        ).reshape(self.projection_dim, -1)

        # Initialization of actions
        if initialization == "random":
            # Random initialization
            self.non_zero_action_entries = torch.nn.Parameter(
                torch.randn_like(
                    non_zero_idcs,
                    dtype=train_inputs.dtype,
                    device=train_inputs.device,
                ).div(math.sqrt(self.num_non_zero))
            )
        elif initialization == "targets":
            # Initialize with training targets
            self.non_zero_action_entries = torch.nn.Parameter(
                train_targets.clone()[: self.num_non_zero * projection_dim].reshape(
                    self.projection_dim, -1
                )
            )
            self.non_zero_action_entries.div(
                torch.linalg.vector_norm(self.non_zero_action_entries, dim=1).reshape(
                    -1, 1
                )
            )
        elif initialization == "eigen":
            # Initialize via top eigenvectors of kernel submatrices
            with torch.no_grad():
                X = train_inputs.clone()[0 : num_non_zero * projection_dim].reshape(
                    projection_dim, num_non_zero, train_inputs.shape[-1]
                )
                K_sub_matrices = self.covar_module(X)
                _, evecs = torch.linalg.eigh(K_sub_matrices.to_dense())
            self.non_zero_action_entries = torch.nn.Parameter(evecs[:, -1])
        else:
            raise ValueError(f"Unknown initialization: '{initialization}'.")

        self.actions_op = BlockDiagonalSparseLinearOperator(
            non_zero_idcs=non_zero_idcs,
            blocks=self.non_zero_action_entries,
            size_input_dim=self.projection_dim * self.num_non_zero,
        )

    def __call__(self, x: torch.Tensor) -> MultivariateNormal:
        if self.training:
            # In training mode, just return the prior.
            return MultivariateNormal(
                self.mean_module(x),
                self.covar_module(x),
            )
        elif settings.prior_mode.on():
            # Prior mode
            return MultivariateNormal(
                self.mean_module(x),
                self.covar_module(x),
            )
        else:
            # Posterior mode
            if x.ndim == 1:
                x = torch.atleast_2d(x).mT

            K_lazy = self.covar_module.forward_keops(
                self.train_inputs[0].view(
                    self.projection_dim,
                    self.num_non_zero,
                    self.train_inputs[0].shape[-1],
                ),
                self.train_inputs[0].view(
                    self.projection_dim,
                    1,
                    self.num_non_zero,
                    self.train_inputs[0].shape[-1],
                ),
            )
            gram_SKS = (
                (
                    K_lazy
                    @ self.actions_op.blocks.view(
                        self.projection_dim, 1, self.num_non_zero, 1
                    )
                ).squeeze(-1)
                * self.actions_op.blocks
            ).sum(-1)

            StrS_diag = (self.actions_op.blocks**2).sum(
                -1
            )  # NOTE: Assumes orthogonal actions.
            gram_SKhatS = gram_SKS + torch.diag(self.likelihood.noise * StrS_diag)
            self.cholfac_gram_SKhatS = linop_utils.cholesky.psd_safe_cholesky(
                gram_SKhatS.to(dtype=torch.float64), upper=False
            )

            # Cross-covariance mapped to the low-dimensional space spanned by the actions: k(x, X)S
            covar_x_train_actions = (
                (
                    self.covar_module.forward_keops(
                        x,
                        self.train_inputs[0].view(
                            self.projection_dim,
                            self.num_non_zero,
                            self.train_inputs[0].shape[-1],
                        ),
                    )
                    @ self.actions_op.blocks.view(
                        self.projection_dim, self.num_non_zero, 1
                    )
                )
                .squeeze(-1)
                .mT
            )

            # Matrix-square root of the covariance downdate: k(x, X)L^{-1}
            covar_x_train_actions_cholfac_inv = torch.linalg.solve_triangular(
                self.cholfac_gram_SKhatS, covar_x_train_actions.mT, upper=False
            ).mT

            # "Projected" training data (with mean correction)
            actions_target = self.actions_op @ (
                self.train_targets - self.mean_module(self.train_inputs[0])
            )

            # Compressed representer weights
            compressed_repr_weights = (
                torch.cholesky_solve(
                    actions_target.unsqueeze(1).to(dtype=torch.float64),
                    self.cholfac_gram_SKhatS,
                    upper=False,
                )
                .squeeze(-1)
                .to(self.train_inputs[0].dtype)
            )

            # (Combined) posterior mean and covariance evaluated at the test point(s)
            mean = self.mean_module(x) + covar_x_train_actions @ compressed_repr_weights
            covar = self.covar_module(x) - operators.RootLinearOperator(
                root=covar_x_train_actions_cholfac_inv
            )

            return MultivariateNormal(mean, covar)


class ComputationAwareELBO(MarginalLogLikelihood):
    """Computation-aware ELBO."""

    def __init__(
        self,
        likelihood: GaussianLikelihood,
        model: ComputationAwareGP,
    ):
        if not isinstance(likelihood, _GaussianLikelihoodBase):
            raise NotImplementedError(
                "Likelihood must be Gaussian for computation-aware inference."
            )
        super().__init__(likelihood, model)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, **kwargs):
        # Initialize some useful variables
        train_inputs = self.model.train_inputs[0]
        train_targets = self.model.train_targets
        num_train_data = len(train_targets)
        prior_evaluated_at_train_inputs = outputs[
            0:num_train_data
        ]  # Training data size might not exactly equal NNZ * PROJ_DIM

        if settings.debug.on():
            # Check whether training objective is evaluated at the training data
            # Note that subsetting is needed here, since a block sparse projection with equal block size
            # necessitates that num_train_data = NNZ * PROJ_DIM
            if (
                not torch.equal(
                    train_inputs, outputs.lazy_covariance_matrix.x1[0:num_train_data]
                )
            ) or (not torch.equal(train_targets, targets[0:num_train_data])):
                raise RuntimeError(
                    "You must evaluate the objective on the training inputs!"
                )

        # Explicitly free up memory from prediction to avoid unnecessary memory overhead
        del self.model.cholfac_gram_SKhatS

        # Lazily evaluate kernel at training inputs as a 4D tensor with shape (PROJ_DIM, PROJ_DIM, NNZ, NNZ)
        K_lazy = self.model.covar_module.forward_keops(
            train_inputs.view(
                self.model.projection_dim,
                self.model.num_non_zero,
                train_inputs.shape[-1],
            ),
            train_inputs.view(
                self.model.projection_dim,
                1,
                self.model.num_non_zero,
                train_inputs.shape[-1],
            ),
        )

        # Compute S'K in block shape (PROJ_DIM, PROJ_DIM, NNZ)
        StK_block_shape = (
            K_lazy
            @ self.model.actions_op.blocks.view(
                self.model.projection_dim, 1, self.model.num_non_zero, 1
            )
        ).squeeze(-1)
        covar_x_batch_X_train_actions = StK_block_shape.view(
            self.model.projection_dim,
            self.model.projection_dim * self.model.num_non_zero,
        ).mT

        # Projected Gramians S'KS and S'(K + noise)S
        gram_SKS = (StK_block_shape * self.model.actions_op.blocks).sum(-1)
        StrS_diag = (self.model.actions_op.blocks**2).sum(
            -1
        )  # NOTE: Assumes orthogonal actions.
        gram_SKhatS = gram_SKS + torch.diag(self.likelihood.noise * StrS_diag)

        # Cholesky factor of Gramian
        cholfac_gram_SKhatS = linop_utils.cholesky.psd_safe_cholesky(
            gram_SKhatS.to(dtype=torch.float64), upper=False
        )

        # Save Cholesky factor for prediction
        self.model.cholfac_gram_SKhatS = cholfac_gram_SKhatS.clone().detach()

        # "Projected" training data (with mean correction)
        actions_targets = self.model.actions_op._matmul(
            torch.atleast_2d(train_targets - prior_evaluated_at_train_inputs.mean).mT
        ).squeeze(-1)

        # Compressed representer weights
        compressed_repr_weights = torch.cholesky_solve(
            actions_targets.unsqueeze(1).to(dtype=torch.float64),
            cholfac_gram_SKhatS,
            upper=False,
        ).squeeze(-1)

        # Expected log-likelihood term
        f_pred_mean_batch = (
            prior_evaluated_at_train_inputs.mean
            + covar_x_batch_X_train_actions
            @ torch.atleast_1d(compressed_repr_weights).to(dtype=targets.dtype)
        )
        sqrt_downdate = torch.linalg.solve_triangular(
            cholfac_gram_SKhatS, covar_x_batch_X_train_actions.mT, upper=False
        )
        trace_downdate = torch.sum(sqrt_downdate**2, dim=-1)
        f_pred_var_batch = torch.sum(
            prior_evaluated_at_train_inputs.variance
        ) - torch.sum(trace_downdate)
        expected_log_likelihood_term = -0.5 * (
            num_train_data * torch.log(self.likelihood.noise)
            + 1
            / self.likelihood.noise
            * (
                torch.linalg.vector_norm(train_targets - f_pred_mean_batch) ** 2
                + f_pred_var_batch
            )
            + num_train_data * torch.log(torch.as_tensor(2 * math.pi))
        ).div(num_train_data)

        # KL divergence to prior
        kl_prior_term = 0.5 * (
            torch.inner(
                compressed_repr_weights,
                (gram_SKS.to(dtype=torch.float64) @ compressed_repr_weights),
            )
            + 2 * torch.sum(torch.log(cholfac_gram_SKhatS.diagonal()))
            - self.model.projection_dim
            * torch.log(self.likelihood.noise).to(dtype=torch.float64)
            - torch.log(StrS_diag.to(dtype=torch.float64).sum())
            - torch.trace(
                torch.cholesky_solve(
                    gram_SKS.to(dtype=torch.float64), cholfac_gram_SKhatS, upper=False
                )
            )
        ).div(num_train_data)

        elbo = torch.squeeze(
            expected_log_likelihood_term - kl_prior_term.to(dtype=targets.dtype)
        )
        return elbo
