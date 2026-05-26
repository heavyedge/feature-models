import gpytorch
import torch
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    IndependentMultitaskVariationalStrategy,
    LMCVariationalStrategy,
    UnwhitenedVariationalStrategy,
)
from gpytorch_qr.likelihoods import (
    MultitaskCenterGapQuantileGPLikelihood,
    MultitaskQuantileGPLikelihood,
)
from gpytorch_qr.means import CenterGapMean
from gpytorch_qr.models import CenterGapQuantileGP, DirectQuantileGP, QuantileGP
from gpytorch_qr.variational import CGBlkdiagLmcVariationalStrategy

try:
    from .prior import PriorMean_H, Unscaler
except ImportError:
    from prior import PriorMean_H, Unscaler

__all__ = [
    "GPR_H_ConstantMean",
    "CgLmcMtgpqr_H",
    "CgLmcMtgpqr_H_ConstantMean",
    "CgLmcMtgpqr_phi",
    "CgIndependentMtgpqr_H",
    "CgIndependentMtgpqr_H_ConstantMean",
    "CgIndependentMtgpqr_phi",
    "DirectLmcMtgpqr_H_ConstantMean",
    "DirectIndependentMtgpqr_H",
    "DirectIndependentMtgpqr_H_ConstantMean",
    "DirectIndependentMtgpqr_phi",
]


class GPR_H_ConstantMean(ExactGP):
    def __init__(
        self,
        train_x,
        train_y,
        likelihood,
        X_scale=None,
        X_min=None,
        batch_shape=torch.Size(),
    ):
        D = train_x.shape[-1]
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = ConstantMean(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(
            RBFKernel(ard_num_dims=D, batch_shape=batch_shape),
            batch_shape=batch_shape,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def quantiles(self, x, quantiles):
        pred = self.likelihood(self(x))
        mean = pred.mean  # (*B, N)
        std = pred.variance.sqrt()  # (*B, N)
        z = torch.distributions.Normal(0, 1).icdf(quantiles)  # (Q,)
        return mean[..., None] + std[..., None] * z  # (*B, N, Q)


class CgLmcMtgpqr_H(CenterGapQuantileGP):
    def __init__(
        self,
        inducing_points,
        num_quantiles,
        num_lower_quantiles,
        num_latents,
        num_lower_latents,
        X_scale=None,
        X_min=None,
        batch_shape=torch.Size(),
    ):
        N, D = inducing_points.shape[-2:]
        full_batch_shape = torch.Size([*batch_shape, num_latents])
        variational_distribution = CholeskyVariationalDistribution(
            N,
            batch_shape=full_batch_shape,
        )
        variational_strategy = CGBlkdiagLmcVariationalStrategy(
            UnwhitenedVariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=False,
            ),
            num_quantiles=num_quantiles,
            num_latents=num_latents,
            num_lower_quantiles=num_lower_quantiles,
            num_lower_latents=num_lower_latents,
        )

        if X_scale is None:
            X_scale = torch.ones(D)
        if X_min is None:
            X_min = torch.zeros(D)
        unscaler = Unscaler(X_scale=X_scale, X_min=X_min)

        mean = CenterGapMean(
            torch.nn.Sequential(
                unscaler,
                PriorMean_H(batch_shape=torch.Size([*batch_shape, 1])),
            ),
            ConstantMean(batch_shape=torch.Size([*batch_shape, num_latents - 1])),
            latent_dim=-1,
        )
        covar = ScaleKernel(
            RBFKernel(ard_num_dims=D, batch_shape=full_batch_shape),
            batch_shape=full_batch_shape,
        )

        super().__init__(variational_strategy, mean, covar, -1, num_lower_quantiles)


class CgLmcMtgpqr_H_ConstantMean(CenterGapQuantileGP):
    def __init__(
        self,
        inducing_points,
        num_quantiles,
        num_lower_quantiles,
        num_latents,
        num_lower_latents,
        X_scale=None,
        X_min=None,
        batch_shape=torch.Size(),
    ):
        N, D = inducing_points.shape[-2:]
        full_batch_shape = torch.Size([*batch_shape, num_latents])
        variational_distribution = CholeskyVariationalDistribution(
            N,
            batch_shape=full_batch_shape,
        )
        variational_strategy = CGBlkdiagLmcVariationalStrategy(
            UnwhitenedVariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=False,
            ),
            num_quantiles=num_quantiles,
            num_latents=num_latents,
            num_lower_quantiles=num_lower_quantiles,
            num_lower_latents=num_lower_latents,
        )

        if X_scale is None:
            X_scale = torch.ones(D)
        if X_min is None:
            X_min = torch.zeros(D)
        unscaler = Unscaler(X_scale=X_scale, X_min=X_min)

        mean = CenterGapMean(
            torch.nn.Sequential(
                unscaler,
                ConstantMean(batch_shape=torch.Size([*batch_shape, 1])),
            ),
            ConstantMean(batch_shape=torch.Size([*batch_shape, num_latents - 1])),
            latent_dim=-1,
        )
        covar = ScaleKernel(
            RBFKernel(ard_num_dims=D, batch_shape=full_batch_shape),
            batch_shape=full_batch_shape,
        )

        super().__init__(variational_strategy, mean, covar, -1, num_lower_quantiles)


class CgLmcMtgpqr_phi(CenterGapQuantileGP):
    def __init__(
        self,
        inducing_points,
        num_quantiles,
        num_lower_quantiles,
        num_latents,
        num_lower_latents,
        X_scale=None,
        X_min=None,
        batch_shape=torch.Size(),
    ):
        N, D = inducing_points.shape[-2:]
        full_batch_shape = torch.Size([*batch_shape, num_latents])
        variational_distribution = CholeskyVariationalDistribution(
            N,
            batch_shape=full_batch_shape,
        )
        variational_strategy = CGBlkdiagLmcVariationalStrategy(
            UnwhitenedVariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=False,
            ),
            num_quantiles=num_quantiles,
            num_latents=num_latents,
            num_lower_quantiles=num_lower_quantiles,
            num_lower_latents=num_lower_latents,
        )

        if X_scale is None:
            X_scale = torch.ones(D)
        if X_min is None:
            X_min = torch.zeros(D)
        unscaler = Unscaler(X_scale=X_scale, X_min=X_min)

        mean = CenterGapMean(
            torch.nn.Sequential(
                unscaler,
                ConstantMean(
                    batch_shape=torch.Size([*batch_shape, 1]),
                ),
            ),
            ConstantMean(batch_shape=torch.Size([*batch_shape, num_latents - 1])),
            latent_dim=-1,
        )
        covar = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=D, batch_shape=full_batch_shape),
            batch_shape=full_batch_shape,
        )

        lower = torch.tensor([1, 1, 1] + [0 for _ in range(D - 3)])
        upper = torch.tensor([1e4 for _ in range(D)])
        init_ls = torch.tensor([2, 2, 2] + [0.5 for _ in range(D - 3)])
        covar.base_kernel.register_constraint(
            "raw_lengthscale", gpytorch.constraints.Interval(lower, upper)
        )
        with torch.no_grad():
            covar.base_kernel.lengthscale = init_ls

        super().__init__(variational_strategy, mean, covar, -1, num_lower_quantiles)


# Center-gap Independent MTGPQR


class CgIndependentMtgpqr_H(CenterGapQuantileGP):
    def __init__(
        self,
        inducing_points,
        num_quantiles,
        num_lower_quantiles,
        num_latents,
        num_lower_latents,
        X_scale=None,
        X_min=None,
        batch_shape=torch.Size(),
    ):
        num_latents = num_quantiles
        N, D = inducing_points.shape[-2:]
        full_batch_shape = torch.Size([*batch_shape, num_latents])
        variational_distribution = CholeskyVariationalDistribution(
            N,
            batch_shape=full_batch_shape,
        )
        variational_strategy = IndependentMultitaskVariationalStrategy(
            UnwhitenedVariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=False,
            ),
            num_tasks=num_quantiles,
        )

        if X_scale is None:
            X_scale = torch.ones(D)
        if X_min is None:
            X_min = torch.zeros(D)
        unscaler = Unscaler(X_scale=X_scale, X_min=X_min)

        mean = CenterGapMean(
            torch.nn.Sequential(
                unscaler,
                PriorMean_H(batch_shape=torch.Size([*batch_shape, 1])),
            ),
            ConstantMean(batch_shape=torch.Size([*batch_shape, num_latents - 1])),
            latent_dim=-1,
        )
        covar = ScaleKernel(
            RBFKernel(ard_num_dims=D, batch_shape=full_batch_shape),
            batch_shape=full_batch_shape,
        )

        super().__init__(variational_strategy, mean, covar, -1, num_lower_quantiles)


class CgIndependentMtgpqr_H_ConstantMean(CenterGapQuantileGP):
    def __init__(
        self,
        inducing_points,
        num_quantiles,
        num_lower_quantiles,
        num_latents,
        num_lower_latents,
        X_scale=None,
        X_min=None,
        batch_shape=torch.Size(),
    ):
        num_latents = num_quantiles
        N, D = inducing_points.shape[-2:]
        full_batch_shape = torch.Size([*batch_shape, num_latents])
        variational_distribution = CholeskyVariationalDistribution(
            N,
            batch_shape=full_batch_shape,
        )
        variational_strategy = IndependentMultitaskVariationalStrategy(
            UnwhitenedVariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=False,
            ),
            num_tasks=num_quantiles,
        )

        if X_scale is None:
            X_scale = torch.ones(D)
        if X_min is None:
            X_min = torch.zeros(D)
        unscaler = Unscaler(X_scale=X_scale, X_min=X_min)

        mean = CenterGapMean(
            torch.nn.Sequential(
                unscaler,
                ConstantMean(batch_shape=torch.Size([*batch_shape, 1])),
            ),
            ConstantMean(batch_shape=torch.Size([*batch_shape, num_latents - 1])),
            latent_dim=-1,
        )
        covar = ScaleKernel(
            RBFKernel(ard_num_dims=D, batch_shape=full_batch_shape),
            batch_shape=full_batch_shape,
        )

        super().__init__(variational_strategy, mean, covar, -1, num_lower_quantiles)


class CgIndependentMtgpqr_phi(CenterGapQuantileGP):
    def __init__(
        self,
        inducing_points,
        num_quantiles,
        num_lower_quantiles,
        num_latents,
        num_lower_latents,
        X_scale=None,
        X_min=None,
        batch_shape=torch.Size(),
    ):
        num_latents = num_quantiles
        N, D = inducing_points.shape[-2:]
        full_batch_shape = torch.Size([*batch_shape, num_latents])
        variational_distribution = CholeskyVariationalDistribution(
            N,
            batch_shape=full_batch_shape,
        )
        variational_strategy = IndependentMultitaskVariationalStrategy(
            UnwhitenedVariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=False,
            ),
            num_tasks=num_quantiles,
        )

        if X_scale is None:
            X_scale = torch.ones(D)
        if X_min is None:
            X_min = torch.zeros(D)
        unscaler = Unscaler(X_scale=X_scale, X_min=X_min)

        mean = CenterGapMean(
            torch.nn.Sequential(
                unscaler,
                ConstantMean(
                    batch_shape=torch.Size([*batch_shape, 1]),
                ),
            ),
            ConstantMean(batch_shape=torch.Size([*batch_shape, num_latents - 1])),
            latent_dim=-1,
        )
        covar = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=D, batch_shape=full_batch_shape),
            batch_shape=full_batch_shape,
        )

        lower = torch.tensor([1, 1, 1] + [0 for _ in range(D - 3)])
        upper = torch.tensor([1e4 for _ in range(D)])
        init_ls = torch.tensor([2, 2, 2] + [0.5 for _ in range(D - 3)])
        covar.base_kernel.register_constraint(
            "raw_lengthscale", gpytorch.constraints.Interval(lower, upper)
        )
        with torch.no_grad():
            covar.base_kernel.lengthscale = init_ls

        super().__init__(variational_strategy, mean, covar, -1, num_lower_quantiles)


# Direct LMC MTGPQR


class DirectLmcMtgpqr_H_ConstantMean(DirectQuantileGP):
    def __init__(
        self,
        inducing_points,
        num_quantiles,
        num_lower_quantiles,
        num_latents,
        num_lower_latents,
        X_scale=None,
        X_min=None,
        batch_shape=torch.Size(),
    ):
        N, D = inducing_points.shape[-2:]
        full_batch_shape = torch.Size([*batch_shape, num_latents])
        variational_distribution = CholeskyVariationalDistribution(
            N,
            batch_shape=full_batch_shape,
        )
        variational_strategy = LMCVariationalStrategy(
            UnwhitenedVariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=False,
            ),
            num_tasks=num_quantiles,
            num_latents=num_latents,
        )

        if X_scale is None:
            X_scale = torch.ones(D)
        if X_min is None:
            X_min = torch.zeros(D)
        unscaler = Unscaler(X_scale=X_scale, X_min=X_min)

        mean = torch.nn.Sequential(
            unscaler,
            ConstantMean(batch_shape=torch.Size([*batch_shape, 1])),
        )
        covar = ScaleKernel(
            RBFKernel(ard_num_dims=D, batch_shape=full_batch_shape),
            batch_shape=full_batch_shape,
        )

        super().__init__(variational_strategy, mean, covar, -1)


# Direct Independent MTGPQR


class DirectIndependentMtgpqr_H(DirectQuantileGP):
    def __init__(
        self,
        inducing_points,
        num_quantiles,
        num_lower_quantiles,
        num_latents,
        num_lower_latents,
        X_scale=None,
        X_min=None,
        batch_shape=torch.Size(),
    ):
        num_latents = num_quantiles
        N, D = inducing_points.shape[-2:]
        full_batch_shape = torch.Size([*batch_shape, num_latents])
        variational_distribution = CholeskyVariationalDistribution(
            N,
            batch_shape=full_batch_shape,
        )
        variational_strategy = IndependentMultitaskVariationalStrategy(
            UnwhitenedVariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=False,
            ),
            num_tasks=num_quantiles,
        )

        if X_scale is None:
            X_scale = torch.ones(D)
        if X_min is None:
            X_min = torch.zeros(D)
        unscaler = Unscaler(X_scale=X_scale, X_min=X_min)

        mean = torch.nn.Sequential(
            unscaler,
            PriorMean_H(offset=True, batch_shape=torch.Size([*batch_shape, 1])),
        )
        covar = ScaleKernel(
            RBFKernel(ard_num_dims=D, batch_shape=full_batch_shape),
            batch_shape=full_batch_shape,
        )

        super().__init__(variational_strategy, mean, covar, -1)


class DirectIndependentMtgpqr_H_ConstantMean(DirectQuantileGP):
    def __init__(
        self,
        inducing_points,
        num_quantiles,
        num_lower_quantiles,
        num_latents,
        num_lower_latents,
        X_scale=None,
        X_min=None,
        batch_shape=torch.Size(),
    ):
        num_latents = num_quantiles
        N, D = inducing_points.shape[-2:]
        full_batch_shape = torch.Size([*batch_shape, num_latents])
        variational_distribution = CholeskyVariationalDistribution(
            N,
            batch_shape=full_batch_shape,
        )
        variational_strategy = IndependentMultitaskVariationalStrategy(
            UnwhitenedVariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=False,
            ),
            num_tasks=num_quantiles,
        )

        if X_scale is None:
            X_scale = torch.ones(D)
        if X_min is None:
            X_min = torch.zeros(D)
        unscaler = Unscaler(X_scale=X_scale, X_min=X_min)

        mean = torch.nn.Sequential(
            unscaler,
            ConstantMean(batch_shape=torch.Size([*batch_shape, 1])),
        )
        covar = ScaleKernel(
            RBFKernel(ard_num_dims=D, batch_shape=full_batch_shape),
            batch_shape=full_batch_shape,
        )

        super().__init__(variational_strategy, mean, covar, -1)


class DirectIndependentMtgpqr_phi(DirectQuantileGP):
    def __init__(
        self,
        inducing_points,
        num_quantiles,
        num_lower_quantiles,
        num_latents,
        num_lower_latents,
        X_scale=None,
        X_min=None,
        batch_shape=torch.Size(),
    ):
        num_latents = num_quantiles
        N, D = inducing_points.shape[-2:]
        full_batch_shape = torch.Size([*batch_shape, num_latents])
        variational_distribution = CholeskyVariationalDistribution(
            N,
            batch_shape=full_batch_shape,
        )
        variational_strategy = IndependentMultitaskVariationalStrategy(
            UnwhitenedVariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=False,
            ),
            num_tasks=num_quantiles,
        )

        if X_scale is None:
            X_scale = torch.ones(D)
        if X_min is None:
            X_min = torch.zeros(D)
        unscaler = Unscaler(X_scale=X_scale, X_min=X_min)

        mean = torch.nn.Sequential(
            unscaler,
            ConstantMean(batch_shape=torch.Size([*batch_shape, 1])),
        )
        covar = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=D, batch_shape=full_batch_shape),
            batch_shape=full_batch_shape,
        )

        lower = torch.tensor([1, 1, 1] + [0 for _ in range(D - 3)])
        upper = torch.tensor([1e4 for _ in range(D)])
        init_ls = torch.tensor([2, 2, 2] + [0.5 for _ in range(D - 3)])
        covar.base_kernel.register_constraint(
            "raw_lengthscale", gpytorch.constraints.Interval(lower, upper)
        )
        with torch.no_grad():
            covar.base_kernel.lengthscale = init_ls

        super().__init__(variational_strategy, mean, covar, -1)


def save_model(
    train_x,
    train_y,
    model,
    likelihood,
    scaler,
    inducing_points,
    quantiles,
    num_lower_quantiles,
    num_latents,
    num_lower_latents,
    path,
):
    torch.save(
        {
            "train_x": train_x,
            "train_y": train_y,
            "model_state_dict": model.state_dict(),
            "likelihood_state_dict": likelihood.state_dict(),
            "scaler": scaler,
            "inducing_points": inducing_points,
            "quantiles": quantiles,
            "num_lower_quantiles": num_lower_quantiles,
            "num_latents": num_latents,
            "num_lower_latents": num_lower_latents,
        },
        path,
    )


def load_model(model_class, path, device=None):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if issubclass(model_class, ExactGP):
        likelihood = GaussianLikelihood()
    elif issubclass(model_class, CenterGapQuantileGP):
        likelihood = MultitaskCenterGapQuantileGPLikelihood(
            checkpoint["quantiles"],
            checkpoint["num_lower_quantiles"],
        )
    elif issubclass(model_class, DirectQuantileGP):
        likelihood = MultitaskQuantileGPLikelihood(
            checkpoint["quantiles"],
        )
    else:
        raise ValueError("Unsupported model class.")
    if issubclass(model_class, ExactGP):
        model = model_class(
            train_x=checkpoint["train_x"],
            train_y=checkpoint["train_y"],
            likelihood=likelihood,
        )
    if issubclass(model_class, QuantileGP):
        model = model_class(
            inducing_points=checkpoint["inducing_points"],
            num_quantiles=len(checkpoint["quantiles"]),
            num_lower_quantiles=checkpoint["num_lower_quantiles"],
            num_latents=checkpoint["num_latents"],
            num_lower_latents=checkpoint["num_lower_latents"],
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    likelihood.load_state_dict(checkpoint["likelihood_state_dict"])
    if device is not None:
        model.to(device)
        likelihood.to(device)
    scaler = checkpoint["scaler"]
    return model, likelihood, scaler
