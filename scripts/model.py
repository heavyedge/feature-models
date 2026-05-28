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

try:
    from .prior import PriorMean_H, Unscaler
except ImportError:
    from prior import PriorMean_H, Unscaler

__all__ = [
    "GPR_H",
    "GPR_b",
    "GPR_phi",
    "DirectLmcMtgpqr_H",
    "CgIndependentMtgpqr_phi",
    "save_model",
    "load_model",
]


# GPR


class _PriorMean_H_GPR(gpytorch.means.Mean):
    """Modified version of model by Schmitt.

    Input X must be [Rgt, Ca, surface_tension, ...].
    """

    def __init__(self, offset=False, batch_shape=torch.Size()):
        super().__init__()
        self.batch_shape = batch_shape
        if offset:
            self.register_parameter(
                "offset",
                torch.nn.Parameter(torch.zeros(batch_shape)),
            )
        else:
            self.register_buffer(
                "offset",
                torch.zeros(batch_shape),
            )

    def forward(self, x):
        Rgt = x[..., 0]
        Ca = x[..., 1]
        cos_theta = x[..., 2]

        a, b, c = 0.22, -0.43, 0.77  # From GPR prior
        lamda = a * Ca**b * cos_theta**c
        E = 2 / (-lamda + torch.sqrt(lamda**2 + (4 / Rgt)))

        model = Rgt / E
        corrected_model = torch.where(model >= 1, model, torch.ones_like(model))
        return corrected_model + self.offset[..., None, None]


class GPR_H(ExactGP):
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

        if X_scale is None:
            X_scale = torch.ones(D)
        if X_min is None:
            X_min = torch.zeros(D)
        unscaler = Unscaler(X_scale=X_scale, X_min=X_min)

        self.mean_module = torch.nn.Sequential(
            unscaler,
            _PriorMean_H_GPR(offset=False, batch_shape=batch_shape),
        )
        self.covar_module = ScaleKernel(
            RBFKernel(ard_num_dims=D, batch_shape=batch_shape),
            batch_shape=batch_shape,
        )

    def forward(self, x):
        mean_x = self.mean_module(x.unsqueeze(-3)).squeeze(-2)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def quantiles(self, x, quantiles):
        """Estimate quantile levels of response variable.

        Parameters
        ----------
        x: torch.Tensor in shape (*B, N, D)
        quantiles: torch.Tensor in shape (Q,)

        Returns
        -------
        quantiles_x: torch.Tensor in shape (*B, N, Q)
        """
        pred = self.likelihood(self(x))
        mean = pred.mean  # (*B, N)
        std = pred.variance.sqrt()  # (*B, N)
        z = torch.distributions.Normal(0, 1).icdf(quantiles)  # (Q,)
        return mean[..., None] + std[..., None] * z  # (*B, N, Q)


class _Kernel_b(gpytorch.kernels.Kernel):
    """Blends two kernels around a changepoint in the first (Rgt) dimension.

    k(x, x') = sigma(x) * sigma(x') * k1(x, x')
    + (1-sigma(x)) * (1-sigma(x')) * k2(x, x')
    where sigma(x) = sigmoid(sharpness * (Rgt(x) - changepoint))
    """

    is_stationary = False

    def __init__(self, kernel1, kernel2, unscaler, **kwargs):
        super().__init__(**kwargs)
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.unscaler = unscaler
        self.register_parameter(
            "raw_changepoint",
            torch.nn.Parameter(torch.tensor(1.2)),
        )

    @property
    def changepoint(self):
        return self.raw_changepoint

    def forward(self, x1, x2, diag=False, **params):
        rgt1 = self.unscaler(x1)[..., 0]  # [..., N]
        rgt2 = self.unscaler(x2)[..., 0]  # [..., M]

        s1 = torch.sigmoid(10 * (rgt1 - self.changepoint))
        s2 = torch.sigmoid(10 * (rgt2 - self.changepoint))

        if diag:
            w1 = s1 * s2  # [..., N]
            w2 = (1 - s1) * (1 - s2)  # [..., N]
            k1 = self.kernel1(x1, x2, diag=True)
            k2 = self.kernel2(x1, x2, diag=True)
            return w1 * k1 + w2 * k2

        w1 = s1.unsqueeze(-1) * s2.unsqueeze(-2)  # [..., N, M]
        w2 = (1 - s1).unsqueeze(-1) * (1 - s2).unsqueeze(-2)  # [..., N, M]

        k1 = self.kernel1(x1, x2).to_dense()
        k2 = self.kernel2(x1, x2).to_dense()

        return w1 * k1 + w2 * k2


class GPR_b(ExactGP):
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

        if X_scale is None:
            X_scale = torch.ones(D)
        if X_min is None:
            X_min = torch.zeros(D)
        unscaler = Unscaler(X_scale=X_scale, X_min=X_min)

        self.mean_module = torch.nn.Sequential(
            unscaler,
            ConstantMean(batch_shape=torch.Size([*batch_shape, 1])),
        )

        lower = torch.tensor([0.01, 0.01, 0.01] + [0 for _ in range(D - 3)])
        upper = torch.tensor([1e4, 1e4, 1e4] + [1e4 for _ in range(D - 3)])
        init_ls = torch.tensor([0.5, 0.5, 0.5] + [0.5 for _ in range(D - 3)])
        kernel1 = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=D, batch_shape=batch_shape),
            batch_shape=batch_shape,
        )
        kernel1.base_kernel.register_constraint(
            "raw_lengthscale", gpytorch.constraints.Interval(lower, upper)
        )
        with torch.no_grad():
            kernel1.base_kernel.lengthscale = init_ls
        kernel2 = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=D, batch_shape=batch_shape),
            batch_shape=batch_shape,
        )
        kernel2.base_kernel.register_constraint(
            "raw_lengthscale", gpytorch.constraints.Interval(lower, upper)
        )
        with torch.no_grad():
            kernel2.base_kernel.lengthscale = init_ls
        self.covar_module = _Kernel_b(kernel1, kernel2, unscaler)

    def forward(self, x):
        mean_x = self.mean_module(x.unsqueeze(-3)).squeeze(-2)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPR_phi(ExactGP):
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

        if X_scale is None:
            X_scale = torch.ones(D)
        if X_min is None:
            X_min = torch.zeros(D)
        unscaler = Unscaler(X_scale=X_scale, X_min=X_min)

        self.mean_module = torch.nn.Sequential(
            unscaler,
            ConstantMean(batch_shape=torch.Size([*batch_shape, 1])),
        )
        self.covar_module = ScaleKernel(
            RBFKernel(ard_num_dims=D, batch_shape=batch_shape),
            batch_shape=batch_shape,
        )
        lower = torch.tensor([1, 1, 1] + [0 for _ in range(D - 3)])
        upper = torch.tensor([1e4 for _ in range(D)])
        init_ls = torch.tensor([2, 2, 2] + [0.5 for _ in range(D - 3)])
        self.covar_module.base_kernel.register_constraint(
            "raw_lengthscale", gpytorch.constraints.Interval(lower, upper)
        )
        with torch.no_grad():
            self.covar_module.base_kernel.lengthscale = init_ls

    def forward(self, x):
        mean_x = self.mean_module(x.unsqueeze(-3)).squeeze(-2)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def quantiles(self, x, quantiles):
        """Estimate quantile levels of response variable.

        Parameters
        ----------
        x: torch.Tensor in shape (*B, N, D)
        quantiles: torch.Tensor in shape (Q,)

        Returns
        -------
        quantiles_x: torch.Tensor in shape (*B, N, Q)
        """
        pred = self.likelihood(self(x))
        mean = pred.mean  # (*B, N)
        std = pred.variance.sqrt()  # (*B, N)
        z = torch.distributions.Normal(0, 1).icdf(quantiles)  # (Q,)
        return mean[..., None] + std[..., None] * z  # (*B, N, Q)


# GPQR


class DirectLmcMtgpqr_H(DirectQuantileGP):
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
            PriorMean_H(
                offset=True, batch_shape=torch.Size([*batch_shape, num_latents])
            ),
        )
        covar = ScaleKernel(
            RBFKernel(ard_num_dims=D, batch_shape=full_batch_shape),
            batch_shape=full_batch_shape,
        )

        super().__init__(variational_strategy, mean, covar, -1)


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
