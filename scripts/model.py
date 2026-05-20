import gpytorch
import torch
from gpqr import (
    CenterGapGP,
    CenterGapLikelihood,
    CenterGapLmcVariationalStrategy,
    centergap_to_quantiles,
)
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
)

__all__ = [
    "Unscaler",
    "PriorMean_H",
    "MTGP",
    "MTGPQR_H",
    "MTGPQR_phi",
    "train_model",
    "save_model",
    "load_model",
]


class Unscaler(torch.nn.Module):
    """Un-min-max-scaling."""

    def __init__(self, X_scale, X_min):
        super().__init__()
        self.register_buffer("X_scale", X_scale)
        self.register_buffer("X_min", X_min)

    def forward(self, x):
        return (x - self.X_min) / self.X_scale


class PriorMean_H(gpytorch.means.Mean):
    """Modified version of model by Schmitt.

    Input X must be [Rgt, Ca, surface_tension, ...].
    """

    def forward(self, x):
        Rgt = x[..., 0]
        Ca = x[..., 1]
        cos_theta = x[..., 2]

        a, b, c = 0.22, -0.43, 0.77  # From GPR prior
        lamda = a * Ca**b * cos_theta**c
        E = 2 / (-lamda + torch.sqrt(lamda**2 + (4 / Rgt)))

        model = Rgt / E
        corrected_model = torch.where(model >= 1, model, torch.ones_like(model))
        return corrected_model


class MTGP(CenterGapGP):
    def __init__(
        self,
        inducing_points,
        num_quantiles,
        num_lower_quantiles,
        num_latents,
        num_lower_latents,
        center_mean,
        gap_mean,
        covar_module,
    ):
        N, _ = inducing_points.size()
        variational_strategy = CenterGapLmcVariationalStrategy(
            VariationalStrategy(
                self,
                inducing_points,
                CholeskyVariationalDistribution(
                    N,
                    batch_shape=torch.Size([num_latents]),
                ),
                learn_inducing_locations=False,
            ),
            num_tasks=num_quantiles,
            num_latents=num_latents,
            latent_dim=-1,
            num_lower_quantiles=num_lower_quantiles,
            num_lower_latents=num_lower_latents,
        )
        super().__init__(variational_strategy, center_mean, gap_mean, covar_module)
        self.num_lower_quantiles = num_lower_quantiles

    def mean_quantiles(self, x):
        """Compute quantile functions."""
        function_means = self(x).mean
        median = function_means[..., :1]
        lower_gaps = function_means[..., 1 : 1 + self.num_lower_quantiles]
        upper_gaps = function_means[..., 1 + self.num_lower_quantiles :]
        return centergap_to_quantiles(median, lower_gaps, upper_gaps)


class MTGPQR_H(MTGP):
    def __init__(
        self,
        inducing_points,
        num_quantiles,
        num_lower_quantiles,
        num_latents,
        num_lower_latents,
        X_scale=None,
        X_min=None,
    ):
        _, D = inducing_points.size()
        if X_scale is None:
            X_scale = torch.ones(D)
        if X_min is None:
            X_min = torch.zeros(D)

        unscaler = Unscaler(X_scale, X_min)
        center_mean = torch.nn.Sequential(unscaler, PriorMean_H())
        gap_mean = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([num_latents - 1])
        )
        covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=D,
                batch_shape=torch.Size([num_latents]),
            ),
            batch_shape=torch.Size([num_latents]),
        )

        super().__init__(
            inducing_points,
            num_quantiles,
            num_lower_quantiles,
            num_latents,
            num_lower_latents,
            center_mean,
            gap_mean,
            covar_module,
        )


class MTGPQR_phi(MTGP):
    def __init__(
        self,
        inducing_points,
        num_quantiles,
        num_lower_quantiles,
        num_latents,
        num_lower_latents,
        X_scale=None,
        X_min=None,
    ):
        _, D = inducing_points.size()
        if X_scale is None:
            X_scale = torch.ones(D)
        if X_min is None:
            X_min = torch.zeros(D)

        unscaler = Unscaler(X_scale, X_min)
        center_mean = torch.nn.Sequential(unscaler, gpytorch.means.ConstantMean())
        gap_mean = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([num_latents - 1])
        )
        covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=D,
                batch_shape=torch.Size([num_latents]),
            ),
            batch_shape=torch.Size([num_latents]),
        )
        lower = torch.tensor([1, 1, 1] + [0 for _ in range(D - 3)])
        upper = torch.tensor([1e4 for _ in range(D)])
        init_ls = torch.tensor([2, 2, 2] + [0.5 for _ in range(D - 3)])
        covar_module.base_kernel.register_constraint(
            "raw_lengthscale", gpytorch.constraints.Interval(lower, upper)
        )
        with torch.no_grad():
            covar_module.base_kernel.lengthscale = init_ls

        super().__init__(
            inducing_points,
            num_quantiles,
            num_lower_quantiles,
            num_latents,
            num_lower_latents,
            center_mean,
            gap_mean,
            covar_module,
        )


def train_model(
    train_x,
    train_y,
    model,
    likelihood,
    num_epochs,
    learning_rate=0.001,
    logger=None,
):
    # train_x: (N, D)
    # train_y: (N,)
    model.train()
    likelihood.train()

    # Setup optimizer
    parameters = list(model.parameters()) + list(likelihood.parameters())
    optimizer = torch.optim.Adam(
        parameters,
        lr=learning_rate,
    )

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(train_y))

    # Training loop
    for i in range(num_epochs):
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if logger is not None:
            logger(i, num_epochs, loss.item())

    return model


def save_model(
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
    model = model_class(
        inducing_points=checkpoint["inducing_points"],
        num_quantiles=len(checkpoint["quantiles"]),
        num_lower_quantiles=checkpoint["num_lower_quantiles"],
        num_latents=checkpoint["num_latents"],
        num_lower_latents=checkpoint["num_lower_latents"],
    )
    likelihood = CenterGapLikelihood(taus=checkpoint["quantiles"])
    model.load_state_dict(checkpoint["model_state_dict"])
    likelihood.load_state_dict(checkpoint["likelihood_state_dict"])
    if device is not None:
        model.to(device)
        likelihood.to(device)
    scaler = checkpoint["scaler"]
    return model, likelihood, scaler
