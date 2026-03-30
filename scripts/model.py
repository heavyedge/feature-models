import gpytorch
import torch
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    UnwhitenedVariationalStrategy,
)
from gpytorch_qr import ALD, CenterGapLmcVariationalStrategy, centergap_to_quantiles
from torch.nn import Module

__all__ = [
    "PriorMean_H",
    "MedianGapGP",
    "MedianGapLikelihood",
    "MTGPQR",
    "train_mtgpqr",
    "save_mtgpqr",
    "load_mtgpqr",
]


class PriorMean_H(gpytorch.means.Mean):
    """Modified version of model by Schmitt.

    Input X must be [Rgt, Ca, surface_tension, ...].
    """

    def __init__(self, scaler):
        super().__init__()
        self.register_buffer("X_scale", torch.tensor(scaler.scale_).float())
        self.register_buffer("X_min", torch.tensor(scaler.min_).float())
        self.register_parameter(
            "mean_params",
            torch.nn.Parameter(torch.tensor([0.25, -0.27, -1.0, -1.0])),
        )

    def forward(self, x):
        x_unscaled = (x - self.X_min) / self.X_scale
        Rgt = x_unscaled[..., 0]
        Ca = x_unscaled[..., 1]
        st = x_unscaled[..., 2]

        a, b, c, d = self.mean_params
        lamda = (a * Rgt + b) * Ca**c * st**d
        E = 2 / (-lamda + torch.sqrt(lamda**2 + (4 / Rgt)))

        model = Rgt / E
        corrected_model = torch.where(model >= 1, model, torch.ones_like(model))
        return corrected_model


class MedianGapGP(gpytorch.models.ApproximateGP):
    """
    tasks[0] : median
    tasks[1:1+lower_count] : lower gaps
    tasks[1+lower_count:] : upper gaps
    """

    def __init__(self, train_x, median_mean_module, taus, num_half_lmc_latents):
        self.lower_taus = taus[taus < 0.5]
        self.upper_taus = taus[taus > 0.5]
        assert len(self.lower_taus) == len(self.upper_taus)
        self.taus = torch.cat(
            [
                self.lower_taus,
                torch.tensor([0.5], device=taus.device),
                self.upper_taus,
            ]
        )
        N = train_x.size(0)
        D = train_x.size(1)
        T = 1 + len(self.lower_taus) + len(self.upper_taus)
        Q = 1 + 2 * num_half_lmc_latents

        variational_distribution = CholeskyVariationalDistribution(
            N,
            batch_shape=torch.Size([Q]),
        )
        variational_strategy = CenterGapLmcVariationalStrategy(
            UnwhitenedVariationalStrategy(
                self,
                train_x,
                variational_distribution,
                learn_inducing_locations=False,
            ),
            num_tasks=T,
            num_latents=Q,
            latent_dim=-1,
            num_lower_quantiles=len(self.lower_taus),
            num_lower_latents=num_half_lmc_latents,
        )
        super().__init__(variational_strategy)

        # First task (g1) determines the median, and remaining task determines gap.
        # g1 has median mean prior, while g2, ..., gQ has constant mean prior.
        self.median_mean_module = median_mean_module
        self.gap_mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([Q - 1])
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=D,
                batch_shape=torch.Size([Q]),
            ),
            batch_shape=torch.Size([Q]),
        )

    def forward(self, x):
        median_mean = self.median_mean_module(x)  # (N,)
        gap_mean = self.gap_mean_module(x)  # (Q-1, N)
        mean = torch.concatenate([median_mean.unsqueeze(0), gap_mean], dim=0)  # (Q, N)
        covar = self.covar_module(x)  # (Q, N, N)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class MedianGapLikelihood(gpytorch.likelihoods.Likelihood):
    def __init__(self, taus):
        super().__init__()
        self.register_buffer("taus", taus)
        self.register_parameter(
            name="raw_scales",
            parameter=torch.nn.Parameter(torch.zeros(len(taus))),
        )
        self.register_constraint(
            "raw_scales",
            gpytorch.constraints.Positive(),
        )
        self.lower_count = (taus < 0.5).count_nonzero()

    @property
    def scales(self):
        # (T,)
        return self.raw_scales_constraint.transform(self.raw_scales)

    def forward(self, function_samples):
        # function_samples: (S, N, T) <- from MedianGapGP
        median = function_samples[:, :, :1]
        lower_gaps = function_samples[:, :, 1 : 1 + self.lower_count]
        upper_gaps = function_samples[:, :, 1 + self.lower_count :]
        quantiles = centergap_to_quantiles(median, lower_gaps, upper_gaps)
        return ALD(
            locs=quantiles,  # (S, N, T)
            scales=self.scales,  # (T,)
            taus=self.taus,  # (T,)
        )

    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        lp = super().expected_log_prob(
            observations, function_dist, *args, **kwargs
        )  # (N, T)
        return lp.sum(dim=1)  # (N,)

    def log_marginal(self, observations, function_dist, *args, **kwargs):
        lp = super().log_marginal(
            observations, function_dist, *args, **kwargs
        )  # (N, T)
        return lp.sum(dim=1)  # (N,)


class MTGPQR(Module):
    """Multi-Task Gaussian Process Quantile Regression."""

    def __init__(self, train_x, median_mean_module, taus, num_half_lmc_latents):
        super().__init__()
        self.gp = MedianGapGP(train_x, median_mean_module, taus, num_half_lmc_latents)
        self.likelihood = MedianGapLikelihood(self.gp.taus)

        self.train_x = train_x
        self.taus = taus
        self.num_half_lmc_latents = num_half_lmc_latents

    def forward(self, x):
        function_means = self.gp(x).mean
        median = function_means[..., :1]
        lower_gaps = function_means[..., 1 : 1 + len(self.gp.lower_taus)]
        upper_gaps = function_means[..., 1 + len(self.gp.lower_taus) :]
        return centergap_to_quantiles(median, lower_gaps, upper_gaps)


def train_mtgpqr(
    train_x,
    train_y,
    median_mean_module,
    taus,
    num_half_lmc_latents,
    num_epochs,
    learning_rate=0.001,
    lmc_l2_weight=0.0,
    device=None,
    logger=None,
):
    # train_x: (N, D)
    # train_y: (N,)
    model = MTGPQR(train_x, median_mean_module, taus, num_half_lmc_latents).to(device)
    model.train()

    # Setup optimizer
    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(
        parameters,
        lr=learning_rate,
    )

    mll = gpytorch.mlls.VariationalELBO(
        model.likelihood, model.gp, num_data=len(train_y)
    )

    # Training loop
    for i in range(num_epochs):
        optimizer.zero_grad()
        output = model.gp(train_x)
        loss = -mll(output, train_y)
        if lmc_l2_weight > 0.0:
            lmc_coeff_1 = model.gp.variational_strategy.partial_lmc_coefficients_1
            lmc_coeff_2 = model.gp.variational_strategy.partial_lmc_coefficients_2
            loss = loss + lmc_l2_weight * (
                lmc_coeff_1.pow(2).sum() + lmc_coeff_2.pow(2).sum()
            )
        loss.backward()
        optimizer.step()

        if logger is not None:
            logger(i, num_epochs, loss.item())

    return model


def save_mtgpqr(model, path):
    torch.save(
        {
            "train_x": model.train_x.detach().clone().cpu(),
            "taus": model.taus.detach().clone().cpu(),
            "num_half_lmc_latents": model.num_half_lmc_latents,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_mtgpqr(path, median_mean_module, device=None):
    checkpoint = torch.load(path, map_location=device)
    train_x = checkpoint["train_x"].to(device)
    taus = checkpoint["taus"].to(device)
    num_half_lmc_latents = checkpoint["num_half_lmc_latents"]

    model = MTGPQR(train_x, median_mean_module, taus, num_half_lmc_latents).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model
