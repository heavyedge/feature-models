import gpytorch
import torch

__all__ = [
    "Unscaler",
    "PriorMean_H",
    "Scaler",
    "PriorMean_H_2",
]


class Unscaler(torch.nn.Module):
    """Un-min-max-scaling.

    Parameters
    ----------
    X_scale, X_min: torch.Tensor in shape (*B, D)
    """

    def __init__(self, X_scale, X_min):
        super().__init__()
        self.register_buffer("X_scale", X_scale)
        self.register_buffer("X_min", X_min)

    def forward(self, x):
        # x: (*B, 1, N, D)
        # BELOW IS CORRECT FOR MinMaxScaler (different from StandardScaler)
        X_min = self.X_min[..., None, None, :]
        X_scale = self.X_scale[..., None, None, :]
        x_unscaled = (x - X_min) / X_scale
        return x_unscaled.view_as(x)


class PriorMean_H(gpytorch.means.Mean):
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
        return corrected_model + self.offset[..., None]  # (*B, N)


class Scaler(torch.nn.Module):
    """Min-max-scaling.

    Parameters
    ----------
    X_scale, X_min: torch.Tensor in shape (*B, D)
        Values from sklearn.preprocessing.MinMaxScaler.
    """

    def __init__(self, X_scale, X_min):
        super().__init__()
        self.register_buffer("X_scale", X_scale)
        self.register_buffer("X_min", X_min)

    def forward(self, x):
        # x: (*B, 1, N, D)
        # BELOW IS CORRECT FOR MinMaxScaler (different from StandardScaler)
        X_min = self.X_min[..., None, None, :]
        X_scale = self.X_scale[..., None, None, :]
        x_scaled = x * X_scale + X_min
        return x_scaled.view_as(x)


class PriorMean_H_2(torch.nn.Module):
    """Modified version of model by Schmitt.

    Input X must be [Rgt, Ca, cos_theta, ...].
    """

    def __init__(self, batch_shape=torch.Size()):
        super().__init__()
        self.batch_shape = batch_shape
        self.params = torch.nn.ParameterDict(
            {
                "a": torch.nn.Parameter(torch.tensor(1.0).expand(batch_shape)),
                "b": torch.nn.Parameter(torch.tensor(0.0).expand(batch_shape)),
                "c": torch.nn.Parameter(torch.tensor(0.0).expand(batch_shape)),
            }
        )

    def forward(self, x):
        Rgt = x[..., 0]
        Ca = x[..., 1]
        cos_theta = x[..., 2]

        a = self.params["a"]
        b = self.params["b"]
        c = self.params["c"]

        lamda = a * Ca**b * cos_theta**c
        E = 2 / (-lamda + torch.sqrt(lamda**2 + (4 / Rgt)))

        model = Rgt / E
        corrected_model = torch.where(model >= 1, model, torch.ones_like(model))
        return corrected_model[...]  # (*B, N)
