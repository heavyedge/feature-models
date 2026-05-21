import numpy as np

__all__ = [
    "AnisotropicRBF",
    "AnisotropicMatern52",
    "AnisotropicMatern32",
    "AnisotropicMatern12",
]


class AnisotropicRBF:
    """RBF kernel with per-feature bandwidth:
    K(x,x') = exp(-sum_j gamma_j*(x_j-x'_j)^2).
    """

    def __init__(self, gammas):
        self.gammas = np.asarray(gammas)

    def __call__(self, X, Y):
        diff = X[:, None, :] - Y[None, :, :]  # (n, m, d)
        return np.exp(-np.einsum("nmd,d->nm", diff**2, self.gammas))


class AnisotropicMatern52:
    """Matern 5/2 kernel with per-feature bandwidth:
    K(x,x') = (1 + sqrt(5)*r + 5/3*r^2) * exp(-sqrt(5)*r),
    where r = sqrt(sum_j gamma_j*(x_j-x'_j)^2).
    """

    def __init__(self, gammas):
        self.gammas = np.asarray(gammas)

    def __call__(self, X, Y):
        diff = X[:, None, :] - Y[None, :, :]  # (n, m, d)
        r2 = np.einsum("nmd,d->nm", diff**2, self.gammas)
        r = np.sqrt(np.maximum(r2, 0.0))
        sqrt5_r = np.sqrt(5.0) * r
        return (1.0 + sqrt5_r + 5.0 / 3.0 * r2) * np.exp(-sqrt5_r)


class AnisotropicMatern32:
    """Matern 3/2 kernel with per-feature bandwidth:
    K(x,x') = (1 + sqrt(3)*r) * exp(-sqrt(3)*r),
    where r = sqrt(sum_j gamma_j*(x_j-x'_j)^2).
    """

    def __init__(self, gammas):
        self.gammas = np.asarray(gammas)

    def __call__(self, X, Y):
        diff = X[:, None, :] - Y[None, :, :]  # (n, m, d)
        r2 = np.einsum("nmd,d->nm", diff**2, self.gammas)
        r = np.sqrt(np.maximum(r2, 0.0))
        sqrt3_r = np.sqrt(3.0) * r
        return (1.0 + sqrt3_r) * np.exp(-sqrt3_r)


class AnisotropicMatern12:
    """Matern 1/2 (Ornstein-Uhlenbeck) kernel with per-feature bandwidth:
    K(x,x') = exp(-r),
    where r = sqrt(sum_j gamma_j*(x_j-x'_j)^2).
    """

    def __init__(self, gammas):
        self.gammas = np.asarray(gammas)

    def __call__(self, X, Y):
        diff = X[:, None, :] - Y[None, :, :]  # (n, m, d)
        r2 = np.einsum("nmd,d->nm", diff**2, self.gammas)
        r = np.sqrt(np.maximum(r2, 0.0))
        return np.exp(-r)
