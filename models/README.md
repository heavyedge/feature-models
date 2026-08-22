# Models

# v0

- Trained on mean profile shape features.
- X : `Rgt`, `Ca`, `cos(theta)`.
- y : `H`, `phi`.
- y are modeled using single-output models.

# v1

- Trained on all profile shape features.
- X : `Rgt`, `Ca`, `cos(theta)`.
- y : `H`, `phi_1`, `phi_3`.
- y are modeled as three independent batches in one GPR or GPQR artifact.

# v2

- Trained on all profile shape features.
- X : `Rgt`, `Ca`, `cos(theta)`.
- y : `H`, `phi_1`, `phi_3`.
- y are modeled using a multi-output model.
