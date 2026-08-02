import argparse
import importlib
import pathlib
import sys

MODEL_MODULE_PATH = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(MODEL_MODULE_PATH.parent))

parser = argparse.ArgumentParser(
    description="Predict posterior distribution of mean from GPR.",
)
parser.add_argument(
    "X",
    type=pathlib.Path,
    help=(
        "Input csv file, shape: (N, D). "
        "The first three dimensions must be "
        "the Gap-to-thickness ratio, "
        "the Capillary number, and "
        "the cosine of the contact angle of the fluid on the substrate."
    ),
)
parser.add_argument("--target", required=True, choices=["H", "phi"])
parser.add_argument(
    "--chunk-size",
    type=int,
    default=4096,
    help="Number of samples to process at once.",
)
parser.add_argument(
    "-o",
    "--out",
    type=pathlib.Path,
    required=True,
    help=(
        "Output csv file. "
        "The output shape is (N, 2), where the first dimension corresponds to "
        "the mean and standard deviation of the posterior distribution."
    ),
)
args = parser.parse_args()

try:
    import numpy as np
    import pandas as pd
    import torch
except ImportError:
    setup_module = importlib.import_module(f"{MODEL_MODULE_PATH.name}.setup")
    setup_module.setup(MODEL_MODULE_PATH)

    import numpy as np
    import pandas as pd
    import torch

load_module = importlib.import_module(f"{MODEL_MODULE_PATH.name}.load")

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.target == "H":
    load_models = load_module.load_GPR_H
elif args.target == "phi":
    load_models = load_module.load_GPR_phi
models = load_models(device=device)
for module in models[1:]:
    module.eval()
X_scaler, y_scaler, mean, likelihood, model = models

X = torch.tensor(
    pd.read_csv(args.X, index_col=[0, 1, 2]).values,
    dtype=torch.float32,
    device=device,
)

ret = []
with torch.no_grad():
    for i in range(0, X.shape[0], args.chunk_size):
        X_pred = X[i : i + args.chunk_size]
        X_scaled = X_scaler(X_pred)
        scaled_res_posterior = model(X_scaled)

        scaled_res_mean = scaled_res_posterior.mean.unsqueeze(-1)
        residual_mean = y_scaler.inverse_transform(scaled_res_mean).squeeze(-1)
        residual_std = (
            scaled_res_posterior.variance.sqrt().unsqueeze(-1)
            * y_scaler.X_scale.abs().unsqueeze(-2)
        ).squeeze(-1)

        posterior_mean = mean(X_pred) + residual_mean
        ret.append(torch.stack((posterior_mean, residual_std)).cpu().numpy())
ret = np.concatenate(ret, axis=1)
df = pd.DataFrame(ret.T, columns=[f"{args.target}_mean", f"{args.target}_std"])
df.to_csv(args.out, index=False)
