import argparse
import importlib
import pathlib
import sys

MODEL_MODULE_PATH = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(MODEL_MODULE_PATH.parent))

parser = argparse.ArgumentParser(
    description="Predict prior mean of shape features.",
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
    "-o", "--out", type=pathlib.Path, required=True, help="Output csv file."
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
    load_model = load_module.load_PriorMean_H
elif args.target == "phi":
    load_model = load_module.load_PriorMean_phi
model = load_model(device=device)
model.eval()

X = torch.tensor(
    pd.read_csv(args.X, index_col=[0, 1, 2]).values,
    dtype=torch.float32,
    device=device,
)

ret = []
with torch.no_grad():
    for i in range(0, X.shape[0], args.chunk_size):
        X_pred = torch.tensor(
            X[i : i + args.chunk_size], dtype=torch.float32, device=device
        )
        pred_mean = model(X_pred)
        ret.append(pred_mean.cpu().numpy())
ret = np.concatenate(ret, axis=0)
pd.DataFrame({args.target: ret}).to_csv(args.out)
