import argparse
import logging
import pathlib

import torch

from models.v1.feature_models import prior as model_module
from scripts.v1.train.batch import load_batched_arrays
from scripts.v1.train.save import save_prior_mean

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("Xtrain", type=pathlib.Path, help="Training feature csv file.")
parser.add_argument("ytrain", type=pathlib.Path, help="Training target csv file.")
parser.add_argument(
    "Xval", type=pathlib.Path, nargs="?", help="Validation feature csv file."
)
parser.add_argument(
    "yval", type=pathlib.Path, nargs="?", help="Validation target csv file."
)
parser.add_argument("--index-col", type=int, nargs="*", help="Index columns.")
parser.add_argument(
    "--batch-col", type=int, nargs="*", help="Columns defining CV batch dimensions."
)
parser.add_argument("--model", type=str, help="Model name.")
parser.add_argument("--num-epochs", type=int, help="Number of training epochs.")
parser.add_argument("--learning-rate", type=float, default=0.001)
parser.add_argument("-o", "--out", type=pathlib.Path, required=True)
parser.add_argument("--device", choices=["cpu", "cuda"])
args = parser.parse_args()

device = torch.device(
    args.device
    if args.device is not None
    else "cuda" if torch.cuda.is_available() else "cpu"
)

model_class = getattr(model_module, args.model)


def load_data(X_path, y_path):
    try:
        X_arr, y_arr = load_batched_arrays(
            X_path, y_path, model_class.output_names, args.index_col, args.batch_col
        )
    except ValueError as exc:
        parser.error(str(exc))
    return (
        torch.tensor(X_arr, dtype=torch.float32, device=device),
        torch.tensor(y_arr, dtype=torch.float32, device=device),
    )


if (args.Xval is None) != (args.yval is None):
    parser.error("Xval and yval must be provided together")

X, y = load_data(args.Xtrain, args.ytrain)
if args.Xval is not None:
    Xval, yval = load_data(args.Xval, args.yval)
    if Xval.shape[:-2] != X.shape[:-2]:
        parser.error("training and validation data must have the same batch shape")

    # The K-fold train-validation unions are identical. Prior mean has no HPO,
    # so train one final unbatched model on the first union immediately.
    X = torch.cat((X[0], Xval[0]), dim=-2)
    y = torch.cat((y[0], yval[0]), dim=-1)

model = model_class(batch_shape=X.shape[:-2]).to(device)

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
loss_fn = torch.nn.MSELoss()
for epoch in range(args.num_epochs):
    optimizer.zero_grad()
    output = model(X)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        logger.info("Epoch [%d/%d] Loss: %.6f", epoch + 1, args.num_epochs, loss.item())

save_prior_mean(model, args.out)
