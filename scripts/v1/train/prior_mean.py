import argparse
import logging
import pathlib

import torch

from models.v1.feature_models import prior as model_module
from scripts.v0.train.save import save_prior_mean
from scripts.v1.train.batch import load_batched_arrays

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("X", type=pathlib.Path, help="Feature csv file.")
parser.add_argument("y", type=pathlib.Path, help="Target csv file.")
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
try:
    X_arr, y_arr = load_batched_arrays(
        args.X, args.y, model_class.output_names, args.index_col, args.batch_col
    )
except ValueError as exc:
    parser.error(str(exc))

X = torch.tensor(X_arr, dtype=torch.float32, device=device)  # (*K, N, D)
y = torch.tensor(y_arr, dtype=torch.float32, device=device)  # (*K, 3, N)
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
