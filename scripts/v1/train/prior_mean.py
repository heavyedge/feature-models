import argparse
import logging
import pathlib

import model.prior as model_module  # Needs PYTHONPATH=scripts/v0
import pandas as pd
import torch

logging.basicConfig(
    level=getattr(logging, "INFO"),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("Xtrain", type=pathlib.Path, help="Training feature csv file.")
parser.add_argument("ytrain", type=pathlib.Path, help="Training target csv file.")
parser.add_argument("Xval", type=pathlib.Path, help="Validation feature csv file.")
parser.add_argument("yval", type=pathlib.Path, help="Validation target csv file.")
parser.add_argument("--target", type=str, help="Target variable name.")
parser.add_argument("--model", type=str, help="Model name.")
parser.add_argument("--num-epochs", type=int, help="Number of naxunyn epochs.")
parser.add_argument(
    "--learning-rate", type=float, default=0.001, help="Learning rate for optimizer."
)
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output model file.")
parser.add_argument("--device", choices=["cpu", "cuda"], help="Device to train on")
args = parser.parse_args()

if args.device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(args.device)

Xtrain = torch.tensor(pd.read_csv(args.Xtrain).values).float().to(device)
ytrain = torch.tensor(pd.read_csv(args.ytrain)[args.target].values).float().to(device)
Xval = torch.tensor(pd.read_csv(args.Xval).values).float().to(device)
yval = torch.tensor(pd.read_csv(args.yval)[args.target].values).float().to(device)

dim = Xtrain.shape[-1]
batch_shape = Xtrain.shape[:-2]

PriorMean = getattr(model_module, args.model)
model = PriorMean(batch_shape=batch_shape).to(device)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
loss_fn = torch.nn.MSELoss()
for epoch in range(args.num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(Xtrain)
    train_loss = loss_fn(output, ytrain)
    train_loss.backward()
    optimizer.step()

    with torch.no_grad():
        model.eval()
        val_output = model(Xval)
        val_loss = loss_fn(val_output, yval)

    if (epoch + 1) % 100 == 0:
        logger.info(
            f"Epoch [{epoch + 1}/{args.num_epochs}] "
            f"Train Loss: {train_loss.item():.6f}, "
            f"Validation Loss: {val_loss.item():.6f}"
        )


torch.save(
    model.state_dict(),
    args.out,
)
