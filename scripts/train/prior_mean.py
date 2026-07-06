import argparse
import importlib
import logging
import pathlib
import sys

import pandas as pd
import torch

MODEL_MODULE_PATH = pathlib.Path(__file__).resolve().parent.parent / "model"
sys.path.insert(0, str(MODEL_MODULE_PATH.parent))
model_module = importlib.import_module(MODEL_MODULE_PATH.name)

logging.basicConfig(
    level=getattr(logging, "INFO"),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("X", type=pathlib.Path, help="Feature csv file.")
parser.add_argument("y", type=pathlib.Path, help="Target csv file.")
parser.add_argument("--target", type=str, help="Target variable name.")
parser.add_argument("--model", type=str, help="Model name.")
parser.add_argument("--num-epochs", type=int, help="Number of training epochs.")
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

X = torch.tensor(pd.read_csv(args.X).drop(columns="Slurry").values).float().to(device)
y = torch.tensor(pd.read_csv(args.y)[args.target].values).float().to(device)

dim = X.shape[-1]
batch_shape = X.shape[:-2]

model_class = getattr(model_module, args.model)
model = model_class(batch_shape=batch_shape).to(device)

# Train
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
loss_fn = torch.nn.MSELoss()
for epoch in range(args.num_epochs):
    optimizer.zero_grad()
    output = model(X)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % max(1, args.num_epochs // 10) == 0:
        logger.info(f"Epoch [{epoch + 1}/{args.num_epochs}] Loss: {loss.item():.6f}")


torch.save(
    model.state_dict(),
    args.out,
)
