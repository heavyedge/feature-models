import argparse
import copy
import logging
import pathlib

import pandas as pd
import torch
import v0.model.prior as model_module  # Needs PYTHONPATH=scripts

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
parser.add_argument("--num-epochs", type=int, help="Number of maximum epochs.")
parser.add_argument(
    "--learning-rate",
    type=float,
    default=0.001,
    help="Initial learning rate for optimizer.",
)
parser.add_argument(
    "--early-stopping-patience",
    type=int,
    default=20,
    help="Stop after this many epochs without validation-loss improvement.",
)
parser.add_argument(
    "--early-stopping-min-delta",
    type=float,
    default=0.0,
    help="Minimum validation-loss decrease required to reset early stopping.",
)
parser.add_argument(
    "--lr-scheduler-patience",
    type=int,
    default=10,
    help="Epochs without validation-loss improvement before reducing learning rate.",
)
parser.add_argument(
    "--lr-scheduler-factor",
    type=float,
    default=0.5,
    help="Factor by which to reduce the learning rate.",
)
parser.add_argument(
    "--min-learning-rate",
    type=float,
    default=1e-6,
    help="Minimum learning rate for the scheduler.",
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
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=args.lr_scheduler_factor,
    patience=args.lr_scheduler_patience,
    min_lr=args.min_learning_rate,
)
loss_fn = torch.nn.MSELoss()
best_val_loss = float("inf")
best_model_state = None
epochs_without_improvement = 0

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

    current_val_loss = val_loss.item()
    lr_scheduler.step(current_val_loss)

    if current_val_loss < best_val_loss - args.early_stopping_min_delta:
        best_val_loss = current_val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if (epoch + 1) % 100 == 0:
        logger.info(
            f"Epoch [{epoch + 1}/{args.num_epochs}] "
            f"Train Loss: {train_loss.item():.6f}, "
            f"Validation Loss: {current_val_loss:.6f}, "
            f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}"
        )

    if epochs_without_improvement >= args.early_stopping_patience:
        logger.info(
            "Early stopping at epoch %d; best validation loss: %.6f",
            epoch + 1,
            best_val_loss,
        )
        break

if best_model_state is not None:
    model.load_state_dict(best_model_state)

torch.save(
    model.state_dict(),
    args.out,
)
