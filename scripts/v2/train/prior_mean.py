import argparse
import logging
import pathlib

import torch

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
parser.add_argument(
    "--index-col", type=int, nargs="*", help="Index columns for X and y."
)
parser.add_argument(
    "--batch-col",
    type=int,
    nargs="*",
    help="X CSV column(s) defining batch dimensions.",
)
parser.add_argument("--target", type=str, nargs="+", help="Target variable name.")
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
