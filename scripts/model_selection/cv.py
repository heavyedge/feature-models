import numpy as np
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

__all__ = [
    "split_data",
    "split_extrapolate_data",
    "quantiles_cv_gpqr",
    "mean_cv_gpr",
    "quantiles_cv_gpr",
]


def split_data(X, y, n_folds, device, random_state=42):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    x_train_list, y_train_list, x_test_list, y_test_list = [], [], [], []
    x_scales, x_mins = [], []
    for train_idx, test_idx in kf.split(X):
        scaler = MinMaxScaler()

        x_train_list.append(torch.tensor(scaler.fit_transform(X[train_idx])))
        y_train_list.append(torch.tensor(y[train_idx]))
        x_test_list.append(torch.tensor(scaler.transform(X[test_idx])))
        y_test_list.append(torch.tensor(y[test_idx]))

        x_scales.append(torch.tensor(scaler.scale_))
        x_mins.append(torch.tensor(scaler.min_))

    x_train_cv = torch.stack(x_train_list).float().to(device)
    y_train_cv = torch.stack(y_train_list).float().to(device)
    x_test_cv = torch.stack(x_test_list).float().to(device)
    y_test_cv = torch.stack(y_test_list).float().to(device)
    x_scales = torch.stack(x_scales).float().to(device)
    x_mins = torch.stack(x_mins).float().to(device)

    return x_train_cv, y_train_cv, x_test_cv, y_test_cv, x_scales, x_mins


def split_extrapolate_data(X, y, ratio, device):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    center = np.full(X_scaled.shape[1], 0.5)
    distances = np.linalg.norm(X_scaled - center, axis=1)

    threshold = np.quantile(distances, ratio)
    train_idx = np.where(distances <= threshold)[0]
    test_idx = np.where(distances > threshold)[0]

    x_train_cv = torch.tensor(X_scaled[train_idx]).float().unsqueeze(0).to(device)
    y_train_cv = torch.tensor(y[train_idx]).float().unsqueeze(0).to(device)
    x_test_cv = torch.tensor(X_scaled[test_idx]).float().unsqueeze(0).to(device)
    y_test_cv = torch.tensor(y[test_idx]).float().unsqueeze(0).to(device)
    x_scales = torch.tensor(scaler.scale_).float().unsqueeze(0).to(device)
    x_mins = torch.tensor(scaler.min_).float().unsqueeze(0).to(device)

    return x_train_cv, y_train_cv, x_test_cv, y_test_cv, x_scales, x_mins


def quantiles_cv_gpqr(
    x_train,
    y_train,
    x_test,
    y_test,
    quantiles,
    model,
    likelihood,
    n_epochs,
    learning_rate=0.001,
    logger=lambda msg: None,
):
    mll = VariationalELBO(likelihood, model, num_data=y_train.shape[1])
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(likelihood.parameters()),
        lr=learning_rate,
    )

    test_losses = []
    for i in range(n_epochs):
        model.train()
        likelihood.train()
        output = model(x_train)

        train_loss = -mll(output, y_train)
        train_loss.sum().backward()
        optimizer.step()
        optimizer.zero_grad()

        model.eval()
        likelihood.eval()
        with torch.no_grad():
            test_loss = -mll(model(x_test), y_test)
            test_losses.append(test_loss.detach().cpu().numpy())

        logger(
            f"Epoch {i+1}/{n_epochs}, "
            f"Train Loss: {train_loss.mean().item():.4f}, "
            f"Mean test loss: {test_loss.mean().item():.4f}"
        )

    return np.array(test_losses)


def mean_cv_gpr(
    x_train,
    y_train,
    x_test,
    y_test,
    model,
    likelihood,
    n_epochs,
    learning_rate=0.001,
    logger=lambda msg: None,
):
    mll = ExactMarginalLogLikelihood(likelihood, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    test_losses = []
    for i in range(n_epochs):
        model.train()
        likelihood.train()
        output = model(x_train)

        train_loss = -mll(output, y_train)
        train_loss.sum().backward()
        optimizer.step()
        optimizer.zero_grad()

        model.eval()
        likelihood.eval()
        with torch.no_grad():
            test_loss = -mll(model(x_test), y_test)
            test_losses.append(test_loss.detach().cpu().numpy())

        logger(
            f"Epoch {i+1}/{n_epochs}, "
            f"Train Loss: {train_loss.mean().item():.4f}, "
            f"Mean test loss: {test_loss.mean().item():.4f}"
        )

    return np.array(test_losses)


def quantiles_cv_gpr(
    x_train,
    y_train,
    x_test,
    y_test,
    quantiles,
    model,
    likelihood,
    n_epochs,
    learning_rate=0.001,
    logger=lambda msg: None,
):
    mll = ExactMarginalLogLikelihood(likelihood, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    test_losses_per_fold = []
    for i in range(n_epochs):
        model.train()
        likelihood.train()
        output = model(x_train)

        train_loss = -mll(output, y_train)
        train_loss.sum().backward()
        optimizer.step()
        optimizer.zero_grad()

        model.eval()
        likelihood.eval()
        with torch.no_grad():
            output = model.quantiles(x_test, quantiles)  # (K, N, Q)
            epoch_fold_losses = []
            for y_test_fold, output_fold in zip(y_test, output):
                pinball_losses = []
                for j, q in enumerate(quantiles):
                    test_loss = mean_pinball_loss(
                        y_test_fold.cpu().numpy(),
                        output_fold[:, j].cpu().numpy(),
                        alpha=q.item(),
                    )
                    pinball_losses.append(test_loss)
                epoch_fold_losses.append(np.mean(pinball_losses))
            test_losses_per_fold.append(epoch_fold_losses)

        logger(
            f"Epoch {i+1}/{n_epochs}, "
            f"Train Loss: {train_loss.mean().item():.4f}, "
            f"Mean test pinball loss: {np.mean(epoch_fold_losses):.4f}"
        )

    return np.array(test_losses_per_fold)
