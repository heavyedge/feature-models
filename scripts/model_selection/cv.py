import numpy as np
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

__all__ = [
    "split_data",
    "split_extrapolate_data",
    "cv_gpr",
    "cv_gpqr",
]


def split_data(X, y, n_folds, device, random_state=42):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    x_train_list, y_train_list, x_test_list, y_test_list = [], [], [], []
    for train_idx, test_idx in kf.split(X):

        x_train_list.append(torch.tensor(X[train_idx]))
        y_train_list.append(torch.tensor(y[train_idx]))
        x_test_list.append(torch.tensor(X[test_idx]))
        y_test_list.append(torch.tensor(y[test_idx]))

    x_train = torch.stack(x_train_list).float().to(device)
    y_train = torch.stack(y_train_list).float().to(device)
    x_test = torch.stack(x_test_list).float().to(device)
    y_test = torch.stack(y_test_list).float().to(device)

    return (x_train, y_train, x_test, y_test)


def split_extrapolate_data(X, y, ratio, device):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    center = np.full(X_scaled.shape[1], 0.5)
    distances = np.linalg.norm(X_scaled - center, axis=1)

    threshold = np.quantile(distances, ratio)
    train_idx = np.where(distances <= threshold)[0]
    test_idx = np.where(distances > threshold)[0]

    # Add singleton batch dimension
    x_train = torch.tensor(X[train_idx]).float().to(device).unsqueeze(0)
    y_train = torch.tensor(y[train_idx]).float().to(device).unsqueeze(0)
    x_test = torch.tensor(X[test_idx]).float().to(device).unsqueeze(0)
    y_test = torch.tensor(y[test_idx]).float().to(device).unsqueeze(0)

    return x_train, y_train, x_test, y_test


def cv_gpr(
    x_train,  # (*B, N_train, D)
    y_train,  # (*B, N_train)
    x_test,  # (*B, N_test, D)
    y_test,  # (*B, N_test)
    X_scaler,
    y_scaler,
    mean,
    model,
    likelihood,
    quantiles,
    n_epochs,
    learning_rate=0.001,
    logger=lambda msg: None,
):
    mean.eval()
    mll = ExactMarginalLogLikelihood(likelihood, model)
    optimizer = torch.optim.Adam(
        list(X_scaler.parameters())
        + list(y_scaler.parameters())
        + list(model.parameters()),
        lr=learning_rate,
    )

    test_losses = []
    for i in range(n_epochs):
        X_scaler.train()
        y_scaler.train()
        likelihood.train()
        model.train()
        optimizer.zero_grad()

        train_x_scaled = X_scaler(x_train)
        with torch.no_grad():
            train_mean = mean(x_train)
        train_res = y_scaler((y_train - train_mean).unsqueeze(-1)).squeeze(-1)
        model.set_train_data(
            inputs=train_x_scaled.detach(),
            targets=train_res.detach(),
            strict=False,
        )
        train_output = model(train_x_scaled)
        train_loss = -mll(train_output, train_res)
        train_loss.mean().backward()
        optimizer.step()

        X_scaler.eval()
        y_scaler.eval()
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            test_output = model.quantiles(X_scaler(x_test), quantiles)
            test_res = y_scaler((y_test - mean(x_test)).unsqueeze(-1)).squeeze(-1)
            epoch_fold_losses = []
            for test_res_fold, output_fold in zip(test_res, test_output):
                pinball_losses = []
                for j, q in enumerate(quantiles):
                    test_loss = mean_pinball_loss(
                        test_res_fold.cpu().numpy(),
                        output_fold[:, j].cpu().numpy(),
                        alpha=q.item(),
                    )
                    pinball_losses.append(test_loss)
                epoch_fold_losses.append(np.mean(pinball_losses))
            test_losses.append(epoch_fold_losses)

        if (i + 1) % 100 == 0:
            logger(
                f"Epoch {i+1}/{n_epochs}, "
                f"Train Loss: {train_loss.mean().item():.4f}, "
                f"Mean test pinball loss: {np.mean(epoch_fold_losses):.4f}"
            )

    return np.array(test_losses)  # (N_EPOCHS, N_FOLDS)


def cv_gpqr(
    x_train,  # (*B, N_train, D)
    y_train,  # (*B, N_train)
    x_test,  # (*B, N_test, D)
    y_test,  # (*B, N_test)
    X_scaler,
    y_scaler,
    mean,
    model,
    likelihood,
    quantiles,
    n_epochs,
    learning_rate=0.001,
    logger=lambda msg: None,
):
    mean.eval()
    mll = VariationalELBO(likelihood, model, num_data=y_train.shape[-1])
    optimizer = torch.optim.Adam(
        list(X_scaler.parameters())
        + list(y_scaler.parameters())
        + list(model.parameters())
        + list(likelihood.parameters()),
        lr=learning_rate,
    )

    test_losses = []
    for i in range(n_epochs):
        X_scaler.train()
        y_scaler.train()
        likelihood.train()
        model.train()
        optimizer.zero_grad()

        train_x_scaled = X_scaler(x_train)
        with torch.no_grad():
            train_mean = mean(x_train)
        train_res = y_scaler((y_train - train_mean).unsqueeze(-1)).squeeze(-1)
        train_output = model(train_x_scaled)
        train_loss = -mll(train_output, train_res)
        train_loss.mean().backward()
        optimizer.step()

        X_scaler.eval()
        y_scaler.eval()
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            test_output = model.quantiles(X_scaler(x_test))
            test_res = y_scaler((y_test - mean(x_test)).unsqueeze(-1)).squeeze(-1)
            epoch_fold_losses = []
            for test_res_fold, output_fold in zip(test_res, test_output):
                pinball_losses = []
                for j, q in enumerate(quantiles):
                    test_loss = mean_pinball_loss(
                        test_res_fold.cpu().numpy(),
                        output_fold[:, j].cpu().numpy(),
                        alpha=q.item(),
                    )
                    pinball_losses.append(test_loss)
                epoch_fold_losses.append(np.mean(pinball_losses))
            test_losses.append(epoch_fold_losses)

        if (i + 1) % 100 == 0:
            logger(
                f"Epoch {i+1}/{n_epochs}, "
                f"Train Loss: {train_loss.mean().item():.4f}, "
                f"Mean test pinball loss: {np.mean(epoch_fold_losses):.4f}"
            )

    return np.array(test_losses) # (N_EPOCHS, N_FOLDS)
