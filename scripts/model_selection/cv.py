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
    "split_data2",
    "split_extrapolate_data2",
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

    x_train = torch.stack(x_train_list).float().to(device)
    y_train = torch.stack(y_train_list).float().to(device)
    x_test = torch.stack(x_test_list).float().to(device)
    y_test = torch.stack(y_test_list).float().to(device)
    x_scales = torch.stack(x_scales).float().to(device)
    x_mins = torch.stack(x_mins).float().to(device)

    return x_train, y_train, x_test, y_test, x_scales, x_mins


def split_extrapolate_data(X, y, ratio, device):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    center = np.full(X_scaled.shape[1], 0.5)
    distances = np.linalg.norm(X_scaled - center, axis=1)

    threshold = np.quantile(distances, ratio)
    train_idx = np.where(distances <= threshold)[0]
    test_idx = np.where(distances > threshold)[0]

    x_train = torch.tensor(X_scaled[train_idx]).float().unsqueeze(0).to(device)
    y_train = torch.tensor(y[train_idx]).float().unsqueeze(0).to(device)
    x_test = torch.tensor(X_scaled[test_idx]).float().unsqueeze(0).to(device)
    y_test = torch.tensor(y[test_idx]).float().unsqueeze(0).to(device)
    x_scales = torch.tensor(scaler.scale_).float().unsqueeze(0).to(device)
    x_mins = torch.tensor(scaler.min_).float().unsqueeze(0).to(device)

    return x_train, y_train, x_test, y_test, x_scales, x_mins


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
            output = model.mean_quantiles_delta(x_test)  # (K, N, Q)
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


def split_data2(X, y, n_folds, device, random_state=42):
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


def split_extrapolate_data2(X, y, ratio, device):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    center = np.full(X_scaled.shape[1], 0.5)
    distances = np.linalg.norm(X_scaled - center, axis=1)

    threshold = np.quantile(distances, ratio)
    train_idx = np.where(distances <= threshold)[0]
    test_idx = np.where(distances > threshold)[0]

    # Add singleton batch dimension
    x_train = torch.tensor(X_scaled[train_idx]).float().to(device).unsqueeze(0)
    y_train = torch.tensor(y[train_idx]).float().to(device).unsqueeze(0)
    x_test = torch.tensor(X_scaled[test_idx]).float().to(device).unsqueeze(0)
    y_test = torch.tensor(y[test_idx]).float().to(device).unsqueeze(0)

    return x_train, y_train, x_test, y_test


def mean_cv_gpr(
    x_train,  # (*B, N_train, D)
    y_train,  # (*B, N_train, 1)
    x_test,  # (*B, N_test, D)
    y_test,  # (*B, N_test, 1)
    x_scaler,
    y_scaler,
    mean,
    model,
    likelihood,
    n_epochs,
    learning_rate=0.001,
    logger=lambda msg: None,
):
    mll = ExactMarginalLogLikelihood(likelihood, model)
    optimizer = torch.optim.Adam(
        list(x_scaler.parameters())
        + list(y_scaler.parameters())
        + list(mean.parameters())
        + list(model.parameters()),
        lr=learning_rate,
    )

    test_losses = []
    for i in range(n_epochs):
        x_scaler.train()
        y_scaler.train()
        mean.train()
        model.train()
        likelihood.train()
        optimizer.zero_grad()

        train_x_scaled = x_scaler(x_train)
        train_res = y_scaler(y_train - mean(x_train)).squeeze(-1)
        model.set_train_data(
            inputs=train_x_scaled.detach(),
            targets=train_res.detach(),
            strict=False,
        )
        train_output = model(train_x_scaled)
        train_loss = -mll(train_output, train_res)
        train_loss.sum().backward()
        optimizer.step()

        with torch.no_grad():
            x_scaler.train()
            y_scaler.train()
            train_x_scaled = x_scaler(x_train)
            train_res = y_scaler(y_train - mean(x_train)).squeeze(-1)
            model.set_train_data(
                inputs=train_x_scaled,
                targets=train_res,
                strict=False,
            )

        mean.eval()
        x_scaler.eval()
        y_scaler.eval()
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            test_output = model(x_scaler(x_test))
            test_res = y_scaler(y_test - mean(x_test)).squeeze(-1)
            test_loss = -mll(test_output, test_res)
            test_losses.append(test_loss.detach().cpu().numpy())

        logger(
            f"Epoch {i+1}/{n_epochs}, "
            f"Train Loss: {train_loss.mean().item():.4f}, "
            f"Mean test loss: {test_loss.mean().item():.4f}"
        )

    return np.array(test_losses)


def quantiles_cv_gpr(
    x_train,  # (*B, N_train, D)
    y_train,  # (*B, N_train, 1)
    x_test,  # (*B, N_test, D)
    y_test,  # (*B, N_test, 1)
    x_scaler,
    y_scaler,
    mean,
    model,
    likelihood,
    quantiles,
    n_epochs,
    learning_rate=0.001,
    logger=lambda msg: None,
):
    mll = ExactMarginalLogLikelihood(likelihood, model)
    optimizer = torch.optim.Adam(
        list(x_scaler.parameters())
        + list(y_scaler.parameters())
        + list(mean.parameters())
        + list(model.parameters()),
        lr=learning_rate,
    )

    test_losses = []
    for i in range(n_epochs):
        x_scaler.train()
        y_scaler.train()
        mean.train()
        model.train()
        likelihood.train()
        optimizer.zero_grad()

        train_x_scaled = x_scaler(x_train)
        train_res = y_scaler(y_train - mean(x_train)).squeeze(-1)
        model.set_train_data(
            inputs=train_x_scaled.detach(),
            targets=train_res.detach(),
            strict=False,
        )
        train_output = model(train_x_scaled)
        train_loss = -mll(train_output, train_res)
        train_loss.sum().backward()
        optimizer.step()

        with torch.no_grad():
            x_scaler.train()
            y_scaler.train()
            train_x_scaled = x_scaler(x_train)
            train_res = y_scaler(y_train - mean(x_train)).squeeze(-1)
            model.set_train_data(
                inputs=train_x_scaled,
                targets=train_res,
                strict=False,
            )

        mean.eval()
        x_scaler.eval()
        y_scaler.eval()
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            test_output = model.quantiles(x_scaler(x_test), quantiles)
            test_res = y_scaler(y_test - mean(x_test)).squeeze(-1)
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

        logger(
            f"Epoch {i+1}/{n_epochs}, "
            f"Train Loss: {train_loss.mean().item():.4f}, "
            f"Mean test pinball loss: {np.mean(epoch_fold_losses):.4f}"
        )

    return np.array(test_losses)
