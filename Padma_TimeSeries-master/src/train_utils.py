from copy import deepcopy

import numpy as np
import torch

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    mape = (
        mean_absolute_percentage_error(
            y_true,
            y_pred
        ) * 100
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "R2": float(r2)
    }


class EarlyStopping:
    def __init__(
        self,
        patience=15,
        min_delta=0.0
    ):
        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = np.inf
        self.counter = 0
        self.best_state = None


    def step(
        self,
        validation_loss,
        model
    ):
        improved = (
            validation_loss
            <
            self.best_loss - self.min_delta
        )

        if improved:
            self.best_loss = validation_loss
            self.counter = 0

            self.best_state = deepcopy(
                model.state_dict()
            )

            return False

        self.counter += 1

        if self.counter >= self.patience:
            return True

        return False


def evaluate_loss(
    model,
    loader,
    criterion,
    device
):
    model.eval()

    losses = []

    with torch.no_grad():
        for seq, future, target in loader:

            seq = seq.to(device)
            future = future.to(device)
            target = target.to(device)

            prediction = model(
                seq,
                future
            )

            loss = criterion(
                prediction,
                target
            )

            losses.append(
                loss.item()
            )

    return float(
        np.mean(losses)
    )


def predict(
    model,
    loader,
    device
):
    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():
        for seq, future, target in loader:

            seq = seq.to(device)
            future = future.to(device)

            prediction = model(
                seq,
                future
            )

            predictions.append(
                prediction.cpu().numpy()
            )

            targets.append(
                target.numpy()
            )

    return (
        np.vstack(targets),
        np.vstack(predictions)
    )