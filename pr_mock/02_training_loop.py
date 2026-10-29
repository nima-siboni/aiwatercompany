"""
PR #157: Training loop for the property-prediction MLP with early stopping

Trains a small MLP on the featurised MOF data (see data_prep.py) and keeps
the checkpoint with the best validation loss.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=128, n_layers=3, dropout=0.1):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers += [
                nn.Linear(in_dim if i == 0 else hidden, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def evaluate(model, loader, device):
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        total += nn.functional.mse_loss(pred, yb)
    return total / len(loader)


def train(X_train, y_train, X_val, y_val, epochs=200, lr=1e-3, batch_size=64, patience=10):
    device = "cuda"
    model = MLP(X_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=batch_size,
        shuffle=True,
    )

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    history = []

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        val_loss = evaluate(model, val_loader, device)
        history.append(val_loss)
        print(f"epoch {epoch} train {loss:.4f} val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs > patience:
                break

    model.load_state_dict(best_state)
    return model, history
