"""
LSTM autoencoder health indicator for borehole pump condition monitoring.
Trained on healthy sensor sequences. Reconstruction error = anomaly score.
High reconstruction error → pump degrading → maintenance dispatched proactively.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import json
import argparse

MODEL_DIR = Path("models/saved")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

N_FEATURES = 5  # current, vibration, pressure, flow_rate, water_depth
SEQ_LEN = 48    # 48 × 15min = 12 hours


class Encoder(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, n_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, n_layers, batch_first=True, dropout=0.1)

    def forward(self, x):
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, n_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True, dropout=0.1)
        self.output_layer = nn.Linear(hidden_size, n_features)

    def forward(self, hidden, cell, seq_len):
        input_seq = torch.zeros(hidden.shape[1], seq_len, hidden.shape[2]).to(hidden.device)
        output, _ = self.lstm(input_seq, (hidden, cell))
        return self.output_layer(output)


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int = N_FEATURES, hidden_size: int = 64, n_layers: int = 2):
        super().__init__()
        self.encoder = Encoder(n_features, hidden_size, n_layers)
        self.decoder = Decoder(n_features, hidden_size, n_layers)
        self.seq_len = SEQ_LEN
        self._threshold = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, cell = self.encoder(x)
        return self.decoder(hidden, cell, x.shape[1])

    def compute_anomaly_score(self, x: torch.Tensor) -> float:
        self.eval()
        with torch.no_grad():
            recon = self(x)
            mse = ((x - recon) ** 2).mean().item()
        return float(mse)

    def set_threshold(self, healthy_scores: np.ndarray, percentile: float = 95.0):
        self._threshold = float(np.percentile(healthy_scores, percentile))

    def is_anomalous(self, x: torch.Tensor) -> bool:
        if self._threshold is None:
            raise ValueError("Call set_threshold() first with healthy validation scores.")
        return self.compute_anomaly_score(x) > self._threshold

    @property
    def threshold(self):
        return self._threshold


def generate_synthetic_pump_data(
    n_boreholes: int = 50,
    hours: int = 720,
    failure_rate: float = 0.20,
) -> tuple:
    np.random.seed(42)
    all_healthy, all_degrading = [], []
    failure_labels = []

    for bh_id in range(n_boreholes):
        will_fail = np.random.random() < failure_rate
        time = np.arange(hours)

        current    = 4.5 + 0.3 * np.sin(time * 2 * np.pi / 24) + np.random.normal(0, 0.1, hours)
        vibration  = 0.8 + 0.1 * np.sin(time * 2 * np.pi / 12) + np.random.normal(0, 0.05, hours)
        pressure   = 3.2 + 0.2 * np.sin(time * 2 * np.pi / 24) + np.random.normal(0, 0.08, hours)
        flow_rate  = 12.0 + 1.0 * np.sin(time * 2 * np.pi / 24) + np.random.normal(0, 0.3, hours)
        water_depth = 15.0 + 0.5 * np.sin(time * 2 * np.pi / (24 * 30)) + np.random.normal(0, 0.1, hours)

        if will_fail:
            failure_point = np.random.randint(int(hours * 0.6), int(hours * 0.9))
            degradation = np.linspace(0, 1, hours - failure_point) ** 0.7
            current[failure_point:]   += degradation * np.random.uniform(1.5, 3.0)
            vibration[failure_point:] += degradation * np.random.uniform(0.4, 1.2)
            pressure[failure_point:]  -= degradation * np.random.uniform(0.5, 1.5)
            flow_rate[failure_point:] -= degradation * np.random.uniform(2.0, 5.0)
            failure_labels.append(1)
        else:
            failure_point = None
            failure_labels.append(0)

        sensor_matrix = np.stack([current, vibration, pressure, flow_rate, water_depth], axis=1)
        healthy_end = failure_point - 200 if will_fail and failure_point else hours - 100
        for t in range(0, min(healthy_end, hours - SEQ_LEN), SEQ_LEN):
            all_healthy.append(sensor_matrix[t:t + SEQ_LEN])
        if will_fail and failure_point:
            for t in range(failure_point, min(failure_point + 100, hours - SEQ_LEN), SEQ_LEN):
                all_degrading.append(sensor_matrix[t:t + SEQ_LEN])

    healthy = np.array(all_healthy)
    degrading = np.array(all_degrading) if all_degrading else np.zeros((1, SEQ_LEN, N_FEATURES))
    mean = healthy.mean(axis=(0, 1))
    std = healthy.std(axis=(0, 1)) + 1e-8
    healthy_norm = (healthy - mean) / std
    degrading_norm = (degrading - mean) / std
    return healthy_norm, degrading_norm, mean, std


def train(epochs: int = 30, lr: float = 1e-3) -> dict:
    print("Generating synthetic sensor data...")
    healthy, degrading, mean, std = generate_synthetic_pump_data()
    print(f"  Healthy sequences: {len(healthy)} | Degrading: {len(degrading)}")

    n_val = max(1, int(len(healthy) * 0.15))
    X_train = torch.tensor(healthy[:-n_val], dtype=torch.float32)
    X_val   = torch.tensor(healthy[-n_val:], dtype=torch.float32)
    X_deg   = torch.tensor(degrading, dtype=torch.float32)

    model = LSTMAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    train_loader = DataLoader(TensorDataset(X_train, X_train), batch_size=32, shuffle=True)

    print(f"Training LSTM Autoencoder ({sum(p.numel() for p in model.parameters()):,} params)...")
    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, _ in train_loader:
            optimizer.zero_grad()
            recon = model(x)
            loss = nn.MSELoss()(recon, x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_recon = model(X_val)
            val_loss = nn.MSELoss()(val_recon, X_val).item()
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_DIR / "lstm_autoencoder.pt")

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:03d} | Train: {train_loss/len(train_loader):.5f} | Val: {val_loss:.5f}")

    model.load_state_dict(torch.load(MODEL_DIR / "lstm_autoencoder.pt"))
    with torch.no_grad():
        healthy_scores = [model.compute_anomaly_score(x.unsqueeze(0)) for x in X_val]
    model.set_threshold(np.array(healthy_scores))

    if len(X_deg) > 0:
        deg_scores = [model.compute_anomaly_score(x.unsqueeze(0)) for x in X_deg]
        detection_rate = np.mean(np.array(deg_scores) > model.threshold)
        print(f"  Degrading detection rate: {detection_rate:.1%}")
    else:
        detection_rate = 0.0

    np.save(MODEL_DIR / "scaler_mean.npy", mean)
    np.save(MODEL_DIR / "scaler_std.npy", std)

    results = {
        "best_val_loss": best_val_loss,
        "anomaly_threshold": model.threshold,
        "detection_rate": float(detection_rate),
        "n_features": N_FEATURES,
        "seq_len": SEQ_LEN,
    }
    with open(MODEL_DIR / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Threshold: {model.threshold:.6f} | Model saved to {MODEL_DIR}/")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    if args.train:
        train()
