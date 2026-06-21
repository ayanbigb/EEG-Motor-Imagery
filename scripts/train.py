"""Minimal training script skeleton.

Usage:
    python scripts/train.py --config configs/config.yaml
"""
import argparse
import yaml
import os
import random
import numpy as np

import torch
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.models.classifier import SimpleCNN


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    data = np.load(path)
    X = data["X"]
    y = data["y"] if "y" in data.files else None
    return EEGDataset(X, y)


def train(cfg):
    ds = load_data(cfg["dataset"]["path"])
    batch_size = cfg["training"].get("batch_size", 32)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(in_channels=cfg["model"].get("channels", 22), n_classes=2).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg["training"].get("lr", 1e-3))
    loss_fn = torch.nn.CrossEntropyLoss()

    epochs = cfg["training"].get("epochs", 10)
    for ep in range(epochs):
        model.train()
        running = 0.0
        for batch in loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            if x.dim() == 3:  # (batch, channels, time)
                pass
            else:
                x = x.unsqueeze(1)
            logits = model(x)
            loss = loss_fn(logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += loss.item()
        print(f"Epoch {ep+1}/{epochs} loss={running/len(loader):.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    with open(args.config, "r") as fh:
        cfg = yaml.safe_load(fh)
    set_seed(cfg["training"].get("seed", 42))
    train(cfg)


if __name__ == "__main__":
    main()
