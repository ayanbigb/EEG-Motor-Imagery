import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """A tiny Conv1D model for EEG (channels x time).

    Input: (batch, channels, time)
    Output: logits for 2 classes (left/right)
    """

    def __init__(self, in_channels: int, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)
