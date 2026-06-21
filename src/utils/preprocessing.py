from typing import Optional
import numpy as np


def normalize_epoch(epoch: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Z-score normalize a single epoch across time for each channel."""
    mean = epoch.mean(axis=1, keepdims=True)
    std = epoch.std(axis=1, keepdims=True)
    return (epoch - mean) / (std + eps)


def epoch_signal(raw: np.ndarray, events: np.ndarray, tmin: int, tmax: int) -> np.ndarray:
    """Basic epoching: raw shape (channels, time), events list of sample indices.
    Returns array (n_events, channels, epoch_length).
    """
    epochs = []
    length = tmax - tmin
    for ev in events:
        start = ev + tmin
        stop = start + length
        epochs.append(raw[:, start:stop])
    return np.stack(epochs, axis=0)
