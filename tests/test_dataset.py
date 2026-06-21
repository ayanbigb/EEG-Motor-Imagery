import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.data.dataset import EEGDataset


def test_dataset_basic():
    X = np.random.randn(8, 22, 500).astype(np.float32)
    y = np.random.randint(0, 2, size=(8,))
    ds = EEGDataset(X, y)
    assert len(ds) == 8
    x0, y0 = ds[0]
    assert x0.shape == (22, 500)
    assert isinstance(y0, int)


if __name__ == "__main__":
    test_dataset_basic()
    print("dataset test passed")
