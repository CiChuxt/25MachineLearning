"""Prepare the reproducible 200 x 784 experiment matrix used by the report."""
from pathlib import Path
import numpy as np
from scipy.ndimage import zoom
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'processed'
OUT.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
SAMPLES_PER_CLASS = 20
N = 10 * SAMPLES_PER_CLASS

digits = load_digits()
images = digits.images.astype(np.float32) / 16.0
labels = digits.target.astype(int)
sss = StratifiedShuffleSplit(n_splits=1, train_size=N, random_state=RANDOM_STATE)
indices, _ = next(sss.split(images.reshape(len(images), -1), labels))
selected = images[indices]
# Derived high-dimensional representation: 8x8 -> 28x28 -> 784 dimensions
images_28 = np.stack([zoom(im, (3.5, 3.5), order=3) for im in selected])
images_28 = np.clip(images_28, 0, 1).astype(np.float32)
np.savez_compressed(
    OUT / 'digits_28x28_stratified_200.npz',
    X=images_28.reshape(N, -1),
    y=labels[indices],
    original_indices=indices,
)
print(f'Saved: {OUT / "digits_28x28_stratified_200.npz"}')
