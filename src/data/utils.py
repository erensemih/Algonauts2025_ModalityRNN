import numpy as np


def pad_or_crop(feat: np.ndarray, T: int) -> np.ndarray:
    """
    If visual_feat has fewer than T timesteps, pad by repeating the last row;
    if it has more, crop to the first T; otherwise return as‐is.
    """
    n, dim = feat.shape
    if n < T:
        # repeat the last row to reach length T
        last_row = feat[-1:]
        padding = np.repeat(last_row, T - n, axis=0)
        return np.vstack([feat, padding])
    elif n > T:
        # crop down to the first T rows
        return feat[:T]
    else:
        # already exactly T
        return feat.copy()
