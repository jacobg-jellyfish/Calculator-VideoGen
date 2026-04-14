"""Train/test matrices for architecture-level regressor fitting."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class TrainingFold(NamedTuple):
    """Scaled train/test matrices and targets."""

    feat_train_scaled: np.ndarray
    y_train: pd.Series
    feat_test_scaled: np.ndarray
    y_test: pd.Series
    scaler: StandardScaler
