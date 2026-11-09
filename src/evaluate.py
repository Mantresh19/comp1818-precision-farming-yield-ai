from pathlib import Path
import sys
from typing import Dict
import numpy as np 
import pandas as pd
from sklearn.metrics import mean_squarede_error, mean_absolute_error, r2_score

def compute_metrices(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100.0)