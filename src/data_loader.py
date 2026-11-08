from typing import Tuple, List
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def load_and_preprocess_data(
    csv_path: str = "data/crop_yield.csv",
    test_size: float = 0.20,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer, List[str]]:
    df = pd.read_csv(csv_path)