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

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df["yield_tonnes_ha"] = df["hg/ha_yield"] / 10000.0

    num_features = ["Year", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]
    cat_features = ["Area", "Item"]

    x = df[num_features + cat_features]
    y = df["yield_tonnes_ha"].values

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state
    )

