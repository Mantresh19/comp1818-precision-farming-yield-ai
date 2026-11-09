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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_features)
        ]
    )

    X_train_proc = preprocessor.fit_transform(X-train)
    X_val_proc = preprocessor.transform(X-val)
    X_test = preprocessor.transform(X_test)

    encoded_cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_features).tolist()
    # we're writing this line to get to know exactly the column names before our python was giving us a list eg [0,1,0,0,0,1
                                                                                                                #1,1,0,2,1,2]
                                                                                                                # but we don't know the column names
    # here cat is one hot encoder what it does is converts words into numbers so computer and our ML model can read it
    all_feature_names = num_features + encoded_cat_names

    return (
        X_train_proc, y_train,
        X_val_proc, y_val,
        X_test_proc, y_test,
        preprocessor,
        all_feature_names
    )

if __name__ == "__main__":
    print("Testing data_loader.py on real FAO dataset...")
    X_tr,y_tr,X_va,y_va,X_te,y_te,prep,name=load_and_preprocess_data()

    print(f"[SUCCESS] Data Processes Cleanly: ")
    print(f" - Training set: {X_tr.shape[0]} samples, {X_tr.shape[1]} features")
    print(f" - Validation set: {X_va.shape[0]} samples")
    print(f" - Test set:       {X_te.shape[0]} samples")
    print(f" - Average Yield:  {y_tr.mean():.2f} tonnes/hectare")