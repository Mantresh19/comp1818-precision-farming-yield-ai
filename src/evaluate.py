from pathlib import Path
import sys
from typing import Dict
import numpy as np 
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compute_metrices(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100.0)

    return {
        "RMSE (t/ha)": round(rmse, 4),
        "MAE (t/ha)": round(mae, 4),
        "R2 Score": round(r2, 4),
        "MAPE (%)": round(mape, 2)
    }

def evaluate_trivial_baselines(
    y_train: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:

    results: Dict[str, Dict[str, float]] = {}

    mean_pred = np.full(
        shape=len(y_test),
        fill_value=np.mean(y_train)
        )
    
    results["Trivial: Global Mean"] = compute_metrices(y_test, mean_pred)
    
    median_pred = np.full(
        shape=len(y_test),
        fill_value=np.median(y_train)
    )
    results["Trivial: Global Median"] = compute_metrices(y_test, median_pred)

    rng = np.random.default_rng(random_state)

    random_pred = rng.normal(
        loc=np.mean(y_train),
        scale=np.std(y_train),
        size=len(y_test)
    )

    results["Trivial: Random Guessing"] = compute_metrices(y_test, random_pred)

    return results

def format_results_table(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.name = "Model"
    return df

if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.data_loader import load_and_preprocess_data

    print("Loading data and evaluating trivinal baselines")
    _,y_tr,_,_,_,y_te,_,_ = load_and_preprocess_data()

    baseline_results = evaluate_trivial_baselines(y_tr, y_te)
    table = format_results_table(baseline_results)

    print("\n Trival Baseline Benchmarks (Test Set) ")
    print(table.to_string())