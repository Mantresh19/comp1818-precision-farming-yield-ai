from pathlib import Path
import sys
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
def train_ridge(X_train: np.ndarray, y_train: np.ndarray, alpha: float = 1.0) -> Ridge:
    """Fits an L2-regularized linear regression baseline."""
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train, y_train)
    return model
def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 20
) -> RandomForestRegressor:
    """Fits a non-linear decision tree ensemble benchmark."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model
class CropYieldANN(nn.Module):
    """
    PyTorch Multi-Layer Perceptron for crop yield regression.
    Supports both the vanilla paper replication and the tuned architecture.
    """
    def __init__(self, input_dim: int, enhanced: bool = False):
        super().__init__()
        if not enhanced:
            # Paper Replication (Baseline ANN): 2 hidden layers with standard ReLU
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )
        else:
            # Tuned Extension: BatchNorm + LeakyReLU + Dropout regularization
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.1),
                nn.Linear(64, 32),
                nn.LeakyReLU(0.1),
                nn.Linear(32, 1),
            )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
def train_ann_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    enhanced: bool = False,
    epochs: int = 40,
    batch_size: int = 128,
    random_state: int = 42
) -> Tuple[CropYieldANN, List[float], List[float]]:
    """
    Trains either the Vanilla ANN or Extended ANN and tracks train/validation MSE loss.
    When enhanced=True, applies AdamW, ReduceLROnPlateau, and best-checkpoint restoration.
    """
    torch.manual_seed(random_state)
    model = CropYieldANN(input_dim=X_train.shape[1], enhanced=enhanced)
    criterion = nn.MSELoss()
    if not enhanced:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = None
        epochs = min(epochs, 25)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=4
        )
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        ),
        batch_size=batch_size,
        shuffle=True
    )
    val_x_tensor = torch.tensor(X_val, dtype=torch.float32)
    val_y_tensor = torch.tensor(y_val, dtype=torch.float32)
    train_losses: List[float] = []
    val_losses: List[float] = []
    best_val_loss = float("inf")
    best_weights = None
    for _ in range(epochs):
        model.train()
        running_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(bx)
        train_losses.append(running_loss / len(X_train))
        model.eval()
        with torch.no_grad():
            val_preds = model(val_x_tensor)
            val_loss = criterion(val_preds, val_y_tensor).item()
        val_losses.append(val_loss)
        if scheduler is not None:
            scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
    # Restore best validation checkpoint to prevent overfitting
    if enhanced and best_weights is not None:
        model.load_state_dict(best_weights)
    return model, train_losses, val_losses
def predict_ann(model: CropYieldANN, X: np.ndarray) -> np.ndarray:
    """Runs inference on numpy features using a trained PyTorch ANN."""
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X, dtype=torch.float32)
        return model(x_tensor).numpy()
def train_hybrid_meta_learner(
    rf_val_preds: np.ndarray,
    ann_val_preds: np.ndarray,
    y_val: np.ndarray
) -> Ridge:
    """
    Fits a Ridge stacking meta-learner on validation predictions from Random Forest
    and Extended ANN to combine tree-based splits with smooth neural representations.
    """
    stacked_features = np.column_stack([rf_val_preds, ann_val_preds])
    meta_model = Ridge(alpha=1.0, random_state=42)
    meta_model.fit(stacked_features, y_val)
    return meta_model
if __name__ == "__main__":
    # Self-contained verification check
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.data_loader import load_and_preprocess_data
    from src.evaluate import compute_metrices
    print("Loading preprocessed FAO dataset...")
    X_tr, y_tr, X_va, y_va, X_te, y_te, _, _ = load_and_preprocess_data()
    print("Training models (this takes ~6 seconds)...")
    ridge = train_ridge(X_tr, y_tr)
    rf = train_random_forest(X_tr, y_tr)
    vanilla_ann, _, _ = train_ann_model(X_tr, y_tr, X_va, y_va, enhanced=False)
    ext_ann, _, _ = train_ann_model(X_tr, y_tr, X_va, y_va, enhanced=True)
    # Validation predictions for hybrid stacking
    rf_va = rf.predict(X_va)
    ext_va = predict_ann(ext_ann, X_va)
    meta = train_hybrid_meta_learner(rf_va, ext_va, y_va)
    # Test set predictions
    rf_te = rf.predict(X_te)
    ext_te = predict_ann(ext_ann, X_te)
    hybrid_te = meta.predict(np.column_stack([rf_te, ext_te]))
    print("\n--- MODEL VERIFICATION (TEST SET RMSE / R2) ---")
    print("Ridge:        ", compute_metrices(y_te, ridge.predict(X_te)))
    print("Vanilla ANN:  ", compute_metrices(y_te, predict_ann(vanilla_ann, X_te)))
    print("Extended ANN: ", compute_metrices(y_te, ext_te))
    print("Random Forest:", compute_metrices(y_te, rf_te))
    print("Hybrid Stack: ", compute_metrices(y_te, hybrid_te))
    print(f"Hybrid Blend Weights -> RF: {meta.coef_[0]:.2f}, Extended ANN: {meta.coef_[1]:.2f}")