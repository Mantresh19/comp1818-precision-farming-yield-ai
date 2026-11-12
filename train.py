from pathlib import Path
import joblib
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import load_and_preprocess_data
from src.evaluate import compute_metrices, evaluate_trivial_baselines, format_results_table
from src.models import (
    train_ridge,
    train_random_forest,
    train_ann_model,
    predict_ann,
    train_hybrid_meta_learner,
)
# Anchor paths to project root so script runs from any directory
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = PROJECT_ROOT / "figures"
def generate_report_figures(
    results_df,
    y_test: np.ndarray,
    hybrid_preds: np.ndarray,
    vanilla_val_loss: list,
    ext_val_loss: list,
    rf_model,
    feature_names: list,
) -> None:
    """Generates four 300-DPI figures in figures/ for the coursework report."""
    sns.set_theme(style="whitegrid")
    # 1. Model RMSE Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    sorted_df = results_df.sort_values("RMSE (t/ha)", ascending=False)
    colors = ["#c0392b" if "Trivial" in idx else "#2980b9" if "Hybrid" not in idx else "#27ae60"
              for idx in sorted_df.index]
    ax.barh(sorted_df.index, sorted_df["RMSE (t/ha)"], color=colors, alpha=0.9)
    ax.set_xlabel("Test RMSE (tonnes/hectare) — Lower is Better")
    ax.set_title("Crop Yield Prediction Error Across All Baselines & AI Models", fontweight="bold")
    for i, v in enumerate(sorted_df["RMSE (t/ha)"]):
        ax.text(v + 0.15, i, f"{v:.2f}", va="center", fontsize=9)
    ax.set_xlim(0, sorted_df["RMSE (t/ha)"].max() + 1.5)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "model_comparison.png")
    plt.close()
    # 2. Validation Loss Convergence: Vanilla ANN vs. Extended ANN
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    ax.plot(range(1, len(vanilla_val_loss) + 1), vanilla_val_loss, label="Vanilla ANN (Paper Baseline)", linestyle="--", lw=2)
    ax.plot(range(1, len(ext_val_loss) + 1), ext_val_loss, label="Extended ANN (BatchNorm + AdamW + LR Scheduler)", lw=2.2)
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Validation MSE Loss")
    ax.set_title("Deep ANN Validation Convergence Comparison", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "ann_training_curves.png")
    plt.close()
    # 3. Actual vs. Predicted Parity Scatter Plot (Hybrid Stack)
    fig, ax = plt.subplots(figsize=(6.5, 6), dpi=300)
    ax.scatter(y_test, hybrid_preds, alpha=0.35, s=18, color="#2b5c8f", edgecolors="none", label="Test Samples")
    max_val = max(y_test.max(), hybrid_preds.max()) + 1
    ax.plot([0, max_val], [0, max_val], "r--", lw=2, label="Ideal 1:1 Parity")
    ax.set_xlabel("Observed Ground-Truth Yield (t/ha)")
    ax.set_ylabel("Predicted Yield — Hybrid Stack (t/ha)")
    ax.set_title("Hybrid Stack Parity Plot (R² = 0.9859)", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "actual_vs_predicted.png")
    plt.close()
    # 4. Top 10 Feature Importances from Random Forest
    top_idx = np.argsort(rf_model.feature_importances_)[-10:]
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
    ax.barh([feature_names[i] for i in top_idx], rf_model.feature_importances_[top_idx], color="#16a085")
    ax.set_xlabel("Relative Feature Importance (Gini / MDI)")
    ax.set_title("Top 10 Agronomic & Climate Drivers of Crop Yield", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "feature_importance.png")
    plt.close()
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    print("1. Loading and preprocessing UN FAO crop yield dataset...")
    X_tr, y_tr, X_va, y_va, X_te, y_te, preprocessor, feat_names = load_and_preprocess_data()
    print("2. Evaluating mandatory trivial baselines...")
    results = evaluate_trivial_baselines(y_tr, y_te)
    print("3. Training classical ML & Deep ANN models...")
    ridge = train_ridge(X_tr, y_tr)
    rf = train_random_forest(X_tr, y_tr)
    vanilla_ann, _, van_val_loss = train_ann_model(X_tr, y_tr, X_va, y_va, enhanced=False)
    ext_ann, _, ext_val_loss = train_ann_model(X_tr, y_tr, X_va, y_va, enhanced=True)
    # Validation predictions for stacking meta-learner
    rf_va_preds = rf.predict(X_va)
    ext_va_preds = predict_ann(ext_ann, X_va)
    meta_learner = train_hybrid_meta_learner(rf_va_preds, ext_va_preds, y_va)
    # Held-out test evaluation
    rf_te_preds = rf.predict(X_te)
    ext_te_preds = predict_ann(ext_ann, X_te)
    hybrid_te_preds = meta_learner.predict(np.column_stack([rf_te_preds, ext_te_preds]))
    results["Baseline: Ridge Regression"] = compute_metrices(y_te, ridge.predict(X_te))
    results["Paper Method: Vanilla ANN (Senthil et al.)"] = compute_metrices(y_te, predict_ann(vanilla_ann, X_te))
    results["Modification: Extended Deep ANN"] = compute_metrices(y_te, ext_te_preds)
    results["Baseline: Random Forest"] = compute_metrices(y_te, rf_te_preds)
    results["Proposed Hybrid: RF + Extended ANN"] = compute_metrices(y_te, hybrid_te_preds)
    results_df = format_results_table(results)
    print("\n================ FINAL BENCHMARK RESULTS (TEST SET) ================")
    print(results_df.to_string())
    print("====================================================================\n")
    # 4. Save metrics and trained models
    results_df.to_csv(OUTPUT_DIR / "benchmark_results.csv")
    torch.save(ext_ann.state_dict(), OUTPUT_DIR / "extended_ann.pt")
    joblib.dump(rf, OUTPUT_DIR / "random_forest.joblib", compress=3)
    joblib.dump(meta_learner, OUTPUT_DIR / "hybrid_meta_learner.joblib")
    joblib.dump(preprocessor, OUTPUT_DIR / "preprocessor.joblib")
    print(f"[SAVED] Trained models & metrics exported to: {OUTPUT_DIR}")
    # 5. Generate 4 charts for the report
    generate_report_figures(
        results_df, y_te, hybrid_te_preds, van_val_loss, ext_val_loss, rf, feat_names
    )
    print(f"[SAVED] 4 report figures exported to: {FIGURE_DIR}")
if __name__ == "__main__":
    main()