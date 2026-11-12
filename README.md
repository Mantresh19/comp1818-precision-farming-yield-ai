# Precision Agriculture Crop Yield Prediction (COMP1818)

An end-to-end machine learning and deep learning pipeline for global crop yield prediction (`tonnes/hectare`), replicating and extending recent precision agriculture research on **28,242 real UN FAO / World Bank agricultural records** across 101 countries and 10 staple crops.

## Project Structure

comp1818-ai-coursework/
├── data/crop_yield.csv # UN FAO / World Bank dataset (28,242 records)
├── figures/ # Generated 300-DPI evaluation & diagnostic charts
├── outputs/ # Saved trained models (.pt, .joblib) & metrics CSV
├── src/
│ ├── data_loader.py # Zero-leakage 65/15/20 splitting, scaling & encoding
│ ├── evaluate.py # Regression metrics (RMSE, MAE, R2, MAPE) & trivial baselines
│ └── models.py # Ridge, Random Forest, PyTorch ANNs, and Hybrid Meta-Learner
├── train.py # One-click end-to-end training & evaluation entry point
└── requirements.txt # Cross-platform Python dependencies

## How to Replicate Locally

1. Clone the repository and navigate inside:

   git clone https://github.com/Mantresh19/comp1818-precision-farming-yield-ai.git
   cd comp1818-precision-farming-yield-ai

2. Install dependencies (Python 3.10+):

   pip install -r requirements.txt

3. Run the complete training, evaluation, and plotting pipeline:

   python train.py

   This reproduces the benchmark table, exports trained models to `outputs/`, and generates all 4 analytical figures in `figures/`.

## References & Data Attribution

- Primary Research Anchor: Senthil, G. A., Suganthi, S. U., Prinslin, L., Selvi, R., & Prabha, R. (2025). Generative AI in Agri: Sustainability in Smart Precision Farming Yield Prediction Mapping System Based on GIS Using Deep Learning and GPS. Procedia Computer Science, 252, 365–380.
- Supporting Literature: Saha, S., Kucher, O. D., Utkina, A. O., & Rebouh, N. Y. (2025). Precision agriculture for improving crop yield predictions: a literature review\*. Frontiers in Agronomy, 7, 1566201.
- Dataset Source: Food and Agriculture Organization (FAO) & World Bank Crop Yield Dataset (`yield_df.csv`, 1990–2013), retrieved via open-source ML benchmark repository.
