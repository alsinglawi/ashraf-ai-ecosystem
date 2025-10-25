"""
model_training.py
-----------------------------------
Simple modular training pipeline for predicting daily quantity_dispensed
based on simple features. Saves model and evaluation report.

Usage:
    python src/ai_core/model_training.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import matplotlib.pyplot as plt

# Paths
RAW_PATH = Path("data/health_supply_chain/raw.csv")
PREP_PATH = Path("data/health_supply_chain/prepared.csv")
MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "demand_forecast_model.pkl"
PLOT_DIR = Path("reports/plots")
REPORT_FILE = Path("reports/evaluation_report.txt")

def prepare_dataframe(path=RAW_PATH):
    """Load CSV and prepare features. Returns dataframe."""
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"])
    else:
        raise FileNotFoundError(f"{path} not found. Please create sample data first.")
    # Basic cleaning
    df = df.drop_duplicates().dropna().copy()
    # Normalize column names if needed
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Example feature engineering:
    # - Convert date to numeric features (day_of_week, month)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    # - Lag features: previous day's dispensed (simple, requires sorting)
    df = df.sort_values(["facility_name", "item_name", "date"])
    df["prev_dispensed_1"] = df.groupby(["facility_name", "item_name"])["quantity_dispensed"].shift(1).fillna(0)

    # - Basic ratio: fill rate (dispensed / received) if received > 0
    df["fill_rate"] = np.where(df["quantity_received"] > 0,
                               df["quantity_dispensed"] / df["quantity_received"],
                               0.0)

    # Drop rows still with NaNs (if any)
    df = df.dropna().reset_index(drop=True)
    return df

def train_model(df, target_col="quantity_dispensed", feature_cols=None, random_state=42):
    """Train a RandomForest regressor and return model + X_test/y_test/preds"""
    if feature_cols is None:
        # default selection
        feature_cols = ["quantity_received", "stock_on_hand", "lead_time_days",
                        "day_of_week", "month", "prev_dispensed_1", "fill_rate"]

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state
    )

    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    return model, X_test, y_test, preds, feature_cols

def evaluate(y_true, y_pred):
    """Return MAE, RMSE, and MAPE."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    # MAPE: handle zero actuals by using masked array (avoid division by zero)
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    mask = y_true_arr != 0
    if mask.sum() > 0:
        mape = (np.abs((y_true_arr[mask] - y_pred_arr[mask]) / y_true_arr[mask])).mean() * 100
    else:
        mape = np.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

def save_model(model, path=MODEL_FILE):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"✅ Model saved to {path}")

def save_evaluation_report(metrics: dict, report_file=REPORT_FILE):
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"✅ Evaluation report saved to {report_file}")

def plot_predictions(y_true, y_pred, filename=PLOT_DIR / "pred_vs_actual.png"):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.scatter(range(len(y_true)), y_true, label="Actual", alpha=0.6)
    plt.scatter(range(len(y_pred)), y_pred, label="Predicted", alpha=0.6)
    plt.xlabel("Test sample index")
    plt.ylabel("Quantity dispensed")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"✅ Plot saved to {filename}")

def main():
    print("🔁 Preparing data...")
    df = prepare_dataframe(PREP_PATH if PREP_PATH.exists() else RAW_PATH)
    print(f"Rows after prep: {len(df)}")

    print("⚙️ Training model...")
    model, X_test, y_test, preds, feat_cols = train_model(df)

    print("📈 Evaluating...")
    metrics = evaluate(y_test, preds)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) and not np.isnan(v) else f"{k}: {v}")

    save_model(model)
    save_evaluation_report(metrics)
    plot_predictions(y_test, preds)

if __name__ == "__main__":
    main()
