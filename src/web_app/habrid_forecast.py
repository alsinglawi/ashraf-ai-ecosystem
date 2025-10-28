import pandas as pd
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def train_hybrid_model(df, date_col, y_col, extra_features=None, periods=30):
    df = df[[date_col, y_col] + (extra_features or [])].copy()
    df.rename(columns={date_col: "ds", y_col: "y"}, inplace=True)
    df.sort_values("ds", inplace=True)

    # 1️⃣ Train Prophet
    model_prophet = Prophet()
    if extra_features:
        for feat in extra_features:
            model_prophet.add_regressor(feat)
    model_prophet.fit(df)

    # 2️⃣ Prophet forecast
    future = model_prophet.make_future_dataframe(periods=periods)
    if extra_features:
        for feat in extra_features:
            future[feat] = df[feat].iloc[-1]  # Use last known values
    forecast = model_prophet.predict(future)

    # 3️⃣ Compute residuals
    df["prophet_pred"] = forecast.loc[: len(df) - 1, "yhat"].values
    df["residual"] = df["y"] - df["prophet_pred"]

    # 4️⃣ Train Random Forest on residuals
    X = df.index.values.reshape(-1, 1)
    y_res = df["residual"].values
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X, y_res)

    # 5️⃣ Predict residuals for future
    future_idx = np.arange(len(df), len(df) + periods).reshape(-1, 1)
    future_residuals = rf.predict(future_idx)

    # 6️⃣ Combine Prophet + RF predictions
    forecast["hybrid_yhat"] = forecast["yhat"]
    forecast.loc[len(df):, "hybrid_yhat"] += future_residuals

    return forecast.tail(periods), model_prophet, rf


def evaluate_model(df, forecast):
    y_true = df["y"].values
    y_pred = df["prophet_pred"].values
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse
