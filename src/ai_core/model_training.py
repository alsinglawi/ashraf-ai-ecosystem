"""
model_training.py
-----------------------------------
Handles model training, saving, and prediction.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib


class ModelTrainer:
    def __init__(self, model_path="models/demand_forecast_model.pkl"):
        self.model_path = model_path

    def train(self, df, target_col, feature_cols):
        """Train a simple regression model"""
        X = df[feature_cols]
        y = df[target_col]
        model = LinearRegression()
        model.fit(X, y)
        joblib.dump(model, self.model_path)
        print(f"✅ Model saved to {self.model_path}")

    def predict(self, df):
        """Load model and make predictions"""
        model = joblib.load(self.model_path)
        predictions = model.predict(df)
        return predictions
