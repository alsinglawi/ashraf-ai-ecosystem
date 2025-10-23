"""
data_pipeline.py
-----------------------------------
Manages data ingestion and preparation for forecasting, analytics, and AI models.
"""

import pandas as pd
from pathlib import Path


class DataPipeline:
    def __init__(self, raw_data_path="data/health_supply_chain/raw.csv"):
        self.raw_data_path = Path(raw_data_path)

    def load_data(self):
        """Load raw data from CSV"""
        try:
            df = pd.read_csv(self.raw_data_path)
            print(f"✅ Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except FileNotFoundError:
            print("❌ Data file not found.")
            return pd.DataFrame()

    def clean_data(self, df: pd.DataFrame):
        """Perform basic cleaning"""
        df = df.dropna().drop_duplicates()
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df

    def save_prepared_data(self, df: pd.DataFrame, output_path="data/health_supply_chain/prepared.csv"):
        """Save cleaned data"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✅ Prepared data saved to {output_path}")


if __name__ == "__main__":
    pipeline = DataPipeline()
    data = pipeline.load_data()
    if not data.empty:
        clean = pipeline.clean_data(data)
        pipeline.save_prepared_data(clean)
