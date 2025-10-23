"""
report_generation.py
-----------------------------------
Generates automated summary reports from analytics results.
"""

import pandas as pd
from datetime import datetime

def generate_report(df, filename="reports/summary_report.csv"):
    df_summary = df.describe(include="all")
    df_summary.to_csv(filename)
    print(f"✅ Report generated at {filename} on {datetime.now()}")

