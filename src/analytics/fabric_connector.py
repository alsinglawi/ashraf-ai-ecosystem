"""
fabric_connector.py
-----------------------------------
Connects to Microsoft Fabric Lakehouse and retrieves datasets.
"""

import os
import pyodbc
from dotenv import load_dotenv

load_dotenv()

class FabricConnector:
    def __init__(self):
        self.conn_str = os.getenv("DB_CONNECTION_STRING")

    def fetch_data(self, query):
        with pyodbc.connect(self.conn_str) as conn:
            df = pd.read_sql(query, conn)
        print(f"✅ Retrieved {df.shape[0]} rows from Fabric Lakehouse.")
        return df
