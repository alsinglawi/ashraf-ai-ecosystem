# src/dashboard/app.py
"""
Streamlit dashboard for Ashraf AI Ecosystem
- Upload or view supply chain data
- Display model predictions vs actual
- Interactive filtering (facility, item, date)
- CSV upload, download & model retraining
- Optional AI assistant
Run: streamlit run src/dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import os
import time

# ---------- SETUP ----------
RAW_PATH = Path("data/health_supply_chain/raw.csv")
PREP_PATH = Path("data/health_supply_chain/prepared.csv")
MODEL_PATH = Path("models/demand_forecast_model.pkl")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# Optional assistant
USE_ASSISTANT = bool(os.getenv("OPENAI_API_KEY"))
assistant_available = False
if USE_ASSISTANT:
    try:
        from src.ai_core.assistant_api import AssistantAPI
        assistant = AssistantAPI()
        assistant_available = True
    except Exception as e:
        st.write("Assistant import failed:", e)

st.set_page_config(page_title="Ashraf AI Dashboard", layout="wide")
st.title("📊 Ashraf AI — Supply Chain Intelligence Dashboard")

# ---------- LOAD DATA ----------
@st.cache_data(show_spinner=False)
def load_data():
    path = PREP_PATH if PREP_PATH.exists() else RAW_PATH
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

df_local = load_data()
model = load_model()

# ---------- UPLOAD CSV ----------
st.sidebar.header("📤 Upload or Train Model")

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    try:
        df_uploaded = pd.read_csv(uploaded_file, parse_dates=["date"])
        df_uploaded.columns = [c.strip().lower().replace(" ", "_") for c in df_uploaded.columns]
        st.sidebar.success(f"✅ Uploaded: {uploaded_file.name}")
        df = df_uploaded.copy()
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
        df = df_local.copy()
else:
    df = df_local.copy()

if df.empty:
    st.error("No data found. Upload a CSV file or place one in data/health_supply_chain/.")
    st.stop()

# ---------- SIDEBAR FILTERS ----------
with st.sidebar:
    st.header("🔍 Filters & Actions")
    facilities = ["All"] + sorted(df["facility_name"].unique().tolist())
    items = ["All"] + sorted(df["item_name"].unique().tolist())
    facility = st.selectbox("Facility", facilities)
    item = st.selectbox("Item", items)

    st.write("---")
    st.write("📅 Date range")
    min_date, max_date = df["date"].min(), df["date"].max()
    drange = st.date_input("Select range", [min_date, max_date])
    if len(drange) == 2:
        start_date, end_date = pd.to_datetime(drange[0]), pd.to_datetime(drange[1])
    else:
        start_date, end_date = min_date, max_date

# ---------- FILTER DATA ----------
mask = (df["date"] >= start_date) & (df["date"] <= end_date)
if facility != "All":
    mask &= (df["facility_name"] == facility)
if item != "All":
    mask &= (df["item_name"] == item)
df_view = df.loc[mask].copy()

# ---------- TRAIN NEW MODEL BUTTON ----------
st.sidebar.write("---")
st.sidebar.subheader("🧠 Train or Update Model")

if st.sidebar.button("🚀 Train Model from Current Data"):
    if not df.empty:
        with st.spinner("Training model... please wait ⏳"):
            progress = st.progress(0)
            time.sleep(0.5)
            df_train = df.copy()

            # Feature engineering
            df_train["day_of_week"] = df_train["date"].dt.dayofweek
            df_train["month"] = df_train["date"].dt.month
            df_train = df_train.sort_values(["facility_name", "item_name", "date"])
            df_train["prev_dispensed_1"] = (
                df_train.groupby(["facility_name", "item_name"])["quantity_dispensed"]
                .shift(1)
                .fillna(0)
            )
            df_train["fill_rate"] = np.where(
                df_train["quantity_received"] > 0,
                df_train["quantity_dispensed"] / df_train["quantity_received"],
                0.0,
            )

            feature_cols = [
                "quantity_received", "stock_on_hand", "lead_time_days",
                "day_of_week", "month", "prev_dispensed_1", "fill_rate"
            ]
            df_train = df_train.dropna(subset=["quantity_dispensed"])
            X = df_train[feature_cols].fillna(0)
            y = df_train["quantity_dispensed"]

            # Train
            model_new = RandomForestRegressor(
                n_estimators=150, random_state=42, n_jobs=-1
            )
            model_new.fit(X, y)
            progress.progress(80)

            # Evaluate
            preds = model_new.predict(X)
            mae = mean_absolute_error(y, preds)
            rmse = np.sqrt(mean_squared_error(y, preds))

            # Save
            joblib.dump(model_new, MODEL_PATH)
            progress.progress(100)
            st.success(f"✅ Model retrained and saved!\nMAE: {mae:.2f} | RMSE: {rmse:.2f}")
            model = model_new
    else:
        st.warning("No data available to train a model.")

# ---------- MAIN CONTENT ----------
col1, col2 = st.columns([2, 1])

# --- Left: Data Preview ---
with col1:
    st.subheader("📄 Data Preview")
    st.dataframe(df_view.sort_values(["date"]).reset_index(drop=True), height=300)

    st.subheader("📈 Summary Statistics")
    if not df_view.empty:
        st.write(df_view[["quantity_received", "quantity_dispensed", "stock_on_hand"]].describe().T)
    else:
        st.info("No filtered rows to summarize.")

# --- Right: Metrics ---
compare = None
with col2:
    st.subheader("🤖 Model Status")
    if model is None:
        st.error("No trained model found.")
    else:
        st.success("Model ready ✅")

    if model is not None and not df_view.empty:
        df_feat = df_view.copy()
        df_feat["day_of_week"] = df_feat["date"].dt.dayofweek
        df_feat["month"] = df_feat["date"].dt.month
        df_feat = df_feat.sort_values(["facility_name", "item_name", "date"])
        df_feat["prev_dispensed_1"] = df_feat.groupby(["facility_name", "item_name"])["quantity_dispensed"].shift(1).fillna(0)
        df_feat["fill_rate"] = np.where(
            df_feat["quantity_received"] > 0,
            df_feat["quantity_dispensed"] / df_feat["quantity_received"],
            0.0,
        )

        feature_cols = [
            "quantity_received", "stock_on_hand", "lead_time_days",
            "day_of_week", "month", "prev_dispensed_1", "fill_rate"
        ]
        X = df_feat[feature_cols].fillna(0)
        y = df_feat["quantity_dispensed"].values
        preds = model.predict(X)

        mae = mean_absolute_error(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))
        st.metric("MAE", f"{mae:.2f}")
        st.metric("RMSE", f"{rmse:.2f}")

        compare = pd.DataFrame({
            "date": df_feat["date"].dt.date,
            "actual": y,
            "predicted": np.round(preds, 2)
        }).reset_index(drop=True)
        st.write("Actual vs Predicted (first 10)")
        st.table(compare.head(10))

        # Download predictions
        csv = compare.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv,
            file_name="predictions_filtered.csv",
            mime="text/csv"
        )
    else:
        st.info("Need both model and filtered data to compute metrics.")

# ---------- VISUALIZATION ----------
st.write("---")
st.subheader("📊 Visualization: Actual vs Predicted (Filtered)")

if model is not None and not df_view.empty and compare is not None:
    import altair as alt
    df_plot = compare.copy()
    c = (
        alt.Chart(df_plot)
        .transform_fold(["actual", "predicted"], as_=["series", "value"])
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title="Quantity Dispensed"),
            color=alt.Color("series:N", title="Series"),
            tooltip=["date:T", "series:N", "value:Q"]
        )
        .properties(width=900, height=350)
    )
    st.altair_chart(c, use_container_width=True)
else:
    st.info("No predictions to show.")

# ---------- ASSISTANT ----------
st.write("---")
st.subheader("💬 Ask the Assistant (optional)")

if assistant_available:
    q = st.text_area("Ask Ashraf AI:", value="How can I improve forecasting accuracy?")
    if st.button("Ask"):
        with st.spinner("Thinking..."):
            try:
                reply = assistant.respond(q)
            except Exception as e:
                reply = f"Assistant error: {e}"
        st.write(reply)
else:
    st.info("Assistant not available. Set OPENAI_API_KEY in .env to enable it.")
