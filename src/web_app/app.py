# src/dashboard/app.py
"""
Streamlit dashboard for Ashraf AI Ecosystem
- View supply chain data summary
- Show model predictions vs actual
- Interactive filtering (facility, item, date)
- Optional AI assistant (if OPENAI_API_KEY set)
Run: streamlit run src/dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# Optional: load assistant if available
USE_ASSISTANT = bool(os.getenv("OPENAI_API_KEY"))
assistant_available = False
if USE_ASSISTANT:
    try:
        from src.ai_core.assistant_api import AssistantAPI
        assistant = AssistantAPI()
        assistant_available = True
    except Exception as e:
        st.write("Assistant import failed:", e)

# File paths
RAW_PATH = Path("data/health_supply_chain/raw.csv")
PREP_PATH = Path("data/health_supply_chain/prepared.csv")
MODEL_PATH = Path("models/demand_forecast_model.pkl")

# Streamlit page setup
st.set_page_config(page_title="Ashraf AI Dashboard", layout="wide")
st.title("📊 Ashraf AI — Supply Chain Dashboard")

# ---------- LOAD DATA & MODEL ----------
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

df = load_data()
model = load_model()

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("🔍 Filters & Actions")
    if df.empty:
        st.warning("No data found in data/health_supply_chain/")
    else:
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
if not df.empty:
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    if facility != "All":
        mask &= (df["facility_name"] == facility)
    if item != "All":
        mask &= (df["item_name"] == item)
    df_view = df.loc[mask].copy()
else:
    df_view = pd.DataFrame()

# ---------- MAIN DASHBOARD ----------
col1, col2 = st.columns([2, 1])

# --- Left column: Data Preview ---
with col1:
    st.subheader("📄 Data preview")
    if not df_view.empty:
        st.dataframe(df_view.sort_values(["date"]).reset_index(drop=True), height=300)
    else:
        st.info("No data available for this selection.")

    st.subheader("📈 Summary statistics")
    if not df_view.empty:
        st.write(df_view[["quantity_received", "quantity_dispensed", "stock_on_hand"]].describe().T)
    else:
        st.info("No filtered rows to summarize.")

# --- Right column: Model info & metrics ---
with col2:
    st.subheader("🤖 Model status")
    if model is None:
        st.error("No trained model found at models/demand_forecast_model.pkl")
    else:
        st.success("Model loaded successfully ✅")

    st.write("---")
    st.subheader("Quick metrics (filtered view)")

    if model is not None and not df_view.empty:
        # Build features (consistent with training)
        df_feat = df_view.copy()
        df_feat["day_of_week"] = df_feat["date"].dt.dayofweek
        df_feat["month"] = df_feat["date"].dt.month
        df_feat = df_feat.sort_values(["facility_name", "item_name", "date"])
        df_feat["prev_dispensed_1"] = df_feat.groupby(["facility_name", "item_name"])["quantity_dispensed"].shift(1).fillna(0)
        df_feat["fill_rate"] = np.where(df_feat["quantity_received"] > 0,
                                        df_feat["quantity_dispensed"] / df_feat["quantity_received"], 0.0)

        feature_cols = [
            "quantity_received", "stock_on_hand", "lead_time_days",
            "day_of_week", "month", "prev_dispensed_1", "fill_rate"
        ]
        X = df_feat[feature_cols].fillna(0)
        y = df_feat["quantity_dispensed"].values
        preds = model.predict(X)

        # Metrics
        mae = mean_absolute_error(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))
        st.metric("MAE", f"{mae:.2f}")
        st.metric("RMSE", f"{rmse:.2f}")

        # Display actual vs predicted table
        st.write("Actual vs Predicted (first 10)")
        compare = pd.DataFrame({
            "date": df_feat["date"].dt.date,
            "actual": y,
            "predicted": np.round(preds, 2)
        }).reset_index(drop=True)
        st.table(compare.head(10))
    else:
        st.info("Need both model and filtered data to compute metrics.")

# ---------- VISUALIZATION ----------
st.write("---")
st.subheader("📊 Visualization: Actual vs Predicted (filtered)")

if model is not None and not df_view.empty:
    import altair as alt

    df_plot = compare.copy()
    df_plot["index"] = range(len(df_plot))

    # ✅ Improved Altair chart (explicit data types + tooltips)
    c = (
        alt.Chart(df_plot)
        .transform_fold(
            ["actual", "predicted"],
            as_=["series", "value"]
        )
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

# ---------- ASSISTANT SECTION ----------
st.write("---")
st.subheader("💬 Ask the Assistant (optional)")

if assistant_available:
    q = st.text_area("Ask Ashraf AI:", value="How can I reduce stockouts for Clinic A?")
    if st.button("Ask"):
        with st.spinner("Thinking..."):
            try:
                reply = assistant.respond(q)
            except Exception as e:
                reply = f"Assistant error: {e}"
        st.write(reply)
else:
    st.info("Assistant not available. Set OPENAI_API_KEY in .env and ensure src.ai_core.assistant_api exists if you want to use it.")
