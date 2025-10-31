import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import openai

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Health Supply Chain Forecast Dashboard",
    layout="wide",
    page_icon="📦",
)

st.markdown(
    """
    <style>
    .main {background-color: #f9f9fb;}
    .stApp {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 0 12px rgba(0, 0, 0, 0.05);
    }
    footer {
        text-align: center;
        font-size: 14px;
        color: gray;
        padding-top: 20px;
    }
    .meric-box {
        background-color: #f1f3f6;
        border-radius: 12px;
        padding: 10px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,

)

# -----------------------------
# DATA UPLOAD / LOAD
# -----------------------------
st.title("📊 Health Supply Chain Forecast Dashboard")
st.caption("Monitor and forecast medicine demand in health facilities")

uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

# Define canonical expected names
expected_columns = {
    "date": ["date", "transaction_date", "recorded_date"],
    "facility": ["facility", "facility_name", "clinic", "site"],
    "item": ["item", "product", "medicine", "drug_name"],
    "quantity_dispensed": ["quantity_dispensed", "dispensed", "qty_dispensed", "used_quantity"],
    "quantity_received": ["quantity_received", "received", "qty_received", "incoming_stock"],
    "stock_on_hand": ["stock_on_hand", "stock", "available_stock", "current_stock"],
    "lead_time_days": ["lead_time_days", "leadtime", "delivery_leadtime", "lt_days"]
}

def auto_map_columns(df, expected_map):
    """Try to auto-map user CSV columns to expected names"""
    new_cols = {}
    lower_cols = {c.lower().strip(): c for c in df.columns}
    for target, variants in expected_map.items():
        for v in variants:
            if v.lower() in lower_cols:
                new_cols[lower_cols[v.lower()]] = target
                break
    return df.rename(columns=new_cols)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = auto_map_columns(df, expected_columns)
    missing_cols = [col for col in expected_columns if col not in df.columns]

    if missing_cols:
        st.warning(f"⚠️ Missing columns {missing_cols}. Demo data will be loaded instead.")
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=60),
            "facility": np.random.choice(["Clinic A", "Clinic B"], 60),
            "item": np.random.choice(["Amoxicillin", "Paracetamol"], 60),
            "quantity_dispensed": np.random.randint(50, 200, 60),
            "quantity_received": np.random.randint(60, 250, 60),
            "stock_on_hand": np.random.randint(100, 500, 60),
            "lead_time_days": np.random.randint(2, 12, 60),
        })
else:
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=60),
        "facility": np.random.choice(["Clinic A", "Clinic B"], 60),
        "item": np.random.choice(["Amoxicillin", "Paracetamol"], 60),
        "quantity_dispensed": np.random.randint(50, 200, 60),
        "quantity_received": np.random.randint(60, 250, 60),
        "stock_on_hand": np.random.randint(100, 500, 60),
        "lead_time_days": np.random.randint(2, 12, 60),
    })

# Clean date and derive features
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df.dropna(subset=["date"], inplace=True)
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month

# --- Sidebar Data Summary ---
st.sidebar.markdown("## 📊 Data Overview")

# Compute summary stats
num_facilities = df["facility"].nunique() if "facility" in df.columns else 0
num_items = df["item"].nunique() if "item" in df.columns else 0
date_min = df["date"].min().strftime("%Y-%m-%d") if "date" in df.columns else "N/A"
date_max = df["date"].max().strftime("%Y-%m-%d") if "date" in df.columns else "N/A"
num_records = len(df)

# Display in elegant style
st.sidebar.markdown(
    f"""
    <div style="background-color:#f8f9fa; padding:15px; border-radius:12px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <h4 style="color:#0d6efd; margin-bottom:10px;">📦 Dataset Summary</h4>
        <p><strong>Records:</strong> {num_records:,}</p>
        <p><strong>Facilities:</strong> {num_facilities}</p>
        <p><strong>Items:</strong> {num_items}</p>
        <p><strong>Date Range:</strong><br>{date_min} → {date_max}</p>
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# FILTERS
# -----------------------------
col1, col2, col3 = st.columns(3)
with col1:
    facility = st.selectbox("🏥 Select Facility", ["All"] + sorted(df["facility"].unique().tolist()))
with col2:
    item = st.selectbox("💊 Select Item", ["All"] + sorted(df["item"].unique().tolist()))
with col3:
    date_range = st.date_input(
        "📅 Date Range",
        [df["date"].min(), df["date"].max()],
        min_value=df["date"].min(),
        max_value=df["date"].max(),
    )

# Apply filters
df_filtered = df.copy()
if facility != "All":
    df_filtered = df_filtered[df_filtered["facility"] == facility]
if item != "All":
    df_filtered = df_filtered[df_filtered["item"] == item]
df_filtered = df_filtered[(df_filtered["date"] >= pd.to_datetime(date_range[0])) &
                          (df_filtered["date"] <= pd.to_datetime(date_range[1]))]

# -----------------------------
# MODEL TRAINING
# -----------------------------
target = "quantity_dispensed"
features = ["quantity_received", "stock_on_hand", "lead_time_days", "day_of_week", "month"]

X = df_filtered[features]
y = df_filtered[target]

st.subheader("⚙️ Model Training & Forecasting")

if st.button("Train Model"):
    model = RandomForestRegressor(random_state=42, n_estimators=200)
    model.fit(X, y)
    joblib.dump(model, "rf_model.pkl")
    st.success("✅ Model trained and saved!")

# -----------------------------
# LOAD MODEL
# -----------------------------
if os.path.exists("rf_model.pkl"):
    model = joblib.load("rf_model.pkl")

    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))

    # Display metrics
    st.write("### 📏 Model Performance")
    mcol1, mcol2 = st.columns(2)
    with mcol1:
        st.metric("MAE", f"{mae:.2f}")
    with mcol2:
        st.metric("RMSE", f"{rmse:.2f}")

    # Actual vs Predicted Table
    df_results = df_filtered.copy()
    df_results["predicted"] = preds
    st.dataframe(df_results[["date", "facility", "item", "quantity_dispensed", "predicted"]].tail(10))

    # -----------------------------
    # CHART
    # -----------------------------
    st.write("### 📈 Actual vs Predicted Over Time")
    fig = px.line(
        df_results,
        x="date",
        y=["quantity_dispensed", "predicted"],
        labels={"value": "Quantity", "variable": "Series"},
        title="Actual vs Predicted Quantities Over Time",
        markers=True,
    )
    fig.update_traces(line=dict(width=3))
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # FORECAST NEXT 30 DAYS
    # -----------------------------
    st.write("### 🔮 30-Day Forecast")
    if st.button("Generate 30-Day Forecast"):
        future_dates = [df_filtered["date"].max() + timedelta(days=i) for i in range(1, 31)]
        future_df = pd.DataFrame({
            "date": future_dates,
            "quantity_received": np.random.randint(60, 250, 30),
            "stock_on_hand": np.random.randint(100, 500, 30),
            "lead_time_days": np.random.randint(2, 12, 30),
        })
        future_df["day_of_week"] = [d.dayofweek for d in future_df["date"]]
        future_df["month"] = [d.month for d in future_df["date"]]
        future_preds = model.predict(future_df[features])
        future_df["forecasted_demand"] = future_preds

        st.dataframe(future_df[["date", "forecasted_demand"]])

        fig_future = px.line(
            future_df,
            x="date",
            y="forecasted_demand",
            title="30-Day Demand Forecast",
            markers=True,
        )
        st.plotly_chart(fig_future, use_container_width=True)

# -----------------------------
# ASSISTANT (OPTIONAL)
# -----------------------------
if os.getenv("OPENAI_API_KEY"):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    st.subheader("💬 Assistant")
    user_input = st.text_area("Ask me anything about this data or forecast:")
    if user_input:
        st.info("Thinking...")
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": user_input}],
        )
        st.success(response.choices[0].message.content)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("<footer>🚀 Ashraf AI Ecosystem © 2025</footer>", unsafe_allow_html=True)
