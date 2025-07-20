# app.py (Self-contained with in-code datasets)
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime

st.set_page_config(page_title="Aqua Risk Intelligence System", layout="wide")
DB_PATH = "aqua_risk.db"

# ---------------------------------------
# Step 1: Create simulated datasets
# ---------------------------------------
@st.cache_data
def load_simulated_data():
    np.random.seed(42)
    n = 500

    farmer_df = pd.DataFrame({
        "annual_income": np.random.randint(100000, 500000, size=n),
        "loan_amount": np.random.randint(50000, 300000, size=n),
        "credit_score": np.random.randint(300, 850, size=n),
        "past_defaults": np.random.randint(0, 3, size=n),
        "loan_default": np.random.choice([0, 1], size=n, p=[0.8, 0.2])
    })

    water_df = pd.DataFrame({
        "temperature": np.random.normal(28, 2, size=n),
        "pH": np.random.normal(7.5, 0.5, size=n),
        "ammonia": np.random.exponential(0.2, size=n),
        "dissolved_oxygen": np.random.normal(6, 1, size=n),
        "turbidity": np.random.normal(15, 5, size=n)
    })
    water_df["farm_failure"] = ((water_df["ammonia"] > 0.5) |
                                (water_df["dissolved_oxygen"] < 4) |
                                (water_df["pH"] < 6.5) |
                                (water_df["pH"] > 8.5)).astype(int)

    return farmer_df, water_df

farmer_df, water_df = load_simulated_data()

# ---------------------------------------
# Step 2: Train Models
# ---------------------------------------
@st.cache_resource
def train_models(farmer_df, water_df):
    X1 = farmer_df.drop("loan_default", axis=1)
    y1 = farmer_df["loan_default"]
    model1 = RandomForestClassifier().fit(X1, y1)

    X2 = water_df.drop("farm_failure", axis=1)
    y2 = water_df["farm_failure"]
    model2 = RandomForestClassifier().fit(X2, y2)

    return model1, model2

loan_model, farm_model = train_models(farmer_df, water_df)

# ---------------------------------------
# Step 3: DB Setup
# ---------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS model_outputs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        age INTEGER,
        region TEXT,
        income INTEGER,
        loan_amount INTEGER,
        credit_score INTEGER,
        past_defaults INTEGER,
        temperature REAL,
        pH REAL,
        ammonia REAL,
        do_level REAL,
        turbidity REAL,
        financial_risk REAL,
        technical_risk REAL,
        result_time TEXT
    )
    ''')
    conn.commit()
    conn.close()

def save_result(data):
    conn = sqlite3.connect(DB_PATH)
    pd.DataFrame([data]).to_sql("model_outputs", conn, if_exists="append", index=False)
    conn.close()

def load_history():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM model_outputs ORDER BY result_time DESC", conn)
    conn.close()
    return df

# ---------------------------------------
# Step 4: Streamlit UI
# ---------------------------------------
st.title("🌊 Aqua Risk Intelligence Dashboard")
tab1, tab2, tab3 = st.tabs(["📝 Risk Assessment", "📊 Risk Summary", "🗃️ History"])

with tab1:
    st.header("📝 Enter Farmer & Water Quality Details")

    with st.form("input_form"):
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Farmer Name")
            age = st.slider("Age", 18, 70, 35)
            region = st.selectbox("Region", ["Andhra Pradesh", "Tamil Nadu", "Odisha", "West Bengal", "Gujarat"])
            income = st.number_input("Annual Income (INR)", 10000, 1000000, 250000)
            loan_amt = st.number_input("Loan Amount (INR)", 10000, 500000, 150000)
            credit_score = st.slider("Credit Score", 300, 850, 650)
            past_defaults = st.selectbox("Past Loan Defaults", [0, 1, 2])

        with col2:
            temperature = st.number_input("Water Temperature (°C)", 20.0, 35.0, 28.0)
            ph = st.number_input("pH Level", 4.0, 10.0, 7.5)
            ammonia = st.number_input("Ammonia (mg/L)", 0.0, 2.0, 0.2)
            do_level = st.number_input("Dissolved Oxygen (mg/L)", 0.0, 10.0, 6.0)
            turbidity = st.number_input("Turbidity (NTU)", 0.0, 50.0, 15.0)

        submit = st.form_submit_button("Evaluate Risk")

    if submit:
        fin_input = pd.DataFrame([[income, loan_amt, credit_score, past_defaults]],
                                 columns=["annual_income", "loan_amount", "credit_score", "past_defaults"])
        tech_input = pd.DataFrame([[temperature, ph, ammonia, do_level, turbidity]],
                                  columns=["temperature", "pH", "ammonia", "dissolved_oxygen", "turbidity"])

        fin_risk = loan_model.predict_proba(fin_input)[0][1]
        tech_risk = farm_model.predict_proba(tech_input)[0][1]

        st.success("✔️ Risk Evaluation Complete")
        st.metric("💰 Financial Risk (Loan Default)", f"{fin_risk:.2%}")
        st.metric("🐟 Technical Risk (Farm Failure)", f"{tech_risk:.2%}")

        result = {
            "name": name,
            "age": age,
            "region": region,
            "income": income,
            "loan_amount": loan_amt,
            "credit_score": credit_score,
            "past_defaults": past_defaults,
            "temperature": temperature,
            "pH": ph,
            "ammonia": ammonia,
            "do_level": do_level,
            "turbidity": turbidity,
            "financial_risk": round(fin_risk, 4),
            "technical_risk": round(tech_risk, 4),
            "result_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        save_result(result)

with tab2:
    st.header("📊 Summary")
    st.info("This section can be enhanced with visualizations like charts and filters.")

with tab3:
    st.header("🗃️ Prediction History")
    df = load_history()
    st.dataframe(df, use_container_width=True)

# Initialize DB on first run
if not os.path.exists(DB_PATH):
    init_db()
