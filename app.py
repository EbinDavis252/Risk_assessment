import streamlit as st
import pandas as pd
import sqlite3
import joblib
from datetime import datetime
import os

# ---------- Setup ----------
DB_PATH = 'database/aqua_finance.db'
FIN_MODEL_PATH = 'models/financial_model.pkl'
TECH_MODEL_PATH = 'models/failure_model.pkl'

os.makedirs('database', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# ---------- Database Functions ----------
def create_tables():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS farmer_profiles (
            farmer_id INTEGER PRIMARY KEY,
            name TEXT,
            age INTEGER,
            income REAL,
            region TEXT,
            credit_score INTEGER
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS water_quality (
            farm_id INTEGER,
            timestamp TEXT,
            temperature REAL,
            ph REAL,
            ammonia REAL,
            do REAL,
            turbidity REAL
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS risk_results (
            entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
            farmer_id INTEGER,
            risk_financial REAL,
            risk_technical REAL,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()

def insert_risk_score(farmer_id, fin_score, tech_score):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO risk_results (farmer_id, risk_financial, risk_technical)
        VALUES (?, ?, ?)
    """, (farmer_id, fin_score, tech_score))
    conn.commit()
    conn.close()

create_tables()

# ---------- Load Models ----------
@st.cache_resource
def load_models():
    fin_model = joblib.load(FIN_MODEL_PATH)
    tech_model = joblib.load(TECH_MODEL_PATH)
    return fin_model, tech_model

if os.path.exists(FIN_MODEL_PATH) and os.path.exists(TECH_MODEL_PATH):
    fin_model, tech_model = load_models()
else:
    st.warning("Please upload trained models to the 'models' folder.")

# ---------- Streamlit App ----------
st.set_page_config(layout="wide")
st.title("💧 AI-Driven Risk Assessment System for Aqua Loan Providers")

tab1, tab2 = st.tabs(["📋 Manual Input", "📁 Upload CSV"])

# ----------------------------- TAB 1: MANUAL INPUT -----------------------------
with tab1:
    with st.form("manual_form"):
        st.subheader("Farmer Profile")
        farmer_id = st.number_input("Farmer ID", step=1)
        name = st.text_input("Farmer Name")
        age = st.slider("Age", 18, 70)
        income = st.number_input("Annual Income (INR)", value=100000)
        farm_size = st.number_input("Farm Size (in acres)", value=1.0)
        loan_amount = st.number_input("Loan Amount", value=50000)
        region = st.selectbox("Region", ["Andhra", "TamilNadu", "Odisha", "West Bengal"])
        credit_score = st.slider("Credit Score", 300, 850)
        loan_term = st.slider("Loan Tenure (months)", 6, 36)
        previous_defaults = st.radio("Previous Defaults", [0, 1])

        st.subheader("Water Quality")
        temp = st.number_input("Temperature (°C)", 20.0, 35.0, 28.0)
        ph = st.number_input("pH", 6.0, 9.0, 7.2)
        ammonia = st.number_input("Ammonia (mg/L)", 0.0, 5.0, 0.5)
        do = st.number_input("Dissolved Oxygen (mg/L)", 0.0, 10.0, 5.0)
        turbidity = st.number_input("Turbidity (NTU)", 0.0, 10.0, 3.0)

        submit_manual = st.form_submit_button("🔍 Assess Risk")

    if submit_manual and 'fin_model' in globals():
        # Prepare inputs
        fin_features = [[age, income, farm_size, loan_amount, previous_defaults, credit_score, loan_term]]
        tech_features = [[temp, ph, ammonia, do, turbidity]]

        # Predict
        fin_score = round(fin_model.predict_proba(fin_features)[0][1], 2)
        tech_score = round(tech_model.predict_proba(tech_features)[0][1], 2)

        # Show results
        st.success(f"📊 Loan Default Risk: {fin_score*100:.1f}%")
        st.warning(f"💧 Fish Farm Failure Risk: {tech_score*100:.1f}%")

        # Save to DB
        insert_risk_score(farmer_id, fin_score, tech_score)

# ----------------------------- TAB 2: UPLOAD FILES -----------------------------
with tab2:
    st.subheader("Upload Aqua Loan Dataset & Water Quality Dataset")

    loan_file = st.file_uploader("Upload aqua_loans.csv", type=["csv"])
    water_file = st.file_uploader("Upload water_quality.csv", type=["csv"])

    if loan_file:
        loan_df = pd.read_csv(loan_file)
        st.dataframe(loan_df.head())

    if water_file:
        water_df = pd.read_csv(water_file)
        st.dataframe(water_df.head())

    if st.button("🔍 Run Batch Risk Assessment") and 'fin_model' in globals():
        if loan_file and water_file:
            results = []

            for i in range(min(len(loan_df), len(water_df))):
                f = loan_df.iloc[i]
                w = water_df.iloc[i]

                fin_features = [[
                    f['age'], f['income'], f['farm_size'], f['loan_amount'],
                    f['previous_defaults'], f['credit_score'], f['loan_term_months']
                ]]
                tech_features = [[
                    w['temperature'], w['ph'], w['ammonia'], w['do'], w['turbidity']
                ]]

                fin_score = round(fin_model.predict_proba(fin_features)[0][1], 2)
                tech_score = round(tech_model.predict_proba(tech_features)[0][1], 2)

                results.append({
                    'farmer_id': f['farmer_id'],
                    'Loan Default Risk (%)': fin_score * 100,
                    'Farm Failure Risk (%)': tech_score * 100
                })

                insert_risk_score(f['farmer_id'], fin_score, tech_score)

            result_df = pd.DataFrame(results)
            st.success("✅ Batch Assessment Complete")
            st.dataframe(result_df)
        else:
            st.warning("Please upload both CSV files.")

# ----------------------------- END -----------------------------
