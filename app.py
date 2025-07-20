import streamlit as st
import pandas as pd
import numpy as np
import joblib
import bcrypt
import sqlite3
from datetime import datetime

# ---------- DATABASE SETUP ----------
conn = sqlite3.connect("database/users.db", check_same_thread=False)
c = conn.cursor()

def create_tables():
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    password TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS farmer_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    name TEXT,
                    income REAL,
                    farm_size REAL,
                    loan_amount REAL,
                    region TEXT,
                    prev_defaults INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS water_quality (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    ph REAL,
                    temperature REAL,
                    ammonia REAL,
                    do_level REAL,
                    turbidity REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS model_outputs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    risk_score_financial REAL,
                    risk_score_technical REAL,
                    risk_label TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()

create_tables()

# ---------- AUTH ----------
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed)

def register_user(username, password):
    hashed_pw = hash_password(password)
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
        conn.commit()
        return True
    except:
        return False

def login_user(username, password):
    c.execute("SELECT id, password FROM users WHERE username=?", (username,))
    data = c.fetchone()
    if data and verify_password(password, data[1]):
        return data[0]  # return user_id
    return None

# ---------- MODELS ----------
loan_model = joblib.load('ml_models/loan_model.pkl')
failure_model = joblib.load('ml_models/failure_model.pkl')

# ---------- APP ----------
st.set_page_config(page_title="Aqua Risk Assessment", layout="wide")
st.title("💧 AI-Driven Risk Assessment System for Aqua Loan Providers")

menu = ["Login", "Register"]
choice = st.sidebar.selectbox("Menu", menu)

if 'user_id' not in st.session_state:
    st.session_state.user_id = None

if choice == "Register":
    st.subheader("Create Account")
    new_user = st.text_input("Username")
    new_password = st.text_input("Password", type="password")
    if st.button("Register"):
        if register_user(new_user, new_password):
            st.success("Account created. Go to Login.")
        else:
            st.error("Username already exists.")

elif choice == "Login":
    st.subheader("Login to Your Account")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user_id = login_user(username, password)
        if user_id:
            st.success(f"Welcome, {username}!")
            st.session_state.user_id = user_id
        else:
            st.error("Invalid credentials")

# ---------- MAIN APP AFTER LOGIN ----------
if st.session_state.user_id:
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload Data", "🔍 Predict Risk", "📈 View History", "ℹ️ About"])

    with tab1:
        st.header("Upload Farmer Loan Data")
        loan_file = st.file_uploader("Upload CSV for Loan Data", type=["csv"], key="loan")
        if loan_file:
            df_loan = pd.read_csv(loan_file)
            st.dataframe(df_loan)

            for _, row in df_loan.iterrows():
                c.execute('''INSERT INTO farmer_profiles (user_id, name, income, farm_size, loan_amount, region, prev_defaults)
                             VALUES (?, ?, ?, ?, ?, ?, ?)''',
                          (st.session_state.user_id, row['name'], row['income'], row['farm_size'],
                           row['loan_amount'], row['region'], row['prev_defaults']))
            conn.commit()
            st.success("Loan data uploaded successfully.")

        st.header("Upload Water Quality Data")
        water_file = st.file_uploader("Upload CSV for Water Data", type=["csv"], key="water")
        if water_file:
            df_water = pd.read_csv(water_file)
            st.dataframe(df_water)

            for _, row in df_water.iterrows():
                c.execute('''INSERT INTO water_quality (user_id, ph, temperature, ammonia, do_level, turbidity)
                             VALUES (?, ?, ?, ?, ?, ?)''',
                          (st.session_state.user_id, row['ph'], row['temperature'], row['ammonia'],
                           row['do_level'], row['turbidity']))
            conn.commit()
            st.success("Water data uploaded successfully.")

    with tab2:
        st.header("Run Risk Predictions")

        st.subheader("Latest Farmer Profile")
        c.execute('''SELECT income, farm_size, loan_amount, prev_defaults 
                     FROM farmer_profiles WHERE user_id=? ORDER BY id DESC LIMIT 1''', (st.session_state.user_id,))
        row = c.fetchone()
        if row:
            input_financial = np.array(row).reshape(1, -1)
            risk_score_fin = loan_model.predict_proba(input_financial)[0][1]
        else:
            risk_score_fin = None
            st.warning("No loan data found.")

        st.subheader("Latest Water Quality")
        c.execute('''SELECT ph, temperature, ammonia, do_level, turbidity
                     FROM water_quality WHERE user_id=? ORDER BY id DESC LIMIT 1''', (st.session_state.user_id,))
        row = c.fetchone()
        if row:
            input_tech = np.array(row).reshape(1, -1)
            risk_score_tech = failure_model.predict_proba(input_tech)[0][1]
        else:
            risk_score_tech = None
            st.warning("No water data found.")

        if risk_score_fin is not None and risk_score_tech is not None:
            st.metric("📊 Financial Risk Score", f"{risk_score_fin:.2f}")
            st.metric("🌊 Technical Risk Score", f"{risk_score_tech:.2f}")
            risk_label = "High" if risk_score_fin > 0.6 or risk_score_tech > 0.6 else "Medium" if risk_score_fin > 0.4 else "Low"
            st.success(f"Overall Risk Assessment: **{risk_label}**")

            c.execute('''INSERT INTO model_outputs (user_id, risk_score_financial, risk_score_technical, risk_label)
                         VALUES (?, ?, ?, ?)''',
                      (st.session_state.user_id, risk_score_fin, risk_score_tech, risk_label))
            conn.commit()
        else:
            st.info("Upload both loan and water data to generate prediction.")

    with tab3:
        st.header("Historical Predictions")
        df = pd.read_sql_query('''SELECT * FROM model_outputs WHERE user_id=? ORDER BY created_at DESC''', conn,
                               params=(st.session_state.user_id,))
        st.dataframe(df)

    with tab4:
        st.markdown("""
        ### About This Project
        **AI-Driven Risk Assessment System for Aqua Loan Providers** combines machine learning models to assess both:
        - 💼 Financial credit risk of fish farmers.
        - 🌊 Technical failure risk of fish farms based on water quality.

        Developed by Ebin Davis as part of MSc. Finance & Analytics internship at Grant Thornton Bharat LLP.
        """)

