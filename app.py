import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ---------- DB Setup ----------
conn = sqlite3.connect('users.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)''')
conn.commit()

# ---------- User Auth ----------
def register_user(username, password):
    c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
    conn.commit()

def login_user(username, password):
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    return c.fetchone()

# ---------- Model Trainer ----------
@st.cache_data
def train_models(farmer_df, water_df):
    X1 = farmer_df.drop('LoanDefault', axis=1)
    y1 = farmer_df['LoanDefault']
    loan_model = RandomForestClassifier().fit(X1, y1)

    X2 = water_df.drop('Failure', axis=1)
    y2 = water_df['Failure']
    farm_model = RandomForestClassifier().fit(X2, y2)

    return loan_model, farm_model

# ---------- Login UI ----------
def show_login_popup():
    with st.modal("Login to Access Dashboard"):
        login_tab, register_tab = st.tabs(["🔐 Login", "📝 Register"])

        with login_tab:
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login"):
                user = login_user(username, password)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.user = username
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid Username or Password")

        with register_tab:
            new_user = st.text_input("Create Username", key="reg_user")
            new_pass = st.text_input("Create Password", type="password", key="reg_pass")
            if st.button("Register"):
                if new_user and new_pass:
                    try:
                        register_user(new_user, new_pass)
                        st.success("Registered! Please login.")
                    except sqlite3.IntegrityError:
                        st.error("Username already exists.")
                else:
                    st.warning("Fill both fields.")

# ---------- Main Dashboard ----------
def main():
    st.set_page_config("Aqua Risk Intelligence", layout="wide")

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        show_login_popup()
        return

    st.sidebar.title("🚪 Logout")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.title("🐟 Aqua Risk Intelligence Dashboard")

    # Upload Section
    st.header("📤 Upload Datasets")
    col1, col2 = st.columns(2)

    with col1:
        farmer_file = st.file_uploader("Upload Farmer Profiles CSV", type="csv")
    with col2:
        water_file = st.file_uploader("Upload Water Quality CSV", type="csv")

    if farmer_file is not None and water_file is not None:
        farmer_df = pd.read_csv(farmer_file)
        water_df = pd.read_csv(water_file)
        st.success("✅ Files uploaded successfully.")
    else:
        st.warning("Using manual sample data. You can also upload your files above.")
        farmer_df = pd.DataFrame({
            'Age': [25, 30, 45], 'Experience': [5, 10, 20],
            'LoanAmount': [100000, 150000, 200000], 'CreditScore': [650, 700, 620],
            'LoanDefault': [0, 0, 1]
        })
        water_df = pd.DataFrame({
            'Temperature': [25, 30, 35], 'pH': [7, 6.5, 8],
            'Ammonia': [0.1, 0.3, 0.5], 'DO': [6.5, 5.2, 4.0],
            'Failure': [0, 0, 1]
        })

    # Optional Manual Edits
    with st.expander("✏️ Manually Edit Farmer Data"):
        farmer_df = st.data_editor(farmer_df, num_rows="dynamic")

    with st.expander("✏️ Manually Edit Water Quality Data"):
        water_df = st.data_editor(water_df, num_rows="dynamic")

    # Train Models
    loan_model, farm_model = train_models(farmer_df, water_df)

    # Insights
    st.header("📊 Risk Insights")

    with st.expander("🎯 Predict Loan Default for New Farmer"):
        age = st.slider("Age", 18, 65, 30)
        exp = st.slider("Experience", 0, 40, 5)
        loan = st.number_input("Loan Amount", 50000, 500000, 100000)
        credit = st.slider("Credit Score", 300, 900, 650)
        input1 = pd.DataFrame([[age, exp, loan, credit]],
                              columns=['Age', 'Experience', 'LoanAmount', 'CreditScore'])
        pred1 = loan_model.predict(input1)[0]
        st.info(f"Loan Default Risk: {'❌ High Risk' if pred1 else '✅ Low Risk'}")

    with st.expander("🌊 Predict Farm Failure Based on Water Quality"):
        temp = st.slider("Temperature (°C)", 15, 40, 25)
        ph = st.slider("pH", 5.0, 9.0, 7.0)
        ammonia = st.slider("Ammonia (mg/L)", 0.0, 1.0, 0.2)
        do = st.slider("Dissolved Oxygen (mg/L)", 2.0, 10.0, 6.0)
        input2 = pd.DataFrame([[temp, ph, ammonia, do]],
                              columns=['Temperature', 'pH', 'Ammonia', 'DO'])
        pred2 = farm_model.predict(input2)[0]
        st.info(f"Farm Failure Risk: {'❌ High Risk' if pred2 else '✅ Low Risk'}")

    # Reports
    st.subheader("📈 Model Performance (on uploaded/manual data)")
    st.text("Loan Default Model:")
    X1 = farmer_df.drop('LoanDefault', axis=1)
    y1 = farmer_df['LoanDefault']
    st.code(classification_report(y1, loan_model.predict(X1)))

    st.text("Farm Failure Model:")
    X2 = water_df.drop('Failure', axis=1)
    y2 = water_df['Failure']
    st.code(classification_report(y2, farm_model.predict(X2)))


if __name__ == '__main__':
    main()
