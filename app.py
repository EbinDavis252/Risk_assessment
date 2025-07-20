import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Aqua Risk Intelligence", layout="wide")

# ----------------------- DATABASE FUNCTIONS -----------------------
conn = sqlite3.connect('users.db', check_same_thread=False)
c = conn.cursor()

def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT, password TEXT)')

def add_userdata(username, password):
    c.execute('INSERT INTO userstable(username,password) VALUES (?,?)', (username, password))
    conn.commit()

def login_user(username, password):
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?', (username, password))
    return c.fetchall()

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return hashed_text == hashlib.sha256(str.encode(password)).hexdigest()

create_usertable()

# ----------------------- MODEL TRAINING FUNCTION -----------------------
@st.cache_resource
def train_models(farmer_df, water_df):
    # Farmer Loan Default Model
    farmer_df = farmer_df.dropna()
    if 'loan_default' not in farmer_df.columns:
        raise ValueError("Missing 'loan_default' in farmer data.")
    X1 = farmer_df.select_dtypes(include=['number']).drop("loan_default", axis=1)
    y1 = farmer_df["loan_default"]
    if X1.empty or y1.empty:
        raise ValueError("Invalid farmer dataset for training.")
    loan_model = RandomForestClassifier().fit(X1, y1)

    # Fish Farm Failure Model
    water_df = water_df.dropna()
    if 'farm_failure' not in water_df.columns:
        raise ValueError("Missing 'farm_failure' in water quality data.")
    X2 = water_df.select_dtypes(include=['number']).drop("farm_failure", axis=1)
    y2 = water_df["farm_failure"]
    if X2.empty or y2.empty:
        raise ValueError("Invalid water quality dataset for training.")
    farm_model = RandomForestClassifier().fit(X2, y2)

    return loan_model, farm_model

# ----------------------- MAIN APP -----------------------
def main_app():
    st.title("💧 Aqua Risk Intelligence Platform")
    st.markdown("An integrated system to **predict loan defaults** and **fish farm failures** for Aqua Finance companies in India.")

    # --- Upload Datasets ---
    st.sidebar.subheader("📁 Upload Datasets (Optional)")

    uploaded_farmer = st.sidebar.file_uploader("Upload Farmer Profile CSV", type=["csv"])
    uploaded_water = st.sidebar.file_uploader("Upload Water Quality CSV", type=["csv"])

    if uploaded_farmer and uploaded_water:
        farmer_df = pd.read_csv(uploaded_farmer)
        water_df = pd.read_csv(uploaded_water)
        st.success("✅ Uploaded both datasets successfully.")
    else:
        st.sidebar.warning("⚠️ You can enter data manually if no CSVs are uploaded.")
        
        # Manual Entry - Farmer
        st.subheader("📋 Enter Sample Farmer Profile")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            income = st.number_input("Annual Income (₹)", value=200000)
            credit_score = st.number_input("Credit Score", value=700)
        with col2:
            loan_amount = st.number_input("Loan Amount (₹)", value=100000)
            num_loans = st.number_input("Number of Active Loans", value=1)
            past_default = st.selectbox("Past Loan Default?", [0, 1])

        farmer_df = pd.DataFrame([{
            'age': age,
            'income': income,
            'credit_score': credit_score,
            'loan_amount': loan_amount,
            'num_loans': num_loans,
            'past_default': past_default,
            'loan_default': 0  # Dummy label
        }])

        # Manual Entry - Water
        st.subheader("🌊 Enter Sample Water Quality")
        col3, col4 = st.columns(2)
        with col3:
            temperature = st.number_input("Temperature (°C)", value=28.0)
            pH = st.number_input("pH Level", value=7.5)
        with col4:
            ammonia = st.number_input("Ammonia Level (ppm)", value=0.02)
            dissolved_oxygen = st.number_input("Dissolved Oxygen (mg/L)", value=6.5)

        water_df = pd.DataFrame([{
            'temperature': temperature,
            'pH': pH,
            'ammonia': ammonia,
            'dissolved_oxygen': dissolved_oxygen,
            'farm_failure': 0  # Dummy label
        }])

    # Show Data Preview
    with st.expander("📄 Preview Farmer Profile Data"):
        st.dataframe(farmer_df)
    with st.expander("📄 Preview Water Quality Data"):
        st.dataframe(water_df)

    # Train & Predict
    try:
        loan_model, farm_model = train_models(farmer_df, water_df)

        st.success("✅ Models trained successfully!")

        # Predictions
        st.subheader("🔍 Prediction Results")

        if 'loan_default' in farmer_df.columns:
            X_pred1 = farmer_df.drop("loan_default", axis=1)
        else:
            X_pred1 = farmer_df

        if 'farm_failure' in water_df.columns:
            X_pred2 = water_df.drop("farm_failure", axis=1)
        else:
            X_pred2 = water_df

        loan_prediction = loan_model.predict(X_pred1)[0]
        farm_prediction = farm_model.predict(X_pred2)[0]

        colA, colB = st.columns(2)
        with colA:
            st.metric("💸 Loan Default Risk", "High" if loan_prediction else "Low")
        with colB:
            st.metric("🐟 Farm Failure Risk", "High" if farm_prediction else "Low")

    except Exception as e:
        st.error(f"❌ Model training/prediction failed: {e}")

# ----------------------- LOGIN UI -----------------------
def login_page():
    st.title("🔐 AquaRisk Login")
    menu = ["Login", "Register"]
    choice = st.selectbox("Choose", menu)

    if choice == "Login":
        st.subheader("Login to Access Dashboard")
        username = st.text_input("Username")
        password = st.text_input("Password", type='password')
        if st.button("Login"):
            hashed_pwd = make_hashes(password)
            result = login_user(username, check_hashes(password, hashed_pwd))
            if result:
                st.success(f"Welcome {username}")
                main_app()
            else:
                st.error("Invalid Username or Password")

    elif choice == "Register":
        st.subheader("Create New Account")
        new_user = st.text_input("Username")
        new_pwd = st.text_input("Password", type='password')
        if st.button("Register"):
            add_userdata(new_user, make_hashes(new_pwd))
            st.success("You have successfully created an account.")
            st.info("Go to Login to continue.")

# ----------------------- RUN APP -----------------------
if __name__ == '__main__':
    login_page()
