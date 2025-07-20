import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
from sklearn.ensemble import RandomForestClassifier

# ------------------ USER AUTH FUNCTIONS ------------------

def create_usertable():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT, password TEXT)')
    conn.commit()
    conn.close()

def add_userdata(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('INSERT INTO userstable(username, password) VALUES (?, ?)', (username, password))
    conn.commit()
    conn.close()

def login_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM userstable WHERE username = ? AND password = ?', (username, password))
    data = c.fetchall()
    conn.close()
    return data

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ------------------ MODEL TRAINING ------------------

@st.cache_data
def train_models(farmer_df, water_df):
    loan_features = ['credit_score', 'loan_amount', 'loan_tenure', 'past_defaults']
    farm_features = ['temperature', 'ph', 'ammonia', 'dissolved_oxygen']

    X1 = farmer_df[loan_features]
    y1 = farmer_df['default']
    loan_model = RandomForestClassifier().fit(X1, y1)

    X2 = water_df[farm_features]
    y2 = water_df['failure']
    farm_model = RandomForestClassifier().fit(X2, y2)

    return loan_model, farm_model

# ------------------ MAIN APPLICATION ------------------

def main():
    st.set_page_config(page_title="AquaRisk AI", layout="wide")
    st.title("💧 Aqua Finance Risk Assessment & Early Warning System")

    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = ""
    if 'farmer_df' not in st.session_state:
        st.session_state.farmer_df = None
    if 'water_df' not in st.session_state:
        st.session_state.water_df = None

    create_usertable()

    # -------------------- LOGIN PAGE --------------------
    if not st.session_state.logged_in:
        menu = st.sidebar.selectbox("Access Panel", ["Login", "Register"])

        if menu == "Login":
            st.subheader("Login to Access Dashboard")
            username = st.text_input("Username")
            password = st.text_input("Password", type='password')
            if st.button("Login"):
                hashed_pw = hash_password(password)
                result = login_user(username, hashed_pw)
                if result:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success(f"Welcome {username} 👋")
                    st.experimental_rerun()
                else:
                    st.error("Invalid Username or Password")

        elif menu == "Register":
            st.subheader("Create New Account")
            new_user = st.text_input("Username")
            new_password = st.text_input("Password", type='password')
            if st.button("Register"):
                if new_user and new_password:
                    hashed_new_pw = hash_password(new_password)
                    add_userdata(new_user, hashed_new_pw)
                    st.success("Account created! Please login.")
                else:
                    st.warning("Both fields are required.")

    # -------------------- DASHBOARD --------------------
    if st.session_state.logged_in:
        st.sidebar.success(f"Logged in as: {st.session_state.username}")
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.experimental_rerun()

        st.subheader("Step 1: Upload Datasets or Enter Manually")
        col1, col2 = st.columns(2)
        with col1:
            farmer_file = st.file_uploader("Upload Farmer Profiles CSV", type=["csv"])
        with col2:
            water_file = st.file_uploader("Upload Water Quality CSV", type=["csv"])

        if farmer_file is not None:
            st.session_state.farmer_df = pd.read_csv(farmer_file)
            st.success("Farmer data uploaded!")
        if water_file is not None:
            st.session_state.water_df = pd.read_csv(water_file)
            st.success("Water data uploaded!")

        # Manual entry fallback
        if st.session_state.farmer_df is None:
            st.markdown("### Or Enter Farmer Data Manually")
            credit_score = st.slider("Credit Score", 300, 900, 650)
            loan_amount = st.number_input("Loan Amount", 1000, 100000)
            loan_tenure = st.slider("Loan Tenure (months)", 6, 60, 12)
            past_defaults = st.number_input("Past Defaults", 0, 10)
            st.session_state.farmer_df = pd.DataFrame([{
                'credit_score': credit_score,
                'loan_amount': loan_amount,
                'loan_tenure': loan_tenure,
                'past_defaults': past_defaults,
                'default': 0  # Placeholder
            }])

        if st.session_state.water_df is None:
            st.markdown("### Or Enter Water Quality Data Manually")
            temp = st.slider("Temperature (°C)", 20.0, 35.0, 27.5)
            ph = st.slider("pH Level", 5.0, 9.0, 7.0)
            ammonia = st.slider("Ammonia (ppm)", 0.0, 5.0, 1.0)
            do = st.slider("Dissolved Oxygen (mg/L)", 2.0, 10.0, 6.5)
            st.session_state.water_df = pd.DataFrame([{
                'temperature': temp,
                'ph': ph,
                'ammonia': ammonia,
                'dissolved_oxygen': do,
                'failure': 0  # Placeholder
            }])

        # Train models
        try:
            loan_model, farm_model = train_models(st.session_state.farmer_df, st.session_state.water_df)
        except Exception as e:
            st.error("Please ensure both datasets have required columns.")
            st.stop()

        st.subheader("Step 2: Run Risk Predictions")

        # Loan Default Prediction
        loan_pred = loan_model.predict(st.session_state.farmer_df[['credit_score', 'loan_amount', 'loan_tenure', 'past_defaults']])[0]
        if loan_pred == 1:
            st.error("🔴 High Risk of Loan Default")
        else:
            st.success("🟢 Low Risk of Loan Default")

        # Fish Farm Failure Prediction
        farm_pred = farm_model.predict(st.session_state.water_df[['temperature', 'ph', 'ammonia', 'dissolved_oxygen']])[0]
        if farm_pred == 1:
            st.error("🔴 Farm Failure Risk Detected")
        else:
            st.success("🟢 Water Conditions Safe for Farming")

        st.markdown("---")
        st.markdown("✅ **Analysis Complete. Modify inputs or upload new files to refresh.**")

if __name__ == '__main__':
    main()
