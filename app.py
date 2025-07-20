import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ------------------- AUTH SECTION -------------------
def create_usertable():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT, password TEXT)')
    conn.commit()
    conn.close()

def add_userdata(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('INSERT INTO userstable(username,password) VALUES (?,?)', (username, password))
    conn.commit()
    conn.close()

def login_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?', (username, password))
    data = c.fetchall()
    conn.close()
    return data

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ------------------ MODEL TRAINING ------------------
@st.cache_data
def train_models(farmer_df, water_df):
    # AquaLoan Default Prediction
    loan_features = ['credit_score', 'loan_amount', 'loan_tenure', 'past_defaults']
    loan_target = 'default'
    X1 = farmer_df[loan_features]
    y1 = farmer_df[loan_target]
    loan_model = RandomForestClassifier().fit(X1, y1)

    # Fish Farm Failure Prediction
    farm_features = ['temperature', 'ph', 'ammonia', 'dissolved_oxygen']
    farm_target = 'failure'
    X2 = water_df[farm_features]
    y2 = water_df[farm_target]
    farm_model = RandomForestClassifier().fit(X2, y2)

    return loan_model, farm_model

# ------------------- MAIN APP -------------------
def main():
    st.set_page_config(page_title="AquaRisk AI", layout="wide")
    st.title("💧 Aqua Finance Risk Assessment & Early Warning System")

    menu = ["Login", "Register"]
    choice = st.sidebar.selectbox("Access Panel", menu)

    create_usertable()

    if choice == "Register":
        st.subheader("Create a New Account")
        new_user = st.text_input("Username")
        new_password = st.text_input("Password", type='password')
        if st.button("Register"):
            if new_user and new_password:
                add_userdata(new_user, hash_password(new_password))
                st.success("Registration successful. Please login.")
            else:
                st.warning("Please fill both fields.")

    elif choice == "Login":
        st.subheader("Login to Access Dashboard")
        username = st.text_input("Username")
        password = st.text_input("Password", type='password')
        if st.button("Login"):
            if login_user(username, hash_password(password)):
                st.success(f"Welcome, {username} 👋")

                # ----- Upload datasets -----
                st.subheader("Upload Datasets")
                col1, col2 = st.columns(2)
                with col1:
                    farmer_file = st.file_uploader("Upload Farmer Profiles CSV", type=["csv"])
                with col2:
                    water_file = st.file_uploader("Upload Water Quality CSV", type=["csv"])

                if farmer_file is not None and water_file is not None:
                    farmer_df = pd.read_csv(farmer_file)
                    water_df = pd.read_csv(water_file)

                    st.success("✅ Files uploaded successfully.")
                    st.write("Farmer Data", farmer_df.head())
                    st.write("Water Data", water_df.head())

                    loan_model, farm_model = train_models(farmer_df, water_df)

                    # ----- Predictions -----
                    st.subheader("📊 Make Predictions")

                    st.markdown("### Predict Loan Default")
                    credit_score = st.slider("Credit Score", 300, 900, 650)
                    loan_amount = st.number_input("Loan Amount", 1000, 100000)
                    loan_tenure = st.slider("Tenure (months)", 6, 60, 12)
                    past_defaults = st.number_input("Past Defaults", 0, 10)
                    pred1 = loan_model.predict([[credit_score, loan_amount, loan_tenure, past_defaults]])[0]
                    st.success("🔴 Default Risk" if pred1 else "🟢 Low Risk")

                    st.markdown("---")
                    st.markdown("### Predict Fish Farm Failure")
                    temp = st.slider("Temperature (°C)", 20.0, 35.0, 27.5)
                    ph = st.slider("pH Level", 5.0, 9.0, 7.0)
                    ammonia = st.slider("Ammonia (ppm)", 0.0, 5.0, 1.0)
                    do = st.slider("Dissolved Oxygen (mg/L)", 2.0, 10.0, 6.5)
                    pred2 = farm_model.predict([[temp, ph, ammonia, do]])[0]
                    st.success("🔴 Farm at Risk" if pred2 else "🟢 Conditions Safe")

                else:
                    st.warning("Please upload both datasets to proceed.")

            else:
                st.error("Invalid Username or Password")

if __name__ == '__main__':
    main()
