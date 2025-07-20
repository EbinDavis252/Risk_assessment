import streamlit as st
import pandas as pd
import sqlite3
import os
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import hashlib

# ------------------ DB SETUP ------------------
conn = sqlite3.connect('aqua_users.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)''')
conn.commit()

# ------------------ SECURITY ------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed):
    return hash_password(password) == hashed

# ------------------ AUTH ------------------
def login():
    st.title("Login to Access Dashboard")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    
    if st.button("Login"):
        c.execute("SELECT password FROM users WHERE username=?", (username,))
        result = c.fetchone()
        if result and verify_password(password, result[0]):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.experimental_rerun()
        else:
            st.error("Invalid Username or Password")


def register():
    st.title("Register")
    new_user = st.text_input("Choose Username")
    new_pass = st.text_input("Choose Password", type='password')
    if st.button("Register"):
        c.execute("SELECT * FROM users WHERE username=?", (new_user,))
        if c.fetchone():
            st.error("Username already exists")
        else:
            hashed_pass = hash_password(new_pass)
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (new_user, hashed_pass))
            conn.commit()
            st.success("Registered successfully! Please login.")

# ------------------ MODEL TRAINING ------------------
def train_models(farmer_df, water_df):
    if 'LoanDefault' not in farmer_df.columns:
        st.error("Uploaded Farmer Profiles CSV must contain column: 'LoanDefault'")
        st.stop()
    if 'Failure' not in water_df.columns:
        st.error("Uploaded Water Quality CSV must contain column: 'Failure'")
        st.stop()

    X1 = farmer_df.drop('LoanDefault', axis=1)
    y1 = farmer_df['LoanDefault']
    loan_model = RandomForestClassifier().fit(X1, y1)

    X2 = water_df.drop('Failure', axis=1)
    y2 = water_df['Failure']
    farm_model = RandomForestClassifier().fit(X2, y2)

    return loan_model, farm_model

# ------------------ MAIN APP ------------------
def main():
    st.set_page_config(layout="wide")
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    menu = ["Login", "Register"]
    choice = st.sidebar.selectbox("Menu", menu)

    if not st.session_state.logged_in:
        if choice == "Login":
            login()
        else:
            register()
        return

    st.sidebar.success(f"Welcome {st.session_state.username}")
    st.title("Aqua Risk Assessment Dashboard")

    farmer_file = st.file_uploader("Upload Farmer Profiles CSV", type=["csv"])
    water_file = st.file_uploader("Upload Water Quality CSV", type=["csv"])

    if farmer_file is not None and water_file is not None:
        farmer_df = pd.read_csv(farmer_file)
        water_df = pd.read_csv(water_file)

        if 'LoanDefault' not in farmer_df.columns or 'Failure' not in water_df.columns:
            st.error("Ensure Farmer CSV has 'LoanDefault' and Water CSV has 'Failure' column")
            return

        st.success("✅ Data uploaded successfully")

        loan_model, farm_model = train_models(farmer_df, water_df)

        st.subheader("Loan Default Prediction")
        with st.form("Loan Prediction"):
            age = st.number_input("Age", 18, 70)
            exp = st.number_input("Experience (years)", 0, 50)
            amount = st.number_input("Loan Amount", 5000, 1000000)
            credit = st.slider("Credit Score", 300, 900)
            submit1 = st.form_submit_button("Predict Loan Risk")
            if submit1:
                result = loan_model.predict([[age, exp, amount, credit]])[0]
                st.info("High Risk of Default" if result else "Low Risk")

        st.subheader("Fish Farm Failure Prediction")
        with st.form("Farm Prediction"):
            temp = st.number_input("Water Temperature", 10.0, 40.0)
            ph = st.slider("pH Level", 5.0, 9.0)
            ammonia = st.number_input("Ammonia Level", 0.0, 10.0)
            do = st.number_input("Dissolved Oxygen (DO)", 0.0, 15.0)
            submit2 = st.form_submit_button("Predict Failure Risk")
            if submit2:
                result = farm_model.predict([[temp, ph, ammonia, do]])[0]
                st.info("High Risk of Farm Failure" if result else "Stable Conditions")

        st.subheader("📊 Raw Data Preview")
        st.write("Farmer Profiles Data")
        st.dataframe(farmer_df.head())
        st.write("Water Quality Data")
        st.dataframe(water_df.head())

if __name__ == '__main__':
    main()
