import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Database setup
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password TEXT NOT NULL
                )''')
    conn.commit()
    conn.close()

# Register page
def register():
    st.subheader("Register")
    new_user = st.text_input("Username", key="reg_user")
    new_pass = st.text_input("Password", type='password', key="reg_pass")
    if st.button("Register"):
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("INSERT OR IGNORE INTO users (username, password) VALUES (?, ?)", (new_user, new_pass))
        conn.commit()
        conn.close()
        st.success("Registered Successfully. Please login.")

# Login page
def login():
    st.subheader("Login to Access Dashboard")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        if c.fetchone():
            st.success("Login successful!")
            st.session_state.logged_in = True
        else:
            st.error("Invalid Username or Password")
        conn.close()

# Model training function
@st.cache_data
def train_models(farmer_df, water_df):
    # Loan Default Model
    if 'LoanDefault' not in farmer_df.columns:
        raise ValueError("Farmer data must contain 'LoanDefault' column")
    X1 = farmer_df.drop('LoanDefault', axis=1)
    y1 = farmer_df['LoanDefault']
    model1 = RandomForestClassifier()
    model1.fit(X1, y1)

    # Farm Failure Model
    if 'Failure' not in water_df.columns:
        raise ValueError("Water data must contain 'Failure' column")
    X2 = water_df.drop('Failure', axis=1)
    y2 = water_df['Failure']
    model2 = RandomForestClassifier()
    model2.fit(X2, y2)

    return model1, model2

# Dashboard UI
def show_dashboard():
    st.subheader("Upload Data or Enter Manually")

    # Uploading files
    farmer_file = st.file_uploader("Upload Farmer Profiles CSV", type=["csv"], key="farmer")
    water_file = st.file_uploader("Upload Water Quality CSV", type=["csv"], key="water")

    if farmer_file is not None and water_file is not None:
        farmer_df = pd.read_csv(farmer_file)
        water_df = pd.read_csv(water_file)
    else:
        st.write("Or Enter Data Manually")
        farmer_data = {
            'Age': st.number_input("Age", 18, 80),
            'LoanAmount': st.number_input("Loan Amount", 0),
            'CreditScore': st.slider("Credit Score", 300, 900),
            'LoanDefault': st.selectbox("Loan Default", [0, 1])
        }
        water_data = {
            'Temperature': st.slider("Temperature", 20.0, 35.0),
            'pH': st.slider("pH", 5.0, 9.0),
            'DissolvedOxygen': st.slider("Dissolved Oxygen", 3.0, 10.0),
            'Ammonia': st.slider("Ammonia", 0.0, 5.0),
            'Failure': st.selectbox("Farm Failure", [0, 1])
        }
        farmer_df = pd.DataFrame([farmer_data])
        water_df = pd.DataFrame([water_data])

    st.success("Data Loaded Successfully")
    st.write("Farmer Data", farmer_df.head())
    st.write("Water Data", water_df.head())

    # Train and predict
    try:
        loan_model, farm_model = train_models(farmer_df, water_df)

        # Predictions
        loan_preds = loan_model.predict(farmer_df.drop('LoanDefault', axis=1))
        farm_preds = farm_model.predict(water_df.drop('Failure', axis=1))

        st.subheader("Loan Default Prediction")
        st.write(pd.DataFrame({'Actual': farmer_df['LoanDefault'], 'Predicted': loan_preds}))
        st.text(f"Accuracy: {accuracy_score(farmer_df['LoanDefault'], loan_preds)*100:.2f}%")

        st.subheader("Farm Failure Prediction")
        st.write(pd.DataFrame({'Actual': water_df['Failure'], 'Predicted': farm_preds}))
        st.text(f"Accuracy: {accuracy_score(water_df['Failure'], farm_preds)*100:.2f}%")
    except Exception as e:
        st.error(f"Model Training Error: {e}")

# Main function
def main():
    st.set_page_config(page_title="Aqua Finance Risk App", layout="centered")
    st.title("Aqua Finance Risk Management Dashboard")

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    menu = ["Login", "Register"]
    choice = st.sidebar.selectbox("Menu", menu)

    if not st.session_state.logged_in:
        if choice == "Login":
            login()
        else:
            register()
    else:
        show_dashboard()

# Run the app
if __name__ == "__main__":
    init_db()
    main()
