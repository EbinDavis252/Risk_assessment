import streamlit as st
import pandas as pd
import sqlite3
import hashlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Aqua Finance Risk System", layout="wide")
if st.session_state.get("page_refresh"):
    st.session_state.page_refresh = False
    st.experimental_rerun()

# ------------------ AUTH SETUP -------------------
conn = sqlite3.connect('users.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users(username TEXT, password TEXT)''')
conn.commit()

def hash_pwd(pwd): return hashlib.sha256(pwd.encode()).hexdigest()
def verify_pwd(pwd, hash): return hash_pwd(pwd) == hash
def register_user(user, pwd): c.execute("INSERT INTO users VALUES (?,?)", (user, hash_pwd(pwd))); conn.commit()
def login_user(user, pwd):
    c.execute("SELECT * FROM users WHERE username=?", (user,))
    data = c.fetchone()
    if data and verify_pwd(pwd, data[1]): return True
    return False

# ------------------ SIDEBAR LOGIN ------------------
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    with st.container():
        st.title("🔐 Login or Register")
        option = st.radio("Select:", ["Login", "Register"])
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")

        if option == "Register":
            if st.button("Register"):
                register_user(user, pwd)
                st.success("Registered! Now login.")
        elif option == "Login":
            if st.button("Login"):
                if login_user(user, pwd):
                    st.success("Login successful. Redirecting...")
                    st.session_state.authenticated = True
                    st.session_state.page_refresh = True
                    st.stop()
                else:
                    st.error("Invalid username or password.")
    st.stop()

# ------------------ DATA UPLOAD ------------------
st.title("🌊 Aqua Finance Risk Assessment System")
st.markdown("Upload your datasets or manually enter sample data.")

uploaded_farmer = st.file_uploader("Upload Farmer Profile CSV", type=["csv"])
uploaded_water = st.file_uploader("Upload Water Quality CSV", type=["csv"])

if uploaded_farmer:
    farmer_df = pd.read_csv(uploaded_farmer)
else:
    st.info("Using default sample farmer data")
    farmer_df = pd.DataFrame({
        'annual_income': [400000, 250000, 500000],
        'loan_amount': [200000, 150000, 300000],
        'credit_score': [700, 620, 680],
        'past_defaults': [0, 1, 0],
        'loan_default': [0, 1, 0]
    })

if uploaded_water:
    water_df = pd.read_csv(uploaded_water)
else:
    st.info("Using default sample water quality data")
    water_df = pd.DataFrame({
        'temperature': [28.5, 31.0, 29.3],
        'pH': [7.5, 6.8, 8.0],
        'ammonia': [0.02, 0.05, 0.01],
        'dissolved_oxygen': [6.5, 4.2, 7.0],
        'turbidity': [3.0, 5.2, 2.5],
        'farm_failure': [0, 1, 0]
    })

# ------------------ TRAIN MODELS ------------------
@st.cache_resource
def train_models(farmer_df, water_df):
    # Clean farmer data
    farmer_df = farmer_df.dropna()
    farmer_df = farmer_df.astype({
        'annual_income': 'float64',
        'loan_amount': 'float64',
        'credit_score': 'float64',
        'past_defaults': 'int64',
        'loan_default': 'int64'
    })
    X1 = farmer_df.drop("loan_default", axis=1)
    y1 = farmer_df["loan_default"]
    loan_model = RandomForestClassifier().fit(X1, y1)

    # Clean water data
    water_df = water_df.dropna()
    water_df = water_df.astype({
        'temperature': 'float64',
        'pH': 'float64',
        'ammonia': 'float64',
        'dissolved_oxygen': 'float64',
        'turbidity': 'float64',
        'farm_failure': 'int64'
    })
    X2 = water_df.drop("farm_failure", axis=1)
    y2 = water_df["farm_failure"]
    farm_model = RandomForestClassifier().fit(X2, y2)

    return loan_model, farm_model

loan_model, farm_model = train_models(farmer_df, water_df)

# ------------------ PREDICTION UI ------------------
st.header("🔍 Make Predictions")

st.subheader("Loan Default Prediction")
col1, col2, col3 = st.columns(3)
with col1:
    income = st.number_input("Annual Income", value=400000)
    loan_amt = st.number_input("Loan Amount", value=200000)
with col2:
    score = st.slider("Credit Score", 300, 900, value=700)
    defaults = st.selectbox("Past Defaults", [0, 1, 2], index=0)

if st.button("Predict Loan Default"):
    result = loan_model.predict([[income, loan_amt, score, defaults]])[0]
    st.success("Prediction: ❌ Loan Default" if result == 1 else "✅ No Default")

st.markdown("---")
st.subheader("Fish Farm Failure Prediction")
col4, col5, col6 = st.columns(3)
with col4:
    temp = st.number_input("Temperature", value=28.0)
    ph = st.number_input("pH Level", value=7.5)
with col5:
    ammonia = st.number_input("Ammonia (mg/L)", value=0.02)
    do = st.number_input("Dissolved Oxygen (mg/L)", value=6.5)
with col6:
    turb = st.number_input("Turbidity", value=3.0)

if st.button("Predict Farm Failure"):
    result = farm_model.predict([[temp, ph, ammonia, do, turb]])[0]
    st.success("Prediction: ⚠️ Farm Failure Risk" if result == 1 else "✅ No Failure Detected")

# ------------------ OPTIONAL ------------------
if st.button("Logout"):
    st.session_state.authenticated = False
    st.experimental_rerun()
