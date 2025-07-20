import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sqlite3
from io import StringIO

# ----------------------- PAGE CONFIG -----------------------
st.set_page_config(page_title="Aqua Risk Intelligence", layout="centered")
st.markdown("<h1 style='text-align:center;'>Aqua Risk Intelligence System 🌊</h1>", unsafe_allow_html=True)

if st.session_state.get("page_refresh"):
    st.session_state.page_refresh = False
    st.experimental_rerun()

# ----------------------- DB Setup -----------------------
conn = sqlite3.connect("aqua_users.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT NOT NULL
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    input_type TEXT,
    result TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')
conn.commit()

# ----------------------- Auth Functions -----------------------
def register_user(username, password):
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def login_user(username, password):
    cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    return cursor.fetchone() is not None

# ----------------------- Data Simulation -----------------------
@st.cache_data
def load_default_data():
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

# ----------------------- File Upload / Fallback -----------------------
uploaded_farmer = st.sidebar.file_uploader("📤 Upload Farmer Profile CSV", type=['csv'])
uploaded_water = st.sidebar.file_uploader("📤 Upload Water Quality CSV", type=['csv'])

if uploaded_farmer and uploaded_water:
    farmer_df = pd.read_csv(uploaded_farmer)
    water_df = pd.read_csv(uploaded_water)
    st.sidebar.success("✅ Uploaded data will be used.")
else:
    farmer_df, water_df = load_default_data()
    st.sidebar.info("⚠️ Using default demo data.")

# ----------------------- Train Models -----------------------
@st.cache_resource
def train_models(farmer_df, water_df):
    X1 = farmer_df.drop("loan_default", axis=1)
    y1 = farmer_df["loan_default"]
    loan_model = RandomForestClassifier().fit(X1, y1)

    X2 = water_df.drop("farm_failure", axis=1)
    y2 = water_df["farm_failure"]
    farm_model = RandomForestClassifier().fit(X2, y2)

    return loan_model, farm_model

loan_model, farm_model = train_models(farmer_df, water_df)

# ----------------------- Auth UI -----------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    auth_option = st.selectbox("Choose Action", ["Login", "Register"])
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if auth_option == "Register":
        if st.button("Register"):
            if register_user(user, pwd):
                st.success("Registered successfully. Please log in.")
            else:
                st.error("Username already exists.")
    else:
        if st.button("Login"):
            if login_user(user, pwd):
                st.success("Login successful! Redirecting...")
                st.session_state.authenticated = True
                st.session_state.username = user
                st.session_state.page_refresh = True
                st.stop()
            else:
                st.error("Invalid credentials.")
    st.stop()

# ----------------------- Tabs -----------------------
tab1, tab2, tab3 = st.tabs(["📊 Risk Prediction", "📈 Insights", "🧾 History"])

# ----------------------- Tab 1: Prediction -----------------------
with tab1:
    st.subheader("📊 Risk Prediction")
    st.markdown("Use the form to test Aqua Loan or Farm Failure Risk.")

    input_type = st.radio("Choose Model", ["AquaLoan Default", "Fish Farm Failure"])

    if input_type == "AquaLoan Default":
        income = st.number_input("Annual Income", 100000, 1000000, step=5000)
        loan = st.number_input("Loan Amount", 10000, 500000, step=5000)
        credit = st.slider("Credit Score", 300, 850)
        defaults = st.slider("Past Defaults", 0, 5)

        if st.button("Predict Loan Default"):
            result = loan_model.predict([[income, loan, credit, defaults]])[0]
            msg = "High Risk of Default ❌" if result == 1 else "Low Risk ✅"
            st.info(f"Prediction: {msg}")
            cursor.execute("INSERT INTO predictions (username, input_type, result) VALUES (?, ?, ?)",
                           (st.session_state.username, input_type, msg))
            conn.commit()

    else:
        temp = st.slider("Water Temperature (°C)", 20.0, 35.0, 28.0)
        ph = st.slider("pH Level", 5.0, 9.0, 7.5)
        ammonia = st.slider("Ammonia Level (mg/L)", 0.0, 1.0, 0.3)
        oxygen = st.slider("Dissolved Oxygen (mg/L)", 2.0, 10.0, 6.0)
        turbidity = st.slider("Turbidity (NTU)", 0.0, 50.0, 15.0)

        if st.button("Predict Farm Failure"):
            result = farm_model.predict([[temp, ph, ammonia, oxygen, turbidity]])[0]
            msg = "High Risk of Failure ❌" if result == 1 else "Stable Water Conditions ✅"
            st.info(f"Prediction: {msg}")
            cursor.execute("INSERT INTO predictions (username, input_type, result) VALUES (?, ?, ?)",
                           (st.session_state.username, input_type, msg))
            conn.commit()

# ----------------------- Tab 2: Insights -----------------------
with tab2:
    st.subheader("📈 Insights from Uploaded or Simulated Data")

    with st.expander("AquaLoan Data Summary"):
        st.dataframe(farmer_df.describe())

    with st.expander("Fish Farm Data Summary"):
        st.dataframe(water_df.describe())

    st.bar_chart(farmer_df['loan_default'].value_counts(), use_container_width=True)
    st.bar_chart(water_df['farm_failure'].value_counts(), use_container_width=True)

# ----------------------- Tab 3: History -----------------------
with tab3:
    st.subheader("🧾 Your Prediction History")

    cursor.execute("SELECT input_type, result, timestamp FROM predictions WHERE username=? ORDER BY timestamp DESC LIMIT 10",
                   (st.session_state.username,))
    rows = cursor.fetchall()

    if rows:
        hist_df = pd.DataFrame(rows, columns=["Type", "Result", "Time"])
        st.dataframe(hist_df)
    else:
        st.info("No predictions yet.")
