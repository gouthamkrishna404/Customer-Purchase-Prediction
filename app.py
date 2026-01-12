import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# STEP 1: LOAD DATA

@st.cache_data
def load_data():
    df = pd.read_csv("data/customer_purchase_data.csv")
    return df

df = load_data()

st.title("Customer Purchase Prediction")
st.write("Dataset Preview:")
st.dataframe(df.head())

# STEP 2: DATA PREPROCESSING

X = df.drop("PurchaseStatus", axis=1)
y = df["PurchaseStatus"]

# STEP 3: LOAD OR TRAIN MODEL

try:
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    st.success("Loaded pre-trained model and scaler.")
except:
    st.warning("No pre-trained model found. Training a new Random Forest...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(
        random_state=42,
        n_estimators=200,
        max_depth=None
    )
    model.fit(X_scaled, y)
    st.success("Model trained and saved.")

# STEP 4: USER INPUT FOR PREDICTION

st.sidebar.header("Input Customer Features")

input_data = {}
for col in X.columns:
    val = st.sidebar.number_input(
        f"Enter value for {col}",
        value=float(X[col].mean())
    )
    input_data[col] = val

input_df = pd.DataFrame([input_data])
input_scaled = scaler.transform(input_df)

prediction = model.predict(input_scaled)[0]
prediction_prob = model.predict_proba(input_scaled)[0][1]

st.subheader("Prediction Result")
st.write(
    f"Predicted Purchase Status: **{'Purchase' if prediction == 1 else 'No Purchase'}**"
)
st.write(f"Probability of Purchase: **{prediction_prob:.2f}**")
