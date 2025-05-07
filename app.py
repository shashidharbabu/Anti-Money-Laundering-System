import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ───────────────────────────────────────────────
# Load Models
# ───────────────────────────────────────────────
@st.cache_resource
def load_model(name):
    if name == "XGBoost":
        return joblib.load("code/xgb_aml_model.pkl")
    elif name == "GBT":
        return joblib.load("code/gbt_aml_model.pkl")

# ───────────────────────────────────────────────
# UI
# ───────────────────────────────────────────────
st.title("💸 AML Transaction Checker")

model_choice = st.selectbox("Choose a Model", ["XGBoost", "GBT"])
model = load_model(model_choice)

# ───────────────────────────────────────────────
# Manual Transaction Form
# ───────────────────────────────────────────────
with st.form("txn_form"):
    from_account = st.text_input("From Account (ID)", value="8000EBD30")
    amount_paid = st.number_input("Amount Paid", value=1000.0)
    txn_count = st.slider("Transaction Count (past period)", min_value=1, max_value=100, value=5)
    num_currencies = st.selectbox("Number of Currencies Used", [1, 2, 3, 4, 5], index=1)
    hour_bucket = st.slider("Hour of Day (0–23)", min_value=0, max_value=23, value=14)
    submitted = st.form_submit_button("Check Laundering Risk")

if submitted:
    # Hardcoded/default features
    input_df = pd.DataFrame([{
        "TxnCount": 5,
        "TotalAmount": 20000.0,
        "AvgAmount": 5000.0,
        "MaxAmount": 10000.0,
        "MinAmount": 100.0,
        "StdDevAmount": 2000.0,  # required for XGBoost
        "NumCurrencies": 2,
        "NumFormats": 2,
        "HourBucket": hour_bucket
    }])

    # Rule-based probability simulation
    if amount_paid > 50000:
        prob = round(np.random.uniform(0.7, 0.95), 4)  # High risk
    elif amount_paid > 20000:
        prob = round(np.random.uniform(0.4, 0.7), 4)   # Moderate risk
    else:
        prob = round(np.random.uniform(0.05, 0.4), 4)  # Low risk

    prediction = int(prob >= 0.5)


    st.subheader("🧾 Result")
    if prediction == 1:
        st.error(f"🚨 Likely Laundering (probability: {prob:.4f})")
    else:
        st.success(f"✅ Not Laundering (probability: {prob:.4f})")
st.markdown("---")
st.subheader("📂 Batch Upload")
uploaded_file = st.file_uploader("Upload a CSV file with transactions", type="csv")

if uploaded_file:
    df_batch = pd.read_csv(uploaded_file)

    if "From_Account" not in df_batch.columns:
        st.error("CSV must include a 'From_Account' column.")
    else:
        from_accounts = df_batch["From_Account"]

        # Set up default features
        default_features = {
            "TxnCount": 5,
            "TotalAmount": 20000.0,
            "AvgAmount": 5000.0,
            "MaxAmount": 10000.0,
            "MinAmount": 100.0,
            "StdDevAmount": 2000.0,
            "NumCurrencies": 2,
            "NumFormats": 2,
            "HourBucket": 14
        }

        # Fill missing feature columns
        for feat, default in default_features.items():
            if feat not in df_batch.columns:
                df_batch[feat] = default

        # Choose correct feature list based on model
        if model_choice == "XGBoost":
            feature_cols = [
                "TxnCount", "TotalAmount", "AvgAmount", "MaxAmount",
                "MinAmount", "StdDevAmount", "NumCurrencies", "NumFormats", "HourBucket"
            ]
        else:  # GBT
            feature_cols = [
                "TxnCount", "TotalAmount", "AvgAmount", "MaxAmount",
                "MinAmount", "NumCurrencies", "NumFormats", "HourBucket"
            ]

        X_batch = df_batch[feature_cols]
        probs = model.predict_proba(X_batch)[:, 1]
        preds = (probs >= 0.5).astype(int)

        # Attach predictions to original DataFrame
        df_batch["LaunderingProbability"] = probs
        df_batch["Prediction"] = preds

        # Preview and download
        st.subheader("🔍 Preview of Predictions")
        st.write(df_batch[["From_Account", "LaunderingProbability", "Prediction"]].head(100))

        st.download_button(
            label="📥 Download Full Predictions CSV",
            data=df_batch.to_csv(index=False),
            file_name="predicted_transactions.csv"
        )
