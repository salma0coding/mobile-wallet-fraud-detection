
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Mobile Wallet Fraud Detection",
    page_icon=None,
    layout="wide"
)

saved_model = joblib.load("fraud_detection_model.joblib")

pipeline = saved_model["pipeline"]
threshold = saved_model["threshold"]
features = saved_model["features"]

st.markdown("""
<style>
    .stApp {
        background-color: #F5F7FA;
    }

    h1 {
        color: #17365D;
    }

    h2, h3 {
        color: #24476B;
    }

    div.stButton > button {
        background-color: #17365D;
        color: white;
        border-radius: 8px;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #24476B;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("Mobile Wallet Fraud Detection")

st.write(
    "Detect potentially fraudulent mobile wallet transactions "
    "using Machine Learning."
)

st.sidebar.title("About the Project")

st.sidebar.write(
    "This application uses a Random Forest model "
    "to identify potentially fraudulent mobile wallet transactions."
)

st.sidebar.divider()

st.sidebar.write("Model: Random Forest")
st.sidebar.write("Dataset: PaySim")
st.sidebar.write(f"Decision Threshold: {threshold:.0%}")

st.divider()

st.header("Single Transaction Prediction")

col1, col2 = st.columns(2)

with col1:
    step = st.number_input(
        "Transaction Step",
        min_value=1,
        value=100,
        step=1
    )

with col2:
    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=1000.0,
        step=100.0
    )

transaction_type = st.selectbox(
    "Transaction Type",
    ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
)

if st.button("Predict Transaction"):

    input_data = pd.DataFrame({
        "step": [step],
        "amount": [amount],
        "type": [transaction_type]
    })

    input_data = input_data[features]

    probability = pipeline.predict_proba(input_data)[0][1]

    st.metric(
        "Fraud Probability",
        f"{probability:.2%}"
    )

    if probability >= threshold:
        st.error("Potential Fraud Detected!")
    else:
        st.success("Transaction Appears Legitimate")

st.divider()

st.header("Batch Transaction Prediction")

st.write(
    "Upload a CSV file containing transaction data "
    "to predict multiple transactions."
)

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

        required_columns = ["step", "amount", "type"]

        if data.empty:
            st.warning(
                "The uploaded CSV file contains no transactions."
            )

        elif not all(
            column in data.columns
            for column in required_columns
        ):
            st.error(
                "CSV must contain these columns: step, amount, type"
            )

        else:
            data = data.copy()

            data["step"] = pd.to_numeric(
                data["step"],
                errors="coerce"
            )

            data["amount"] = pd.to_numeric(
                data["amount"],
                errors="coerce"
            )

            if data[required_columns].isnull().any().any():
                st.error(
                    "CSV contains missing or invalid values. "
                    "Please check your data."
                )

            elif (
                (data["step"] < 1).any()
                or (data["amount"] < 0).any()
            ):
                st.error(
                    "Step must be positive and Amount cannot be negative."
                )

            elif not data["type"].isin(
                ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
            ).all():
                st.error(
                    "CSV contains an unsupported transaction type."
                )

            else:
                input_data = data[features].copy()

                probabilities = pipeline.predict_proba(
                    input_data
                )[:, 1]

                data["Fraud Probability"] = probabilities

                data["Prediction"] = [
                    "Potential Fraud"
                    if probability >= threshold
                    else "Legitimate"
                    for probability in probabilities
                ]

                fraud_count = (
                    data["Prediction"] == "Potential Fraud"
                ).sum()

                legitimate_count = (
                    data["Prediction"] == "Legitimate"
                ).sum()

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Total Transactions",
                    len(data)
                )

                col2.metric(
                    "Potential Fraud",
                    fraud_count
                )

                col3.metric(
                    "Legitimate",
                    legitimate_count
                )

                st.subheader("Prediction Results")

                st.dataframe(
                    data,
                    use_container_width=True
                )

                csv = data.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Download Results",
                    data=csv,
                    file_name="fraud_predictions.csv",
                    mime="text/csv"
                )

    except Exception as error:
        st.error(
            f"Unable to process this file: {error}"
        )

st.divider()

st.caption(
    "Note: Predictions are risk indicators and do not confirm fraud. "
    "Human review is recommended before taking action."
)
