import streamlit as st
import pandas as pd
import joblib
import traceback

# ===================== Load Models & Feature Names =====================
try:
    log_model = joblib.load("logistic_regression.pkl")
    dt_model = joblib.load("decision_tree.pkl")
    rf_model = joblib.load("random_forest.pkl")
    xgb_model = joblib.load("xgboost.pkl")
    feature_names = joblib.load("feature_names.pkl")  # Load training feature order
except Exception as e:
    st.error("⚠️ Could not load models or feature names. Please check .pkl files.")
    st.code(traceback.format_exc())
    st.stop()

models = {
    "Logistic Regression": log_model,
    "Decision Tree": dt_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model,
}

# ===================== Streamlit App =====================
st.set_page_config(page_title="🚬 Smoking Status Prediction", layout="centered")
st.title("🚬 Smoking Status Prediction App")
st.write("This app predicts whether a person is a **Smoker** or **Non-Smoker** using trained ML models.")

# ===================== Feature Inputs =====================
st.subheader("Enter Patient Details")

user_input = {}
for col in feature_names:
    val = st.number_input(f"{col}", min_value=0.0, max_value=500.0, value=1.0)
    user_input[col] = val

# Convert to DataFrame and reorder columns
input_df = pd.DataFrame([user_input])[feature_names]

with st.expander("🔍 Show Input Data Sent to Model"):
    st.write(input_df)

# ===================== Model Selection =====================
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox("Choose a model", list(models.keys()))

# ===================== Prediction =====================
if st.button("Predict Smoking Status"):
    try:
        model = models[model_choice]
        prediction = model.predict(input_df)[0]
        result = "Smoker" if prediction == 1 else "Non-Smoker"
        st.success(f"✅ Prediction using {model_choice}: **{result}**")
    except Exception as e:
        st.error("⚠️ An error occurred during prediction.")
        st.code(traceback.format_exc())
