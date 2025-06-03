import streamlit as st
import pandas as pd
import pickle
import os
from helper_functions import log_info, log_error  

# Paths
ARTIFACTS_PATH = os.path.join("Artifacts")
DATA_OUTPUT_PATH = os.path.join("data", "output")
LOGS_PATH = os.path.join("Logs")
os.makedirs(DATA_OUTPUT_PATH, exist_ok=True)

MODEL_PATH = os.path.join(ARTIFACTS_PATH, "best_classifier.pkl")
PIPELINE_PATH = os.path.join(ARTIFACTS_PATH, "data_processing_pipeline.pkl")
LABEL_ENCODER_PATH = os.path.join(ARTIFACTS_PATH, "label_encoder.pkl")

def load_artifact(filepath):
    try:
        with open(filepath, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        log_error(f"Artifact not found: {filepath}")
        st.error(f"Error: Artifact not found: {filepath}")
        return None

def predict_cancer(input_data):
    pipeline = load_artifact(PIPELINE_PATH)
    model = load_artifact(MODEL_PATH)
    label_encoder = load_artifact(LABEL_ENCODER_PATH)

    if not pipeline or not model or not label_encoder:
        return None
    
    input_df = pd.DataFrame([input_data], columns=input_data.keys())
    transformed_input = pipeline.transform(input_df)
    prediction = model.predict(transformed_input)
    return label_encoder.inverse_transform(prediction)[0]

# Mapping for user-friendly messages
label_to_message = {
    0: "No Cancer Detected",
    1: "Cancer Detected",

}

# Streamlit UI
st.title("🧬 Cancer Prediction App")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Single Prediction", "Batch Prediction"])

if page == "Single Prediction":
    st.header("Enter Patient Details")

    age = st.number_input("Age", min_value=20, max_value=80, value=45)
    gender = st.radio("Gender", ["Male", "Female"])
    gender_encoded = 0 if gender == "Male" else 1

    bmi = st.slider("BMI", min_value=15.0, max_value=40.0, value=22.0)

    smoking = st.radio("Smoking", ["No", "Yes"])
    smoking_encoded = 0 if smoking == "No" else 1

    genetic_risk = st.selectbox("Genetic Risk", ["Low", "Medium", "High"])
    genetic_risk_encoded = {"Low": 0, "Medium": 1, "High": 2}[genetic_risk]

    physical_activity = st.slider("Physical Activity (hrs/week)", min_value=0.0, max_value=10.0, value=3.0)
    alcohol_intake = st.slider("Alcohol Intake (units/week)", min_value=0.0, max_value=5.0, value=1.0)

    cancer_history = st.radio("Cancer History", ["No", "Yes"])
    cancer_history_encoded = 0 if cancer_history == "No" else 1

    if st.button("Predict Cancer Diagnosis"):
        input_data = {
            'Age': age,
            'Gender': gender_encoded,
            'BMI': bmi,
            'Smoking': smoking_encoded,
            'GeneticRisk': genetic_risk_encoded,
            'PhysicalActivity': physical_activity,
            'AlcoholIntake': alcohol_intake,
            'CancerHistory': cancer_history_encoded
        }

        prediction = predict_cancer(input_data)
        if prediction is not None:
            readable_pred = label_to_message.get(prediction, prediction)
            st.success(f"Predicted Diagnosis: {readable_pred}")
            log_info(f"Predicted Diagnosis: {readable_pred}")

elif page == "Batch Prediction":
    st.header("Batch Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file with columns: Age, Gender, BMI, Smoking, GeneticRisk, PhysicalActivity, AlcoholIntake, CancerHistory", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            pipeline = load_artifact(PIPELINE_PATH)
            model = load_artifact(MODEL_PATH)
            label_encoder = load_artifact(LABEL_ENCODER_PATH)

            if pipeline and model and label_encoder:
                transformed_data = pipeline.transform(df)
                predictions = model.predict(transformed_data)
                decoded_preds = label_encoder.inverse_transform(predictions)
                
                # Map predictions to user-friendly messages
                df['Predicted Diagnosis'] = [label_to_message.get(pred, pred) for pred in decoded_preds]

                output_file = os.path.join(DATA_OUTPUT_PATH, "batch_predictions.csv")
                df.to_csv(output_file, index=False)

                st.write(df)
                st.success(f"Batch prediction completed. Results saved at {output_file}")
                log_info("Batch Prediction Completed Successfully!")

        except Exception as e:
            log_error(f"Batch prediction failed: {str(e)}")
            st.error("Batch prediction failed. Please check your file format.")
