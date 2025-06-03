import os
import pickle
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from helper_functions import log_info, log_error

# Define paths (adjust as needed)
ARTIFACTS_PATH = r"Artifacts"
os.makedirs(ARTIFACTS_PATH, exist_ok=True)
MODEL_PATH = os.path.join(ARTIFACTS_PATH, "best_classifier.pkl")
LABEL_ENCODER_PATH = os.path.join(ARTIFACTS_PATH, "label_encoder.pkl")

def training_pipeline(X_train, y_train):
    """
    Trains an XGBoost classifier and saves the model.
    """
    try:
        model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        model.fit(X_train, y_train)
        
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        
        log_info(f"Model trained and saved at {MODEL_PATH}")
        return model
    except Exception as e:
        log_error(f"Error during model training: {e}")
        raise

def load_model():
    """
    Loads the trained model from file.
    """
    try:
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        log_info("Model loaded successfully.")
        return model
    except FileNotFoundError:
        log_error(f"Model file not found at {MODEL_PATH}")
        raise

def prediction_pipeline(X_val):
    """
    Makes predictions using the trained model and decodes labels.
    """
    try:
        model = load_model()
        with open(LABEL_ENCODER_PATH, 'rb') as file:
            label_encoder = pickle.load(file)
        
        preds = model.predict(X_val)
        predictions = label_encoder.inverse_transform(preds)
        
        return predictions
    except FileNotFoundError as e:
        log_error(f"Error loading model or label encoder: {e}")
        raise

def evaluation_matrices(X_val, y_val):
    """
    Evaluates the model using confusion matrix, accuracy, and classification report.
    """
    try:
        pred_vals = prediction_pipeline(X_val)
        
        with open(LABEL_ENCODER_PATH, 'rb') as file:
            label_encoder = pickle.load(file)
        decoded_y_vals = label_encoder.inverse_transform(y_val)
        
        conf_matrix = confusion_matrix(decoded_y_vals, pred_vals, labels=label_encoder.classes_)
        acc_score = accuracy_score(decoded_y_vals, pred_vals)
        class_report = classification_report(decoded_y_vals, pred_vals)
        
        return conf_matrix, acc_score, class_report
    except FileNotFoundError:
        log_error("Label encoder file not found.")
        raise
