import os
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from helper_functions import log_info, log_error

# Paths
ARTIFACTS_PATH = r"Artifacts"
os.makedirs(ARTIFACTS_PATH, exist_ok=True)
PIPELINE_PATH = os.path.join(ARTIFACTS_PATH, "data_processing_pipeline.pkl")
LABEL_ENCODER_PATH = os.path.join(ARTIFACTS_PATH, "label_encoder.pkl")

def create_data_pipeline(X):
    """
    Creates a data processing pipeline.
    For now, it only scales numerical features using StandardScaler.
    You can customize this with more steps (e.g., imputers, encoders).
    """
    try:
        pipeline = Pipeline([
            ('scaler', StandardScaler())
        ])
        log_info("Data pipeline created.")
        return pipeline
    except Exception as e:
        log_error(f"Error creating data pipeline: {e}")
        raise

def save_pipeline(pipeline):
    """
    Saves the pipeline to disk.
    """
    try:
        with open(PIPELINE_PATH, 'wb') as f:
            pickle.dump(pipeline, f)
        log_info(f"Data pipeline saved at {PIPELINE_PATH}")
    except Exception as e:
        log_error(f"Error saving pipeline: {e}")
        raise

def encode_response_variable(y):
    """
    Encodes target variable labels and saves the LabelEncoder.
    """
    try:
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        with open(LABEL_ENCODER_PATH, 'wb') as f:
            pickle.dump(label_encoder, f)
        log_info(f"Label encoder saved at {LABEL_ENCODER_PATH}")
        return y_encoded
    except Exception as e:
        log_error(f"Error encoding target variable: {e}")
        raise

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits the data into train and validation sets.
    """
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        log_info(f"Data split into train and validation sets with test size={test_size}")
        return X_train, X_val, y_train, y_val
    except Exception as e:
        log_error(f"Error splitting data: {e}")
        raise
