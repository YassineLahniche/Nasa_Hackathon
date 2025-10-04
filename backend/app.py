# File: backend/app.py (with Mock Model and No File Dependencies)

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os

# --- Initialization ---
app = Flask(__name__)
CORS(app) # Allows our frontend to make requests to this backend

# --- Mock Model and Scaler ---
# These classes replace the need to load external files.

class MockModel:
    """A mock model that simulates predictions."""
    def predict(self, data_df):
        # Prediction logic is sensitive to planetary radius
        prad = data_df['koi_prad'].iloc[0]
        # Creates a smooth probability curve based on radius
        proba = 0.95 * np.exp(-0.1 * prad) + 0.05
        return np.array([proba])

class MockScaler:
    """A mock scaler that mimics the real one but does nothing."""
    def fit(self, data):
        # In a real scenario, this would learn the scaling parameters
        pass
    def transform(self, data):
        # In a real scenario, this would apply the scaling
        return data

# Instantiate our mock objects
model = MockModel()
scaler = MockScaler()
features = ['koi_period', 'koi_depth', 'koi_duration', 'koi_prad', 'koi_insol', 'koi_steff', 'koi_srad', 'koi_slogg']
print("✅ Backend running in mock mode. No model or data files needed.")


# --- API Endpoints ---

@app.route('/model_stats', methods=['GET'])
def get_model_stats():
    """Returns mock statistics about the model."""
    stats = {
        'accuracy': 97.8,
        'f1Score': 0.97,
        'precision': 0.98,
        'recall': 0.96,
        'totalCandidates': 9564
    }
    return jsonify(stats)

def get_prediction_details(data_df):
    """Helper function to get prediction from the mock model."""
    scaled_data = scaler.transform(data_df)
    proba = model.predict(scaled_data)[0]
    
    # Determine classification based on probability
    if proba > 0.8:
        pred_class = 'CONFIRMED'
    elif proba > 0.5:
        pred_class = 'CANDIDATE'
    else:
        pred_class = 'FALSE POSITIVE'
        
    return pred_class, proba

@app.route('/predict', methods=['POST'])
def predict_single():
    """Handles single predictions from Explorer and Researcher manual entry."""
    json_data = request.get_json()
    input_df = pd.DataFrame([json_data], columns=features)
    
    pred_class, proba = get_prediction_details(input_df)

    # Generate mock feature importance based on input values
    params = input_df.iloc[0]
    importance = [
        {'feature': 'Planetary Radius', 'value': params['koi_prad'] * 5},
        {'feature': 'Transit Depth', 'value': params['koi_depth'] / 100},
        {'feature': 'Stellar Temp', 'value': params['koi_steff'] / 100},
        {'feature': 'Orbital Period', 'value': 200 / params['koi_period']},
    ]

    return jsonify({
        'prediction': pred_class,
        'probability': proba,
        'featureImportance': sorted(importance, key=lambda x: x['value'], reverse=True)
    })

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Handles batch predictions from CSV uploads."""
    # This endpoint is harder to mock without real data,
    # so for now it returns a simple canned response.
    # You can still test the file upload UI.
    return jsonify([
        {'koi_prad': 2.5, 'koi_period': 10.1, 'koi_depth': 150.0, 'prediction': 'CONFIRMED', 'probability': 0.92},
        {'koi_prad': 30.0, 'koi_period': 5.2, 'koi_depth': 9000.0, 'prediction': 'FALSE POSITIVE', 'probability': 0.11},
    ])

if __name__ == '__main__':
    app.run(debug=True, port=5001)