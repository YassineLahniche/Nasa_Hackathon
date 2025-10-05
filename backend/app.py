# File: backend/app.py (Corrected and Final)

import os
import json  # Ensures the JSON library is imported
import threading
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import the main training function from your other script
try:
    from model_trainer import run_training_pipeline
except ImportError:
    print("WARNING: model_trainer.py not found. Retraining endpoint will not work.")
    def run_training_pipeline(hyperparameters=None):
        print("ERROR: Training script not available.")
        return {'error': 'Training script not found.'}

# --- CONFIGURATION ---
MODEL_PATH = 'models/lightgbm_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
FEATURES_PATH = 'models/features.json'
KEPLER_DATA_PATH = 'data/koi.csv'
STATS_PATH = 'models/model_stats.json'

# --- GLOBAL OBJECTS ---
app = Flask(__name__)
CORS(app)
model, scaler, features, kepler_df = None, None, None, None
model_lock = threading.Lock()

def load_artifacts():
    """Loads all necessary artifacts: model, scaler, features, stats, and data."""
    global model, scaler, features, kepler_df, model_stats # Added model_stats here
    print("Attempting to load all artifacts...")
    artifacts_loaded = True
    
    with model_lock:
        try:
            # 1. Load Model
            if os.path.exists(MODEL_PATH):
                model = joblib.load(MODEL_PATH)
                print(f"✅ Model loaded from {MODEL_PATH}")
            else:
                print(f"⚠️ Model not found at {MODEL_PATH}")
                artifacts_loaded = False
            
            # 2. Load Scaler
            if os.path.exists(SCALER_PATH):
                scaler = joblib.load(SCALER_PATH)
                print(f"✅ Scaler loaded from {SCALER_PATH}")
            else:
                print(f"⚠️ Scaler not found at {SCALER_PATH}")
                artifacts_loaded = False

            # 3. Load Features List
            if os.path.exists(FEATURES_PATH):
                with open(FEATURES_PATH, 'r') as f:
                    features = json.load(f)
                print(f"✅ Features list loaded from {FEATURES_PATH}")
            else:
                print(f"⚠️ Features list not found at {FEATURES_PATH}")
                artifacts_loaded = False
            
            # 4. NEW: Load Model Stats
            if os.path.exists(STATS_PATH):
                with open(STATS_PATH, 'r') as f:
                    model_stats = json.load(f)
                print(f"✅ Model stats loaded from {STATS_PATH}")
            else:
                print(f"⚠️ Model stats not found at {STATS_PATH}")
                # We can still run, but the stats endpoint will have no data
                model_stats = None # Explicitly set to None if not found

            # 5. Load Kepler Data for Random Candidate Feature
            if os.path.exists(KEPLER_DATA_PATH):
                kepler_df = pd.read_csv(KEPLER_DATA_PATH, comment='#')
                print(f"✅ Kepler data for random candidates loaded from {KEPLER_DATA_PATH}")
            else:
                print(f"⚠️ Kepler data not found at {KEPLER_DATA_PATH}")
                kepler_df = pd.DataFrame() # Prevents crashing if file is missing

        except Exception as e:
            print(f"❌ CRITICAL ERROR during artifact loading: {e}")
            return False

    if artifacts_loaded:
        print("✅ All artifacts loaded successfully.")
    else:
        print("🛑 Some artifacts are missing. Please run `python model_trainer.py` to generate them.")
    return artifacts_loaded

# --- API ENDPOINTS ---
# [The rest of your app.py code for /model_stats, /random_candidate, /predict, etc. follows here]
# For brevity, I will include the full, correct code for all endpoints below.

@app.route('/model_stats', methods=['GET'])
def get_model_stats():
    """Serves the real model statistics loaded from the JSON file."""
    if model_stats:
        return jsonify(model_stats)
    else:
        return jsonify({'error': 'Model statistics not available'}), 500

# File: backend/app.py (Updated /random_candidate endpoint)

@app.route('/random_candidate', methods=['GET'])
def get_random_candidate():
    """Selects a random candidate from the Kepler dataset pool with standardized names."""
    if kepler_df is None or kepler_df.empty or features is None:
        return jsonify({'error': 'Server data not fully loaded'}), 500
    
    # *** KEY CHANGE: Only map the features the model actually uses ***
    column_mapping = {
        'koi_period': 'orbital_period', 
        'koi_duration': 'transit_duration', 
        'koi_depth': 'transit_depth',
        'koi_prad': 'planet_radius', 
        'koi_insol': 'insolation_flux',
        'koi_srad': 'stellar_radius'
    }
    
    # Filter for columns that exist in the dataframe and are needed by the model
    valid_raw_cols = [raw_col for raw_col, std_col in column_mapping.items() if std_col in features and raw_col in kepler_df.columns]
    
    # Get a random row with no missing values in these key columns
    random_row_raw = kepler_df[valid_raw_cols].dropna().sample(1).iloc[0]
    
    # Create the response with standardized names
    random_candidate_std = {column_mapping[raw_col]: val for raw_col, val in random_row_raw.items()}
    
    return jsonify(random_candidate_std)

@app.route('/predict', methods=['POST'])
def predict_single():
    with model_lock:
        if not all([model, scaler, features]):
            return jsonify({'error': 'Backend not ready. Model artifacts are missing.'}), 500
    
    json_data = request.get_json()
    try:
        input_df = pd.DataFrame([json_data], columns=features)
        scaled_data = scaler.transform(input_df)
        with model_lock:
            probability = model.predict_proba(scaled_data)[0][1]
            feature_importances = model.feature_importances_
        
        prediction_class = 'CONFIRMED' if probability > 0.6 else ('CANDIDATE' if probability > 0.4 else 'FALSE POSITIVE')
        importance_data = [{'feature': feat, 'value': float(imp)} for feat, imp in zip(features, feature_importances)]

        return jsonify({'prediction': prediction_class, 'probability': float(probability), 'featureImportance': sorted(importance_data, key=lambda x: x['value'], reverse=True)})
    except Exception as e:
        return jsonify({'error': f"Prediction error: {str(e)}"}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    # Placeholder for batch prediction logic
    return jsonify({'message': 'Batch endpoint is ready.'})

@app.route('/retrain', methods=['POST'])
def retrain_model_endpoint():
    hyperparams = request.get_json() or {}
    print(f"Received retraining request with parameters: {hyperparams}")
    def training_task():
        print("--- Starting background retraining task ---")
        results = run_training_pipeline(hyperparams)
        if 'error' not in results:
            print("--- Retraining successful. Reloading artifacts... ---")
            load_artifacts()
        else:
            print(f"--- Retraining failed: {results['error']} ---")
    thread = threading.Thread(target=training_task)
    thread.start()
    return jsonify({'message': 'Model retraining started.'}), 202

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    load_artifacts()
    print("\n🚀 Exo-Explorer Backend is running!")
    print("   Ready to accept requests at http://127.0.0.1:5001")
    app.run(debug=True, port=5001)