import os
import json
import threading
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from datetime import datetime

# Import training utilities so runtime API and training share the same helpers.
try:
    from model_trainer import run_training_pipeline, clean_dataset, standardize_columns
except ImportError:
    print("WARNING: model_trainer.py not found. Retraining and random candidate features will not work correctly.")

    # Fallback no-op functions allow the server to start even if training code is missing.
    def run_training_pipeline(hyperparameters=None): return {'error': 'Training script not found.'}
    def clean_dataset(data, name, col): return data
    def standardize_columns(df, name): return df

MODEL_PATH = 'models/lightgbm_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
FEATURES_PATH = 'models/features.json'
STATS_PATH = 'models/model_stats.json'
KEPLER_DATA_PATH = 'data/koi.csv'
USER_DATA_PATH = 'user_data'

app = Flask(__name__)
CORS(app)
model, scaler, features, model_stats = None, None, None, None

kepler_df_cleaned_std = None
model_lock = threading.Lock()

def load_artifacts():
    """Loads all necessary artifacts: model, scaler, features, stats, and data."""
    global model, scaler, features, model_stats, kepler_df_cleaned_std
    print("Attempting to load all artifacts...")
    artifacts_loaded = True

    with model_lock:
        try:

            if os.path.exists(MODEL_PATH): model = joblib.load(MODEL_PATH)
            else: artifacts_loaded = False; print(f"⚠️ Model not found at {MODEL_PATH}")

            if os.path.exists(SCALER_PATH): scaler = joblib.load(SCALER_PATH)
            else: artifacts_loaded = False; print(f"⚠️ Scaler not found at {SCALER_PATH}")

            if os.path.exists(FEATURES_PATH):
                with open(FEATURES_PATH, 'r') as f: features = json.load(f)
            else: artifacts_loaded = False; print(f"⚠️ Features list not found at {FEATURES_PATH}")

            if os.path.exists(KEPLER_DATA_PATH):
                # Mirror the training pipeline so random_candidate samples the same feature space.
                raw_kepler_df = pd.read_csv(KEPLER_DATA_PATH, comment='#')
                cleaned_kepler = clean_dataset(raw_kepler_df, 'Kepler', 'koi_disposition')
                kepler_df_cleaned_std = standardize_columns(cleaned_kepler, 'Kepler')
                print(f"✅ Kepler data cleaned and standardized for random candidates.")
            else:
                print(f"⚠️ Kepler data not found at {KEPLER_DATA_PATH}")
                kepler_df_cleaned_std = pd.DataFrame()

            if os.path.exists(STATS_PATH):
                with open(STATS_PATH, 'r') as f: model_stats = json.load(f)
                print(f"✅ Model stats loaded from {STATS_PATH}")
            else:
                print(f"⚠️ Model stats not found at {STATS_PATH}")
                model_stats = {'accuracy': 0, 'f1Score': 0, 'precision': 0, 'recall': 0}

        except Exception as e:
            print(f"❌ CRITICAL ERROR during artifact loading: {e}")
            return False

    if artifacts_loaded:
        print("✅ All artifacts loaded successfully.")
    else:
        print("🛑 Some artifacts are missing. Please run `python model_trainer.py` to generate them.")
    return artifacts_loaded

@app.route('/model_stats', methods=['GET'])
def get_model_stats():
    """Serves the real model statistics loaded from the JSON file."""
    if model_stats:
        return jsonify(model_stats)
    else:
        return jsonify({'error': 'Model statistics not available'}), 500

@app.route('/random_candidate', methods=['GET'])
def get_random_candidate():
    """Selects a random candidate from the PRE-CLEANED data pool, ensuring all features are present."""
    if kepler_df_cleaned_std is None or kepler_df_cleaned_std.empty or features is None:
        return jsonify({'error': 'Server data not fully loaded for random candidates'}), 500

    random_row = kepler_df_cleaned_std[features].dropna().sample(1)

    return jsonify(random_row.to_dict('records')[0])

@app.route('/predict', methods=['POST'])
def predict_single():
    """Handles single predictions from sliders or manual entry."""
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

        prediction_class = 'CONFIRMED' if probability > 0.5 else 'FALSE POSITIVE'

        importance_data = [{'feature': feat, 'value': float(imp)} for feat, imp in zip(features, feature_importances)]

        return jsonify({
            'prediction': prediction_class,
            'probability': float(probability),
            'featureImportance': sorted(importance_data, key=lambda x: x['value'], reverse=True)
        })
    except Exception as e:
        return jsonify({'error': f"Prediction error: {str(e)}"}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Handles batch predictions AND saves the uploaded CSV."""
    with model_lock:
        if not all([model, scaler, features]):
            return jsonify({'error': 'Backend not ready.'}), 500

    if 'file' not in request.files or not request.files['file'].filename:
        return jsonify({'error': 'No file selected'}), 400

    file = request.files['file']

    try:
        # Persist a copy of the upload so analysts can inspect the batch later.
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_path = os.path.join(USER_DATA_PATH, f"{timestamp}_{filename}")
        os.makedirs(USER_DATA_PATH, exist_ok=True)
        file.save(save_path)
        print(f"✅ User file saved to: {save_path}")
        file.seek(0)
    except Exception as e:
        print(f"⚠️ Error saving user file: {str(e)}")

    try:

        df = pd.read_csv(file, comment='#')

        df_std = standardize_columns(df, 'Kepler')

        if not all(feat in df_std.columns for feat in features):
            missing = [f for f in features if f not in df_std.columns]
            return jsonify({'error': f'Uploaded CSV is missing required standardized columns: {missing}'}), 400

        df_clean = df_std[features].dropna()
        if df_clean.empty:
            return jsonify({'error': 'No valid rows with all required features found in the uploaded CSV.'}), 400

        scaled_data = scaler.transform(df_clean)
        probabilities = model.predict_proba(scaled_data)[:, 1]

        df_clean['probability'] = probabilities
        df_clean['prediction'] = df_clean['probability'].apply(
            lambda p: 'CONFIRMED' if p > 0.5 else 'FALSE POSITIVE'
        )

        return jsonify(df_clean.to_dict('records'))
    except Exception as e:

        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/retrain', methods=['POST'])
def retrain_model_endpoint():
    """Triggers model retraining in a background thread."""
    hyperparams = request.get_json() or {}
    print(f"Received retraining request with parameters: {hyperparams}")

    def training_task():
        print("--- Starting background retraining task ---")
        results = run_training_pipeline(hyperparams)
        if 'error' not in results:
            print("--- Retraining successful. Reloading all artifacts... ---")
            load_artifacts()
        else:
            print(f"--- Retraining failed: {results['error']} ---")

    thread = threading.Thread(target=training_task)
    thread.start()
    return jsonify({'message': 'Model retraining started. The new model will be loaded upon completion.'}), 202

if __name__ == '__main__':
    load_artifacts()
    print("\n🚀 Exo-Explorer Backend is running!")
    print("   Ready to accept requests at http://127.0.0.1:5001")
    app.run(debug=True, port=5001)
