# File: backend/model_trainer.py

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import joblib  # Switched to joblib for .pkl files
import json

# --- CONFIGURATION ---
DATA_PATHS = {
    'Kepler': 'data/koi.csv',
    'K2': 'data/k2.csv',
    'TESS': 'data/tess.csv'
}
# Use .pkl for model and scaler files
MODEL_PATH = 'models/lightgbm_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
FEATURES_PATH = 'models/features.json'
STATS_PATH = 'models/model_stats.json'

# --- DATA CLEANING & STANDARDIZATION (Your Provided Functions) ---

def clean_dataset(data, dataset_name, disposition_col=None):
    if data is None:
        print(f"No data for {dataset_name}, skipping.")
        return None
    print(f"Cleaning {dataset_name} dataset...")
    df = data.copy()
    df.dropna(how='all', inplace=True)

    # Define essential columns for each dataset
    essential_cols_map = {
        'Kepler': ['kepoi_name', 'koi_disposition', 'koi_period', 'koi_duration', 'koi_depth'],
        'K2': ['epic_candname', 'k2c_disp', 'pl_orbper', 'pl_trandur'],
        'TESS': ['toi', 'tfopwg_disp', 'pl_orbper', 'pl_trandurh']
    }
    essential_cols = [col for col in essential_cols_map.get(dataset_name, []) if col in df.columns]
    if essential_cols:
        df.dropna(subset=essential_cols, inplace=True)

    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    if disposition_col and disposition_col in df.columns:
        if dataset_name == 'Kepler':
            df['exoplanet_class'] = df[disposition_col].apply(lambda x: 'CONFIRMED' if x == 'CONFIRMED' else 'CANDIDATE' if x == 'CANDIDATE' else 'FALSE POSITIVE' if x == 'FALSE POSITIVE' else 'UNKNOWN')
        elif dataset_name == 'K2':
            df['exoplanet_class'] = df[disposition_col].apply(lambda x: 'CONFIRMED' if x in ['CONFIRMED', 'C'] else 'CANDIDATE' if x in ['CANDIDATE', 'PC'] else 'FALSE POSITIVE' if x in ['FALSE POSITIVE', 'FP'] else 'UNKNOWN')
        elif dataset_name == 'TESS':
            df['exoplanet_class'] = df[disposition_col].apply(lambda x: 'CONFIRMED' if x in ['CP', 'KP'] else 'CANDIDATE' if x == 'PC' else 'FALSE POSITIVE' if x == 'FP' else 'UNKNOWN')
        
        df = df[df['exoplanet_class'] != 'UNKNOWN']
        df['is_exoplanet'] = df['exoplanet_class'].apply(lambda x: 1 if x in ['CONFIRMED', 'CANDIDATE'] else 0)
    
    df['source'] = dataset_name
    return df

def standardize_columns(df, dataset_name):
    if df is None: return None
    print(f"Standardizing columns for {dataset_name}...")
    data = df.copy()
    mapping = {}
    if dataset_name == 'Kepler':
        mapping = {'kepid': 'star_id', 'kepoi_name': 'object_id', 'koi_period': 'orbital_period', 'koi_duration': 'transit_duration', 'koi_depth': 'transit_depth', 'koi_prad': 'planet_radius', 'koi_insol': 'insolation_flux', 'koi_steff': 'stellar_temp', 'koi_srad': 'stellar_radius', 'koi_slogg': 'stellar_log_g'}
    elif dataset_name == 'K2':
        mapping = {'epic_number': 'star_id', 'epic_candname': 'object_id', 'pl_orbper': 'orbital_period', 'pl_trandur': 'transit_duration', 'pl_radj': 'planet_radius_jupiter', 'pl_insol': 'insolation_flux', 'st_rad': 'stellar_radius'}
        if 'pl_radj' in data.columns and 'pl_rade' not in data.columns:
            data['planet_radius'] = data['pl_radj'] * 11.209 # Jupiter radii to Earth radii
            mapping['planet_radius'] = 'planet_radius'
    elif dataset_name == 'TESS':
        mapping = {'tid': 'star_id', 'toi': 'object_id', 'pl_orbper': 'orbital_period', 'pl_trandurh': 'transit_duration', 'pl_trandep': 'transit_depth', 'pl_rade': 'planet_radius', 'pl_insol': 'insolation_flux', 'st_rad': 'stellar_radius'}
        if 'pl_trandurh' in data.columns:
            data['pl_trandurh'] = data['pl_trandurh'] / 24.0 # Hours to Days

    valid_mapping = {old: new for old, new in mapping.items() if old in data.columns}
    data = data.rename(columns=valid_mapping)
    return data

# --- MAIN TRAINING FUNCTION ---

def run_training_pipeline(hyperparameters=None):
    """
    Main function to orchestrate the full data processing and model training pipeline.
    """
    print("\n--- Starting Full Training Pipeline ---\n")
    
    # 1. Load all datasets
    raw_data = {}
    for name, path in DATA_PATHS.items():
        if os.path.exists(path):
            raw_data[name] = pd.read_csv(path, comment='#')
        else:
            print(f"Warning: Data file not found for {name} at {path}")
            raw_data[name] = None
            
    # 2. Clean each dataset
    kepler_clean = clean_dataset(raw_data['Kepler'], 'Kepler', 'koi_disposition')
    k2_clean = clean_dataset(raw_data['K2'], 'K2', 'k2c_disp')
    tess_clean = clean_dataset(raw_data['TESS'], 'TESS', 'tfopwg_disp')
    
    # 3. Standardize column names
    kepler_std = standardize_columns(kepler_clean, 'Kepler')
    k2_std = standardize_columns(k2_clean, 'K2')
    tess_std = standardize_columns(tess_clean, 'TESS')
    
    # 4. Combine into a unified dataset
    datasets_to_combine = [df for df in [kepler_std, k2_std, tess_std] if df is not None]
    if not datasets_to_combine:
        print("Error: No data available to train the model.")
        return
    
    # Find common feature columns to create the unified dataset
    common_feature_cols = list(set.intersection(*[set(df.columns) for df in datasets_to_combine]))
    essential_cols = ['is_exoplanet', 'source'] # Keep these essential columns
    final_cols = list(set(common_feature_cols + essential_cols))
    
    unified_data = pd.concat([df.filter(items=final_cols) for df in datasets_to_combine], ignore_index=True)
    print(f"\nUnified dataset created with {len(unified_data)} rows and {len(unified_data.columns)} columns.")

    # 5. Define final features and prepare for training
    final_features = [
        'orbital_period', 'transit_duration', 'transit_depth',
        'planet_radius', 'insolation_flux', 'stellar_radius'
    ]
    final_features = [f for f in final_features if f in unified_data.columns]
    
    unified_data.dropna(subset=final_features + ['is_exoplanet'], inplace=True)
    
    X = unified_data[final_features]
    y = unified_data['is_exoplanet']
    
    if X.empty:
        print("Error: No valid training data after cleaning and feature selection.")
        return

    print(f"\nTraining model with {len(X.columns)} features: {final_features}")

    # 6. Split, Scale, and Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Use provided hyperparameters or a default set
    if hyperparameters is None:
        hyperparameters = {'objective': 'binary', 'metric': 'binary_logloss', 'n_estimators': 1000, 'learning_rate': 0.05, 'num_leaves': 31, 'random_state': 42}
    
    model = lgb.LGBMClassifier(**hyperparameters)
    model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], callbacks=[lgb.early_stopping(100, verbose=False)])

     # --- Calculate All Performance Metrics ---
    print("\nCalculating performance metrics...")
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    stats = {
        'accuracy': round(accuracy, 2),
        'f1Score': round(f1, 2),
        'precision': round(precision, 2),
        'recall': round(recall, 2),
        'totalCandidates': len(y) # Total candidates used for training/testing
    }
    print(f"Metrics calculated: {stats}")

    # --- Save All Artifacts ---
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(FEATURES_PATH, 'w') as f:
        json.dump(final_features, f)
    with open(STATS_PATH, 'w') as f:
        json.dump(stats, f) # Save the new stats file
        
    print(f"\n--- Pipeline Finished. Test Accuracy: {stats['accuracy']}% ---")
    print(f"✅ All artifacts, including model_stats.json, have been saved.")
    
    return stats

if __name__ == '__main__':
    run_training_pipeline()