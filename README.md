# Exo-Explorer: NASA Hackathon Project

Exo-Explorer is a NASA Space Apps hackathon project that combines a LightGBM-based classifier with an interactive web dashboard to explore exoplanet candidates from the Kepler, K2, and TESS missions. The repository includes the full machine-learning pipeline, a Flask REST API that serves predictions and model telemetry, and a rich frontend experience for both exploratory and batch analysis.

## Repository Layout

- `backend/` – Flask API, model training pipeline, model artifacts, and CSV datasets used for inference
- `frontend/` – Static dashboard built with HTML/CSS/JS (Chart.js) that consumes the backend API
- `Notebooks/` – Jupyter notebook and raw CSV files for experimentation and offline analysis
- `README.md` – Project documentation (this file)

## Data Processing & Feature Engineering

The training pipeline in `backend/model_trainer.py` orchestrates the full data preparation flow:

1. **Dataset ingestion** – Loads Kepler (`koi.csv`), K2 (`k2.csv`), and TESS (`tess.csv`) archives from `backend/data/` (with the same files mirrored under `Notebooks/` for experimentation).
2. **Cleaning per mission** – Drops empty rows, enforces essential columns, fills numeric gaps with medians, and normalises disposition labels into a shared `exoplanet_class` field before deriving the binary target `is_exoplanet`.
3. **Column standardisation** – Maps mission-specific column names to a unified schema (`orbital_period`, `transit_duration`, `transit_depth`, `planet_radius`, `insolation_flux`, `stellar_radius`, etc.) so downstream steps can treat every mission identically.
4. **Dataset merge** – Filters each mission to the shared feature set and concatenates them into one table with a `source` column for traceability.
5. **Feature selection** – Keeps the core transit and stellar descriptors that survive across missions: `orbital_period`, `transit_duration`, `transit_depth`, `planet_radius`, `insolation_flux`, `stellar_radius` (the deployed API currently consumes the first four of these, defined in `models/features.json`).
6. **Scaling & splits** – Applies `StandardScaler` to the selected features and performs a stratified 80/20 train-test split to preserve confirmed vs false-positive balance.

This pipeline is rerunnable; execute `python backend/model_trainer.py` whenever the datasets or hyperparameters change to regenerate refreshed artifacts.

## Model Architecture & Evaluation

- **Primary model** – A single LightGBM binary classifier (`lightgbm.LGBMClassifier`) tuned for imbalanced transit detection with early stopping on a validation fold.
- **Supporting components** – `StandardScaler` for feature normalisation and feature ordering persisted in `features.json` to keep the API and frontend in sync.
- **Artifacts** – Stored under `backend/models/`: `lightgbm_model.pkl`, `scaler.pkl`, `features.json`, and `model_stats.json`.
- **Metrics** – Accuracy, precision, recall, and F1-score are computed after training and exposed through the REST API. See `model_stats.json` for the latest values.
- **Extensibility** – The trainer is modular; additional estimators can be swapped in by modifying `run_training_pipeline` before persisting new artifacts. LightGBM is the only model currently shipped to production for hackathon submissions.

## Backend API

### Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python model_trainer.py     # optional if artifacts already exist
python app.py               # starts the API on http://127.0.0.1:5001
```

### Endpoints

| Method | Path                | Description |
| ------ | ------------------- | ----------- |
| GET    | `/model_stats`      | Returns metrics captured in `model_stats.json` |
| GET    | `/random_candidate` | Samples a mission record, renames features, and returns a model-ready payload |
| POST   | `/predict`          | Accepts JSON with the features listed in `features.json`; responds with class label, probability, and feature importances |
| POST   | `/predict_batch`    | Placeholder batch endpoint; currently returns a stub message until CSV handling is implemented |
| POST   | `/retrain`          | Launches background retraining with optional LightGBM hyperparameters and reloads artifacts on success |

CORS is enabled via `flask-cors`, so the frontend can run from any origin (localhost or hosted).

## Frontend Application

- **Explorer mode** – Slider-controlled parameter tuning with live predictions, random-candidate generation against Kepler data, confidence animations, and a feature-importance chart.
- **Researcher mode** – Manual numeric form, CSV upload workflow (awaiting backend batch support), and results table with quick statistics on confirmed/candidate/false-positive counts.
- **Notifications & shortcuts** – Lightweight toast-style console notifications plus keyboard shortcuts for mode toggles and random sampling (`Ctrl/Cmd + 1`, `Ctrl/Cmd + 2`, `Ctrl/Cmd + R`).
- **Configurable backend** – `API_BASE_URL` in `frontend/script.js` centralises the REST endpoint so deployments can point to remote servers without code changes.

### Running Locally

Serve the frontend through any static file server to avoid CORS issues in browsers:

```bash
cd frontend
python -m http.server 5500
```

Open http://127.0.0.1:5500 while the Flask API is running on port 5001.

### Hosting Plan

We are preparing to host the Flask API and dashboard on a managed server. When the production domain is ready:

1. Update `API_BASE_URL` in `frontend/script.js` to the deployed backend URL.
2. Publish the `frontend/` bundle via your hosting provider (or build a simple container that serves the static assets).
3. Re-run `python backend/model_trainer.py` and push the resulting artifacts if retraining is required prior to deployment.

## Notebook & Offline Analysis

`Notebooks/Exoplanet_Classification_Project.ipynb` mirrors the training logic from the backend, expands on exploratory visuals, and allows manual experimentation with candidate profiles. CSVs (`koi.csv`, `k2.csv`, `tess.csv`, `sample_candidates.csv`) in the same directory keep notebook experiments aligned with the deployed pipeline.

## Sample Data & Testing

- `Notebooks/sample_candidates.csv` provides seed inputs for manual or batch tests.
- After starting the backend, verify predictions with:

```bash
curl -X POST http://127.0.0.1:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
        "orbital_period": 10.5,
        "transit_duration": 0.25,
        "insolation_flux": 15.2,
        "stellar_radius": 0.95
      }'
```

This returns a JSON body containing `prediction`, `probability`, and a sorted `featureImportance` array.

## Development Notes

- Retraining saves new artifacts in `backend/models/`; restart the API (or call `/retrain`) to load them.
- Implement CSV parsing inside `/predict_batch` to unlock full batch analytics in the Researcher mode UI.
- Keep datasets in `backend/data/` and `Notebooks/` synchronised to avoid discrepancies between the notebook and deployed service.

## Data Sources

- NASA Exoplanet Archive datasets for Kepler, K2, and TESS missions (public domain)

## License

This repository is intended for educational and hackathon use. Data retains NASA's original licensing terms.
