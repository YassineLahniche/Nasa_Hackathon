# Exoplanet Classification Project

This project analyzes NASA's exoplanet datasets from Kepler, K2, and TESS missions to build a machine learning model that can identify new exoplanets. It includes a Jupyter notebook for analysis and model development, plus a Streamlit web application for user interaction.

## Project Structure

- `Exoplanet_Classification_Project.ipynb`: Main Jupyter notebook with data analysis, model training, and evaluation
- `exoplanet_app.py`: Streamlit web application for interactive exoplanet classification
- `exoplanet_model.pkl`: Trained machine learning model (created by the notebook)
- `sample_candidates.csv`: Sample dataset for testing batch classification

## Setup Instructions

1. **Install Required Libraries**

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter streamlit
```

2. **Run the Jupyter Notebook**

Open and run the Jupyter notebook to perform data analysis and train the model:

```bash
jupyter notebook Exoplanet_Classification_Project.ipynb
```

3. **Launch the Streamlit Application**

After running the notebook (which saves the trained model), launch the web app:

```bash
streamlit run exoplanet_app.py
```

## Using the Web Application

The Streamlit application offers three main functionalities:

1. **Manual Input**: Enter transit parameters using sliders and get instant classification
2. **File Upload**: Upload a CSV file with multiple candidates for batch classification
3. **About Exoplanets**: Educational information about exoplanets and detection methods

For batch classification, ensure your CSV has columns matching the feature names used in training.

## Model Features

The model analyzes various transit parameters including:

- Transit duration (hours)
- Transit depth (ppm)
- Orbital period (days)
- Planet radius (Earth radii)
- Stellar radius (Solar radii)
- Stellar temperature (K)
- Insolation flux (Earth flux)

## Interactive Prediction System

The notebook includes an interactive widget system for testing the model within Jupyter. This allows for quick parameter adjustments and instant predictions.

## Data Sources

This project uses data from:

- NASA Exoplanet Archive
- Kepler, K2, and TESS mission databases

## License

This project is for educational purposes. All data used is publicly available from NASA sources.