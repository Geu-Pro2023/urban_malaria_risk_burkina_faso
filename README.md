# Predictive Modelling of Urban Malaria Risk Using Spatial-Temporal Survey and Environmental Data — Burkina Faso

![Malaria Risk Map](Images/malaria_risk_map.png)

## Project Overview

Malaria remains a major public health concern in Burkina Faso. This project aims to predict urban malaria prevalence using a combination of spatial-temporal survey data and environmental covariates. By integrating Random Forest and XGBoost models with geospatial analysis, we identify hotspots of malaria risk in Ouagadougou and highlight key environmental drivers.  

This work demonstrates actionable modeling for public health interventions in low- and middle-income countries (LMICs).

---

## Author

**Geu Aguto Garang Bior**  
Software Engineering Student – Machine Learning | Health Mission Student  

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Data](#data)  
- [Methods](#methods)  
- [Machine Learning Models](#machine-learning-models)  
- [Model Evaluation](#model-evaluation)  
- [Spatial Mapping](#spatial-mapping)  
- [Results](#results)  
- [Folder Structure](#folder-structure)  
- [Usage](#usage)  
- [Requirements](#requirements)  
- [References](#references)  

---

## Data

The project uses malaria prevalence survey data and environmental variables for Burkina Faso:

- **Source:** Survey data for Ouagadougou, Burkina Faso  
- **Variables:**  
  - Geographic coordinates (LAT, LONG)  
  - Year, Month  
  - Vegetation Index (EVI)  
  - Temperature Suitability Index (TSI)  
  - Precipitation (PRES_N, PRES_MM)  
  - Season (SEASON_Wet123)  
  - PfPR2_10 (Plasmodium falciparum prevalence in children 2–10 years)

> **Note:** Sensitive or restricted datasets are stored locally in `data/raw/` and are not included in the repo.

---

## Methods

1. **Data Preprocessing**  
   - Handle missing values for environmental covariates using median imputation  
   - Convert categorical/seasonal variables to numeric  
   - Filter out rows with missing target variable  

2. **Train-Test Split**  
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Machine Learning Models
### Random Forest Regressor:
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train, y_train)


### XGBoost Regressor:
from xgboost import XGBRegressor
xgb = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train)


## Model Evaluation
- Metrics: RMSE, R²
- Residual analysis: Check for bias in predictions
- Feature importance: Visualize key drivers of malaria risk

## Spatial Mapping
- Convert predictions to GeoDataFrame using geopandas
- Generate malaria risk maps to visualize hotspots across Ouagadougou
-

## Results
### Random Forest Performance:
- RMSE: 11.38
- R²: 0.38

### XGBoost Performance:
- RMSE: 10.92
- R²: 0.42

### Key Drivers of Malaria Risk:
- Vegetation Index (EVI)
- Seasonality
- Temperature Suitability Index (TSI)

### Spatial Hotspots:
- High-risk zones identified in central and northern districts of Ouagadougou
Outputs (maps and CSV predictions) are available in the outputs/ folder.


## Folder Structure
urban-malaria-risk-burkina-faso/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── data/
│   ├── burkina_malaria_data.csv                # Original datasets
│   
│
├── notebooks/
│   ├──urban_malaria_prediction_burkina_faso.ipynb
│  
│
├── models/                     # Saved trained models
│   ├── random_forest_model.pkl
│   └── xgboost_model.pkl
│
├── Images/                     # Saved plots and maps
│   ├── feature_importance.png
│   └── malaria_risk_map.png
│
└── outputs/                    # Final outputs for presentation/report
    └── model_metrics.txt

## Usage
### Clone the repository:
git clone https://github.com/Geu-Pro2023/urban_malaria_risk_burkina_faso.git
cd urban-malaria-risk-burkina-faso

### Create and activate a virtual environment:
python -m venv venv
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

### Install dependencies:
pip install -r requirements.txt


## Requirements
- Python 3.9+
- pandas, numpy, scikit-learn, xgboost
- matplotlib, seaborn
- geopandas, contextily

## References
- Baragatti, M., et al. (2009). Malaria prevalence survey data, Burkina Faso.
- WHO. World Malaria Report 2022.
- Scikit-learn Documentation
- XGBoost Documentation

## License
This project is licensed under the MIT License – see the LICENSE file for details.
