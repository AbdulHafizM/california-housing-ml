# California Housing Price Prediction Pipeline

This repository contains an end-to-end machine learning workflow designed to predict median house values across California districts using the 1990 census dataset. Built entirely with `scikit-learn`, the project demonstrates production-ready data science practices, including custom feature engineering, robust data transformation pipelines, and rigorous prevention of target leakage.

## System Architecture

The pipeline processes raw geographic and demographic data into scaled, encoded features ready for model ingestion. 

* **Data Segregation:** The target variable (`median_house_value`) is fully isolated from the training predictors prior to any transformations to prevent data leakage.
* **Exploratory Data Analysis (EDA):** Leverages `seaborn` and `matplotlib` to map district coordinates, visualize housing value distributions, and extract high-correlation features via a localized correlation matrix.
* **Custom Feature Engineering:** Implements a custom `BaseEstimator` and `TransformerMixin` class (`CombinedAttributesAdder`) to dynamically compute context-rich metrics:
  * `rooms_per_household`
  * `bedrooms_per_room`
  * `population_per_household`
* **Preprocessing Pipeline:** Utilizes a `ColumnTransformer` to route distinct data types through specialized operations:
  * **Numerical Features:** Processed via `SimpleImputer` (median strategy) to handle missing values, followed by `StandardScaler` for zero-mean, unit-variance standardization.
  * **Categorical Features:** Handled via `OneHotEncoder` to create binary vector representations of the `ocean_proximity` attribute.
* **Model Training & Evaluation:** Fits predictive models (baseline `LinearRegression`, extensible to `RandomForestRegressor` via `GridSearchCV`) and evaluates final performance on a hold-out test set using Root Mean Squared Error (RMSE).

## Repository Structure

```text
├── data/
│   └── housing.csv          # Raw California housing dataset
├── notebooks/
│   └── housing.ipynb        # Core ML pipeline and exploratory data analysis
├── src/                     # (Optional) Extracted python modules for production
│   └── custom_transformers.py
├── .gitignore
├── requirements.txt
└── README.md
