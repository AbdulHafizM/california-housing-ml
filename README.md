#California Housing Price Prediction


#Project Overview
This repository contains an end-to-end machine learning pipeline designed to predict median house values across California districts. Built with scikit-learn, the project demonstrates core data science workflows including data cleaning, custom feature engineering, rigorous pipeline construction, and model evaluation without data leakage.

#Architecture & Workflow
Data Segregation: Separates predictor features from target labels (median_house_value) immediately after the train-test split to strictly prevent target leakage during transformations.

Exploratory Data Analysis (EDA): Uses correlation matrices and distribution histograms to identify high-impact predictors, such as median district income.

Feature Engineering: Implements a custom scikit-learn transformer (CombinedAttributesAdder) to dynamically generate context-rich ratios:

rooms_per_household

bedrooms_per_room

population_per_household

Preprocessing Pipeline: Utilizes a ColumnTransformer to route data types through appropriate transformations:

Numerical: Missing value imputation via SimpleImputer (median strategy) and feature scaling via StandardScaler.

Categorical: Binary vectorization of ocean_proximity via OneHotEncoder.

Model Training & Evaluation: Fits predictive models (e.g., LinearRegression, RandomForestRegressor via GridSearchCV) onto the prepared dataset and evaluates performance against the test set using Root Mean Squared Error (RMSE).

#Requirements

Python 3.8+

pandas

numpy

matplotlib

seaborn

scikit-learn

#Usage

Clone this repository to your local machine.

Ensure the source dataset housing.csv is placed in the root directory.

Open housing.ipynb in Jupyter Notebook or JupyterLab.

Execute the cells sequentially to ingest the data, build the transformation pipeline, train the model, and output the final RMSE metric.
