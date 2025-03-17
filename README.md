This repository demonstrates a Multi-Factor Modelling approach to analyze and predict CAGR (Compound Annual Growth Rate) using various machine learning techniques, including Linear Regression, Random Forest, and XGBoost.

🛠 Libraries Used

pandas

matplotlib

sklearn (LinearRegression, RandomForest, XGBClassifier, etc.)

statsmodels

xgboost

📂 Files

WSC_case-study_ds.xlsx: Input dataset containing company factors.

Return.xlsx: File containing the Annualized Return data (CAGR).

filename.xlsx: Generated output from the pivot table.

MultiFactorModelling.ipynb: The main Python script that performs data analysis and machine learning.

📊 Data Cleaning and Processing

Factor Identification: The script loads data and identifies unique factors and companies from the dataset.

Data Transformation: The dataset is transformed, and pivot tables are created to structure the data for modeling.

Feature Engineering: The CAGR_category column is generated based on CAGR values, classifying them into Low, Medium, or High.

🔬 Modeling and Analysis

1. Linear Regression

A basic linear regression model is created to understand the relationship between the selected factors and CAGR.

The model outputs the intercept, coefficients, and a summary of significant factors.

2. Random Forest Classifier

A Random Forest model is used to predict CAGR_category (Low, Medium, High).

The script evaluates model accuracy and generates a classification report.

The top 10 most important features (factors) are also displayed.

3. Area Under Curve (AUC)

ROC curves are generated for each class (Low, Medium, High), and the AUC is computed to evaluate the classifier's performance.

4. XGBoost Classifier

An XGBoost model is used for classification and feature selection.

Dimensionality reduction is performed using Variance Inflation Factor (VIF) to eliminate collinear features.

A classification report and feature importance plot are generated.

📈 Visualizations

ROC Curves: Plot for multi-class classification showing the AUC for each category (Low, Medium, High).

Feature Importance: Bar plots to show the top 10 most influential features in the prediction.
