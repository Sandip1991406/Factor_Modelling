# %% [markdown]
# # MultiFactor Modelling

# %% [markdown]
# ## Library Imports

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBClassifier

# %% [markdown]
# ## Load Data
factors = pd.read_excel('WSC_case-study_ds.xlsx')
factors.head()

# %% [markdown]
# ## Data Cleaning & Processing

unique_factors = factors['Criteria ID'].unique()
list_unique_factors = list(unique_factors)

unique_factors_count = factors['Criteria ID'].nunique()
print(f"Unique Factor Count: {unique_factors_count}")

companies_name = factors['Company'].unique()
list_companies = list(companies_name)

unique_companies_count = factors['Company'].nunique()
print(f"Unique Company Count: {unique_companies_count}")

# Creating a DataFrame with repeated company names for two years (2020 & 2023)
years = [2020, 2023] * len(list_companies)
df = pd.DataFrame({'company_name': list_companies * 2, 'year': years})
df.head()

df_pivot = factors.pivot_table(index=['Company', 'Year'], columns='Criteria ID', values='Score').reset_index()
df_pivot.to_excel("cleaned_data.xlsx", index=False)

# %% [markdown]
# ## Merge with Annualized Returns

rets = pd.read_excel('Return.xlsx')  # CAGR - Cumulative Annual Growth Rate
df3 = df_pivot.merge(rets[['Company', 'Year', 'CAGR']], on=['Company', 'Year'], how='left')

# Categorizing CAGR into Low, Medium, and High
df3['CAGR_category'] = pd.cut(df3['CAGR'], bins=[-float('inf'), 0.2, 0.8, float('inf')], labels=['Low', 'Medium', 'High'])

# Cloning data for different models
df4, df5, df6, df7 = df3.copy(), df3.copy(), df3.copy(), df3.copy()

# %% [markdown]
# ## Factor Modelling

# %% [markdown]
# ### Linear Regression

X = df6.drop(columns=['CAGR', 'Company', 'Year', 'CAGR_category'], errors='ignore')
y = df6['CAGR']

# Sklearn Linear Regression
model = LinearRegression()
model.fit(X, y)
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")

# Statsmodels OLS Regression
X = sm.add_constant(X)
olsm_model = sm.OLS(y, X).fit()
print(olsm_model.summary())

# Extracting significant factors
significant_factors = olsm_model.summary2().tables[1]
significant_factors = significant_factors[significant_factors['P>|t|'] < 0.05]
print("Significant Factors (p < 0.05):")
print(significant_factors)

# %% [markdown]
# ### Random Forest Classification

X = df4.drop(columns=['Company', 'Year', 'CAGR', 'CAGR_category'])
y = df4['CAGR_category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Feature Importance
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Top 10 Important Factors:")
print(feature_importances.head(10))

# %% [markdown]
# ### Multi-Class ROC and AUC Curve

y_bin = label_binarize(y, classes=['Low', 'Medium', 'High'])
n_classes = y_bin.shape[1]
X_train, X_test, y_train_bin, y_test_bin = train_test_split(X, y_bin, test_size=0.2, random_state=42)

rf_model = OneVsRestClassifier(RandomForestClassifier(random_state=42))
rf_model.fit(X_train, y_train_bin)
y_score = rf_model.predict_proba(X_test)

# Compute ROC curve and ROC area for each class
fpr, tpr, roc_auc = {}, {}, {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
colors = ['blue', 'orange', 'green']
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label=f'ROC curve of class {i} (AUC = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-Class ROC and AUC Curve')
plt.legend(loc="lower right")
plt.show()

# %% [markdown]
# ### Dimensionality Reduction using XGBoost

def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

# Remove features with high VIF
X = df7.drop(columns=['Company', 'Year', 'CAGR', 'CAGR_category'])
while True:
    vif_data = calculate_vif(X)
    high_vif_features = vif_data[vif_data["VIF"] > 10]
    if high_vif_features.empty:
        break
    feature_to_remove = high_vif_features.sort_values("VIF", ascending=False).iloc[0]["Feature"]
    X = X.drop(columns=[feature_to_remove])

# Encoding target variable
y_encoded = LabelEncoder().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Feature Importance
feature_importances = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
feature_importances.head(10).plot(kind='bar', figsize=(10, 6))
plt.title("Top 10 Features")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.xticks(rotation=45)
plt.show()