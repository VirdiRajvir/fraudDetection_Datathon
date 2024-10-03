import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score

data = pd.read_csv("fraudDetection_Datathon/csv files/undersampled_numeric_timeconvert.csv")
y = data['is_fraud']
X = data.drop('is_fraud', axis=1)
X = data.drop('Unnamed: 0', axis=1)
X = data.drop('logamt', axis=1)

data_test = pd.read_csv("fraudDetection_Datathon/csv files/test_undersampled_numerics.csv")
y_test = data_test['is_fraud']
X_test = data_test.drop('is_fraud', axis=1)
X_test = data_test.drop('Unnamed: 0', axis=1)
X_test = data_test.drop('logamt', axis=1)

# # Number of trees in forest
# n_estimators = [100, 200, 300, 500]
# #number of features to consider at every split
# max_features = ['auto', 'sqrt']
# #maximum number of levels in tree
# max_depth = [10, 20, None]
# #minimum number of samples required to split a node
# min_samples_split = [10,20]
# #minimum number of samples required at each leaf node
# min_samples_leaf = [2,4]
# #method of selecting samples for training tree
# bootstrap = [True, False]

# Create the random grid
param_grid = {
    'n_estimators': [50, 100],  # Fewer trees to prevent overfitting
    'max_features': ['sqrt'],  # Use square root of features to reduce complexity
    'max_depth': [5, 10],  # Significantly limit tree depth
    'min_samples_split': [10, 20],  # Increase min samples to split a node
    'min_samples_leaf': [5, 10],  # Increase min samples per leaf
    'bootstrap': [True]
}


rf_Model = RandomForestClassifier()
rf_Model = RandomForestClassifier(class_weight='balanced')
rf_Grid = GridSearchCV(estimator = rf_Model, param_grid = param_grid, cv = 5, scoring = "recall", verbose = 2, n_jobs = 4)
rf_Grid.fit(X, y)

rf_Grid.best_params_

# rf_Model.fit(X, y)

# Test set prediction
y_proba = rf_Grid.predict_proba(X_test)[:, 1]  # Probability for class 1 (fraud)

# Adjust the classification threshold to capture more fraud cases
threshold = 0.3  # Set to a lower threshold for fraud detection
y_pred_adjusted = (y_proba >= threshold).astype(int)


print(f'Train Accuracy: {rf_Grid.score(X, y):.3f}')
print(f'Test Accuracy: {rf_Grid.score(X_test, y_test):.3f}')
y_pred = rf_Grid.predict(X_test)
print(classification_report(y_test, y_pred_adjusted))
print(f'AUC-ROC: {roc_auc_score(y_test, y_pred):.3f}')


importances = rf_Grid.best_estimator_.feature_importances_
sorted_importances = sorted(zip(importances, X.columns), reverse=True)
print("Feature importances (top 5):", sorted_importances[:5])
# print(data.head())
