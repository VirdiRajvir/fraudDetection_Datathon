import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score

data = pd.read_csv("fraudDetection_Datathon/csv files/undersampled_numeric_timeconvert.csv")
y = data['is_fraud']
X = data.drop(['is_fraud','Unnamed: 0', 'amt'], axis=1)

data_test = pd.read_csv("fraudDetection_Datathon/csv files/test_undersampled_numerics.csv")
y_test = data_test['is_fraud']
X_test = data_test.drop(['is_fraud','Unnamed: 0', 'amt'], axis=1)

# Create the random grid
param_grid = {
    'n_estimators': [50],  # Fewer trees to prevent overfitting
    'max_features': ['sqrt'],  # Use square root of features to reduce complexity
    'max_depth': [10],  # Significantly limit tree depth
    'min_samples_split': [10],  # Increase min samples to split a node
    'min_samples_leaf': [10],  # Increase min samples per leaf
    'bootstrap': [True]
}


rf_Model = RandomForestClassifier()
rf_Model = RandomForestClassifier(class_weight='balanced')
rf_Grid = GridSearchCV(estimator = rf_Model, param_grid = param_grid, cv = 5, scoring = "recall", verbose = 2, n_jobs = 4)
rf_Grid.fit(X, y)

rf_Grid.best_params_


# Test set prediction
y_proba = rf_Grid.predict_proba(X_test)[:, 1]  # Probability for class 1 (fraud)



print(f'Train Accuracy: {rf_Grid.score(X, y):.3f}')
print(f'Test Accuracy: {rf_Grid.score(X_test, y_test):.3f}')
y_pred = rf_Grid.predict(X_test)
print(classification_report(y_test, y_pred))
print(f'AUC-ROC: {roc_auc_score(y_test, y_pred):.3f}')

print(X.shape)


importances = rf_Grid.best_estimator_.feature_importances_
sorted_importances = sorted(zip(importances, X.columns), reverse=True)
print("Feature importances (top 5):", sorted_importances[:5])

results = rf_Grid.cv_results_

# Display all parameter combinations and their corresponding scores
for i in range(len(results['params'])):
    print(f"Parameters: {results['params'][i]}, Mean Test Score: {results['mean_test_score'][i]:.5f}")
