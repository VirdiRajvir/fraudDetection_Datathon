import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score

# Load dataset
data = pd.read_csv("/Users/rsv/Documents/Datathon/fraudDetection_Datathon/csv files/undersampled_numeric_timeconvert.csv")
y = data['is_fraud']
X = data.drop(['is_fraud', 'Unnamed: 0', 'amt'], axis=1)

# Define a single test case (replace these values with your test case values)
# Example Test Case: 11.69	128	13	1320.92	2	436	16	314	61	1047859200.00	1	7.186083743
test_case = [11.69, 128, 13, 2, 436, 16, 314, 61, 1047859200.00, 7.186083743]
test_case = np.array(test_case).reshape(1, -1)  # Reshape to 1 sample with multiple features

# Create the random grid
param_grid = {
    'n_estimators': [50],  # Fewer trees to prevent overfitting
    'max_features': ['sqrt'],  # Use square root of features to reduce complexity
    'max_depth': [10],  # Significantly limit tree depth
    'min_samples_split': [10],  # Increase min samples to split a node
    'min_samples_leaf': [10],  # Increase min samples per leaf
    'bootstrap': [True]
}

# Set up Random Forest model
rf_Model = RandomForestClassifier(class_weight='balanced')
rf_Grid = GridSearchCV(estimator=rf_Model, param_grid=param_grid, cv=5, scoring="recall", verbose=2, n_jobs=4)
rf_Grid.fit(X, y)

# Test prediction for the specific test case
y_pred = rf_Grid.predict(test_case)
y_proba = rf_Grid.predict_proba(test_case)

# Output predicted result for the single test case
print(f'Predicted class for test case: {"Fraud" if y_pred[0] == 1 else "Non-Fraud"}')
print(f'Probability of Fraud: {y_proba[0][1]:.4f}')
print(f'Probability of Non-Fraud: {y_proba[0][0]:.4f}')
