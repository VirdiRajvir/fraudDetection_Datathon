import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

# Load the data
train_df = pd.read_csv('/Users/bharathsreekumarmenon/Downloads/undersampled_numeric_timeconvert.csv')
test_df = pd.read_csv('/Users/bharathsreekumarmenon/Downloads/test_undersampled_numerics.csv')

# Drop unnecessary columns
train_df = train_df.drop(columns=['amt', 'X'], errors='ignore')
test_df = test_df.drop(columns=['amt', 'X'], errors='ignore')

# Split into features and target
X_train = train_df.drop(columns=['is_fraud'])  # Features for training
y_train = train_df['is_fraud']  # Target (Fraud or not) for training
X_test = test_df.drop(columns=['is_fraud'])  # Features for testing
y_test = test_df['is_fraud']  # Target (Fraud or not) for testing

# Scale all features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to DataFrame for easier column management
X_train_final = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_final = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Create individual models
logistic_model = LogisticRegression(class_weight='balanced',solver = 'liblinear', max_iter=400, C=0.1)

param_grid = {
    'n_estimators': [50],
    'max_features': ['sqrt'],
    'max_depth': [10],
    'min_samples_split': [10],
    'min_samples_leaf': [10],
    'bootstrap': [True]
}

rf_Model = RandomForestClassifier()
rf_Model = RandomForestClassifier(class_weight='balanced')
rf_Grid = GridSearchCV(estimator=rf_Model, param_grid=param_grid, cv=5, scoring="recall", verbose=2, n_jobs=4)
rf_Grid.fit(X_train_final, y_train)

rf_Grid.best_params_

svm_model = SVC(kernel='linear', probability=True)  # SVM model with probability enabled

# Train the models separately
logistic_model.fit(X_train_final, y_train)
svm_model.fit(X_train_final, y_train)
rf_Model.fit(X_train_final, y_train)

# Stacking model
estimators = [('logistic', logistic_model), ('svm', svm_model), ('rf', rf_Grid)]
stack_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000, C = 0.3))

# Train stacked model
stack_model.fit(X_train_final, y_train)

# Predictions for train and test sets
y_train_pred = stack_model.predict(X_train_final)
y_test_pred = stack_model.predict(X_test_final)

# Evaluate the model performance
def evaluate_performance(y_true, y_pred, dataset_type="Test"):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    print(f"{dataset_type} Performance:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"F1 Score: {f1}")
    print(f"Recall: {recall}")
   

evaluate_performance(y_train, y_train_pred, dataset_type="Training")
evaluate_performance(y_test, y_test_pred, dataset_type="Test")

# Feature importance (only for Random Forest model)
importances = rf_Model.feature_importances_
sorted_importances = sorted(zip(importances, X_train_final.columns), reverse=True)
print("Feature importances (top 5):", sorted_importances[:5])

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False)
plt.xlabel('Predicted Fraud')
plt.ylabel('True Fraud')
plt.title('Confusion Matrix (Stacked Model)')
plt.show()

# ROC Curve
y_prob = stack_model.predict_proba(X_test_final)[:, 1]  # Probability of fraud (class 1)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Random chance line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Stacked Model')
plt.legend(loc='lower right')
plt.show()

final_model = stack_model.final_estimator_

# Print the coefficients associated with each base model
print("Coefficients of the final logistic regression model:")
print(final_model.coef_)