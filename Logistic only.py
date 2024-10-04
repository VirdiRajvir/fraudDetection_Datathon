'''#LOGISTICC'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_curve, roc_auc_score


train_df = pd.read_csv('/Users/bharathsreekumarmenon/Downloads/undersampled_numeric_timeconvert.csv')  
test_df = pd.read_csv('/Users/bharathsreekumarmenon/Downloads/test_undersampled_numerics.csv')    
train_df = train_df.drop(columns=['amt', 'X'])
test_df = test_df.drop(columns=['amt', 'X'])


X_train = train_df.drop(columns=['is_fraud'])  # Features for training
y_train = train_df['is_fraud']  # Target (Fraud or not) for training

# Define features (X) and target (y) 
X_test = test_df.drop(columns=['is_fraud'])  # Features for testing
y_test = test_df['is_fraud']  # Target (Fraud or not) for testing

#  Scaling
selected_features = ['logamt', 'city_pop', 'dob_unix']  
X_train_subset = X_train[selected_features]
X_test_subset = X_test[selected_features]

scaler = StandardScaler()
X_train_scaled_subset = scaler.fit_transform(X_train_subset)  # Fit and transform on the training data
X_test_scaled_subset = scaler.transform(X_test_subset)  # Only transform the test data (don't fit again)

# Convert scaled data back to a DataFrame 
X_train_scaled_subset_df = pd.DataFrame(X_train_scaled_subset, columns=selected_features)
X_test_scaled_subset_df = pd.DataFrame(X_test_scaled_subset, columns=selected_features)

# Combine scaled selected features with the unscaled features (categorical and others)
X_train_unscaled = X_train.drop(columns=selected_features)  # Keep the unscaled categorical and other features
X_test_unscaled = X_test.drop(columns=selected_features)  # Same for the test set

X_train_final = pd.concat([X_train_scaled_subset_df, X_train_unscaled.reset_index(drop=True)], axis=1)  # Combine scaled and unscaled features for train
X_test_final = pd.concat([X_test_scaled_subset_df, X_test_unscaled.reset_index(drop=True)], axis=1)  # Combine for test

# Train the Logistic Regression model on the training data
logistic_model = LogisticRegression(class_weight = 'balanced',solver= 'saga',  max_iter=500)
logistic_model.fit(X_train_final, y_train)

# Evaluate the model on the test data
y_pred = logistic_model.predict(X_test_final)

y_pred_prob = logistic_model.predict_proba(X_test_final)[:, 1]  # Get probabilities for the positive class (fraud)

threshold = 0.5
y_pred_new = (y_pred_prob >= threshold).astype(int)

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Calculate metrics on the test data
accuracy = accuracy_score(y_test, y_pred_new)
recall = recall_score(y_test, y_pred_new)
precision = precision_score(y_test, y_pred_new)
f1 = f1_score(y_test, y_pred_new)

# Print the evaluation results
print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"F1 Score: {f1}")


#FEATURE IMPORTANCE LIST
feature_names = X_train_final.columns
coefficients = logistic_model.coef_[0]

# Create a DataFrame to display the feature importance
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})

# Sort by the absolute value of the coefficient to see the most important features
importance_df['Importance'] = np.abs(importance_df['Coefficient'])
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Display the sorted feature importance
print(importance_df[['Feature', 'Coefficient', 'Importance']])



#CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred_new)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False)
plt.xlabel('Predicted Fraud')
plt.ylabel('True Fraud')
plt.title('Confusion Matrix(Logistic)')
plt.show()


#ROC CURVE
y_prob = logistic_model.predict_proba(X_test_final)[:, 1]


fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Calculate Area Under the Curve
roc_auc = roc_auc_score(y_test, y_prob)


plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()









