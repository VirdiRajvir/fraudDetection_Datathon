'''#LOGISTICC'''
'''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
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








''' '''SVM MODEL'''
'''df = pd.read_csv('/Users/bharathsreekumarmenon/Downloads/undersampled_dataset.csv')

# Inspect the first few rows of the dataset
print(df.head())

# Step 2: Data Preprocessing

# Drop unnecessary columns
df = df.drop(columns=['Unnamed: 0', 'X'], errors='ignore')

# Convert 'trans_date_trans_time' to datetime and extract useful time components
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['year'] = df['trans_date_trans_time'].dt.year
df['month'] = df['trans_date_trans_time'].dt.month
df['day'] = df['trans_date_trans_time'].dt.day
df['hour'] = df['trans_date_trans_time'].dt.hour
df['minute'] = df['trans_date_trans_time'].dt.minute
df['second'] = df['trans_date_trans_time'].dt.second
df = df.drop(columns=['trans_date_trans_time'])

# Convert 'dob' to 'age'
df['dob'] = pd.to_datetime(df['dob'])
df['age'] = df['year'] - df['dob'].dt.year
df = df.drop(columns=['dob'])

# Step 3: Encoding categorical features
label_encoders = {}
for column in ['merchant', 'category', 'city', 'state', 'job', 'gender']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Step 4: Create the 'logamt' feature
df['logamt'] = df['amt'].apply(lambda x: np.log(x + 1))

# Step 5: Prepare Features and Target
X = df.drop(columns=['is_fraud'])
y = df['is_fraud']

# Step 6: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Step 8: Train the SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Step 9: Predictions
y_pred = svm_model.predict(X_test)












'''''' RANDOM FOREST'''
'''import pandas as pd
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





STACKINGG




estimator_list = [
    ('svm',svm_model),
    ('rf',rf),
 ]

# Build stack model
stack_model = StackingClassifier(
    estimators=estimator_list, final_estimator=LogisticRegression()
)

# Train stacked model
stack_model.fit(X_train_final, y_train)

# Make predictions
y_train_pred = stack_model.predict(X_train_final)
y_test_pred = stack_model.predict(X_test_final)

# Training set model performance
stack_model_train_accuracy = accuracy_score(y_train, y_pred) # Calculate Accuracy
stack_model_train_precision = precision_score(y_train, y_pred) # Calculate Precision
stack_model_train_f1 = f1_score(y_train, y_pred, average='weighted') # Calculate F1-score

# Test set model performance
stack_model_test_accuracy = accuracy_score(y_test, y_test_pred) # Calculate Accuracy
stack_model_test_precision = precision_score(y_test, y_test_pred) # Calculate MCC
stack_model_test_f1 = f1_score(y_test, y_test_pred, average='weighted') # Calculate F1-score

print('Model performance for Training set')
print('- Accuracy: %s' % stack_model_train_accuracy)
print('- Precision: %s' % stack_model_train_precision)
print('- F1 score: %s' % stack_model_train_f1)
print('----------------------------------')
print('Model performance for Test set')
print('- Accuracy: %s' % stack_model_test_accuracy)
print('- Precision: %s' % stack_model_test_precision)
print('- F1 score: %s' % stack_model_test_f1)



acc_train_list = {'Logistic':log_train_accuracy,
'svm': svm_rbf_train_accuracy,
'rf': rf_train_accuracy,
'stack': stack_model_train_accuracy}

precision_train_list = {'Logistic':log_train_precision,
'svm': svm_rbf_train_precision,
'rf': rf_train_precision,
'stack': stack_model_train_precision}

f1_train_list = {'Logistic':log_train_f1,
'svm': svm_rbf_train_f1,
'rf': rf_train_f1,
'stack': stack_model_train_f1}




acc_df = pd.DataFrame.from_dict(acc_train_list, orient='index', columns=['Accuracy'])
mcc_df = pd.DataFrame.from_dict(mcc_train_list, orient='index', columns=['MCC'])
f1_df = pd.DataFrame.from_dict(f1_train_list, orient='index', columns=['F1'])
df = pd.concat([acc_df, mcc_df, f1_df], axis=1)
df'''
 




'''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

# Load the data
train_df = pd.read_csv('/Users/bharathsreekumarmenon/Downloads/undersampled_numeric_timeconvert.csv')
test_df = pd.read_csv('/Users/bharathsreekumarmenon/Downloads/test_undersampled_numerics.csv')

train_df = train_df.drop(columns=['amt', 'X'], errors='ignore')
test_df = test_df.drop(columns=['amt', 'X'], errors='ignore')

X_train = train_df.drop(columns=['is_fraud'])  # Features for training
y_train = train_df['is_fraud']  # Target (Fraud or not) for training
X_test = test_df.drop(columns=['is_fraud'])  # Features for testing
y_test = test_df['is_fraud']  # Target (Fraud or not) for testing

# Scaling
selected_features = ['logamt', 'city_pop', 'dob_unix', 'trans_unix']  # Example features
X_train_subset = X_train[selected_features]
X_test_subset = X_test[selected_features]

scaler = StandardScaler()
X_train_scaled_subset = scaler.fit_transform(X_train_subset)
X_test_scaled_subset = scaler.transform(X_test_subset)

# Combine scaled selected features with the unscaled features
X_train_unscaled = X_train.drop(columns=selected_features)
X_test_unscaled = X_test.drop(columns=selected_features)

X_train_final = pd.concat([pd.DataFrame(X_train_scaled_subset, columns=selected_features), X_train_unscaled.reset_index(drop=True)], axis=1)
X_test_final = pd.concat([pd.DataFrame(X_test_scaled_subset, columns=selected_features), X_test_unscaled.reset_index(drop=True)], axis=1)

# Create individual models
logistic_model = LogisticRegression(class_weight='balanced', solver='saga', max_iter=500)

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
rf_Grid.fit(X_train_final, y_train)

rf_Grid.best_params_

svm_model = SVC(kernel='linear', probability=True)  # SVM model with probability enabled

# Train the models separately
logistic_model.fit(X_train_final, y_train)
svm_model.fit(X_train_final, y_train)
rf_Model.fit(X_train_final, y_train)

# Stacking model
estimators = [('logistic', logistic_model), ('svm', svm_model), ('rf', rf_Model)]
stack_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=500))

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
    print("-" * 40)

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
print('hello')'''

'''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

# Load the data
train_df = pd.read_csv('/Users/bharathsreekumarmenon/Downloads/undersampled_numeric_timeconvert.csv')
test_df = pd.read_csv('/Users/bharathsreekumarmenon/Downloads/test_undersampled_numerics.csv')

train_df = train_df.drop(columns=['amt', 'X'], errors='ignore')
test_df = test_df.drop(columns=['amt', 'X'], errors='ignore')

X_train = train_df.drop(columns=['is_fraud'])  # Features for training
y_train = train_df['is_fraud']  # Target (Fraud or not) for training
X_test = test_df.drop(columns=['is_fraud'])  # Features for testing
y_test = test_df['is_fraud']  # Target (Fraud or not) for testing

# Scaling all features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create individual models
svm_model = SVC(kernel='linear', probability=True)
rf_model = RandomForestClassifier(class_weight='balanced')

# Stacking model with a different final estimator
final_estimator = RandomForestClassifier(class_weight='balanced')  # You can change this to any classifier

stack_model = StackingClassifier(
    estimators=[('svm', svm_model), ('rf', rf_model)],
    final_estimator=final_estimator
)

# Train stacked model
stack_model.fit(X_train_scaled, y_train)

# Predictions for train and test sets
y_train_pred = stack_model.predict(X_train_scaled)
y_test_pred = stack_model.predict(X_test_scaled)

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
    print("-" * 40)

evaluate_performance(y_train, y_train_pred, dataset_type="Training")
evaluate_performance(y_test, y_test_pred, dataset_type="Test")'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec

# Create a mesh grid for the S-shaped curve
x = np.linspace(-2, 2, 100)  # Log Amount (feature 1)
z = np.linspace(-2, 2, 100)  # Date of Birth in Unix Time (feature 2)
x_grid, z_grid = np.meshgrid(x, z)

# Generate S-shaped probabilities using the logistic function
y = 1 / (1 + np.exp(-x_grid * 3))  # Apply a scaling factor for steepness

# Create random points at y=0 and y=1
num_points = 100

# Data points at y=0 clustered around lower x values
random_x_0 = np.random.uniform(-2, 0, num_points)  # Clustered at low x values
random_z_0 = np.random.uniform(-2, 2, num_points)
random_y_0 = np.zeros(num_points)  # For y=0

# Data points at y=1 clustered around higher x values
random_x_1 = np.random.uniform(0, 2, num_points)  # Clustered at high x values
random_z_1 = np.random.uniform(-2, 2, num_points)
random_y_1 = np.ones(num_points)    # For y=1

# Create a combined figure
fig = plt.figure(figsize=(16, 8))
gs = GridSpec(1, 2, width_ratios=[2, 1])  # 1 row, 2 columns

# 3D plot
ax = fig.add_subplot(gs[0, 0], projection='3d')
ax.plot_surface(x_grid, z_grid, y, cmap='viridis', alpha=0.7, edgecolor='none')

# Plot random points at y=0 and y=1
ax.scatter(random_x_0, random_z_0, random_y_0, color='r', label='Data Points at y=0', s=50)
ax.scatter(random_x_1, random_z_1, random_y_1, color='b', label='Data Points at y=1', s=50)

# Add labels and title
ax.set_title('3D Logistic Regression Curve')
ax.set_xlabel('Log Amount (Feature 1)')
ax.set_ylabel('Date of Birth (Unix Time, Feature 2)')
ax.set_zlabel('Probability of Fraud')
ax.set_ylim(-2, 2)
ax.set_xlim(-2, 2)
ax.set_zlim(0, 1)
plt.legend()

# Heatmap
ax2 = fig.add_subplot(gs[0, 1])
heatmap = ax2.imshow(y, extent=(-2, 2, -2, 2), origin='lower', cmap='viridis', aspect='auto')
ax2.set_title('Heatmap of Probabilities')
ax2.set_xlabel('Log Amount (Feature 1)')
ax2.set_ylabel('Date of Birth (Unix Time, Feature 2)')
plt.colorbar(heatmap, ax=ax2, label='Probability of Fraud')

plt.tight_layout()
plt.show()

