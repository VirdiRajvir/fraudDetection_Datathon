import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("fraudDetection_Datathon/csv files/undersampled_numeric_timeconvert.csv")
y = data['is_fraud']
X = data.drop('is_fraud', axis=1)

data_test = pd.read_csv("fraudDetection_Datathon/csv files/test_undersampled_numerics.csv")
y_test = data_test['is_fraud']
X_test = data_test.drop('is_fraud', axis=1)

rf_Model = RandomForestClassifier()
rf_Model.fit(X, y)


print(X.shape)
print(f'Train Accuracy: {rf_Model.score(X, y):.3f}')
print(f'Test Accuracy: {rf_Model.score(X_test, y_test):.3f}')

# print(data.head())
