import pandas as pd

data = pd.read_csv("fraudDetection_Datathon/csv files/undersampled_dataset_with_numerics.csv")
y = data['is_fraud']
X = data.drop('is_fraud', axis=1)

print(f'data shape: {data.shape}')
print(f'X shape: {X.shape}')

print(data.head())
print(data.isnull().sum())
