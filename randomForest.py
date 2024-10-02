import pandas as pd
import private.py as pr

data = pd.read_csv(pr.file_path)
y = data['is_fraud']
X = data.drop('is_fraud', axis=1)

print(f'data shape: {data.shape}')
print(f'X shape: {X.shape}')

print(data.head())
print(data.isnull().sum())
