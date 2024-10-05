# Fraud Detection
---
## How to work with the model:
### For the Random Forest Model:
1. Download your test dataset and place it in /csv files 
2. Call this folder in the data_test variable
3. Remove the isFraud, Merchant and amt features (if applicable)
4. Run the code
5. Obtain the accuracy, precision, recall, f1-scores and AUC-ROC.

- If you have just a single test case you want to obtain the prediction for, use the randomForest_singleTest.py file

## Pre-Processing

### Fields in the dataset

- Date of Birth

- Amount

- Gender

- Transaction Time

- Category of purchase

- Merchant name
  
- City and State of transaction

- City population
### Fields removed (Low significance to model):

- Credit card number

- First name

- Last name

- Street

- Zip code

- Latitude

- Longitude

- Unix time

- Merchant latitude

- Merchant longitude

- Transaction number


### Things done:

Useless columns removed (eg. Credit card number)


Under-sampling done to make the dataset balanced.



## Modelling

Chosen models to compare:

Random Tree - Rajvir

SVM - Shaurya

Logistic Regression and Stacked – Bharath

## Evaluation

Confusion matrix: -- effectiveness

ROC curve: -- effectiveness
