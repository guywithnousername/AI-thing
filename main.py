import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing

loan_file_path = "loan.csv"
loan_data = pd.read_csv(loan_file_path)
# loan_data['gender'] = loan_data['gender'].astype(str)
# loan_data['occupation'] = loan_data['occupation'].astype(str)
# loan_data['education_level'] = loan_data['education_level'].astype(str)
# loan_data['marital_status'] = loan_data['marital_status'].astype(str)

y = loan_data.loan_status

features = ['age', 'gender', 'occupation', 'education_level', 'marital_status', 'income', 'credit_score']
X = loan_data[features]

le = preprocessing.LabelEncoder()
for i in features:
    loan_data[i] = le.fit_transform(loan_data[i])

loan_model = DecisionTreeRegressor(random_state = 1)

print("Starting to train model")
loan_model.fit(X, y)
for a in X:
    print("Predicting: ")
    print(a)
    print(loan_model.predict(a))