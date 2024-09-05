import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor

loan_file_path = "loan.csv"
loan_data = pd.read_csv(loan_file_path)
y = loan_data.loan_status

features = ['age', 'gender', 'occupation', 'education_level', 'marital_status', 'income', 'credit_score']
X = loan_data[features]

loan_model = DecisionTreeRegressor(random_state = 1)

loan_model.fit(X, y)