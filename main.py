# imports relevant libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.metrics import r2_score

# importing Data set
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data (only if needed)
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
# X = np.array(ct.fit_transform(X))

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling (if needed)
sc_X = StandardScaler()
sc_y = StandardScaler()
standard_X = sc_X.fit_transform(X_train)
standard_y = sc_y.fit_transform(y_train.reshape(len(y_train), 1))
standard_testX = sc_X.transform(X_test)
standard_testY = sc_y.transform(y_test.reshape(len(y_test), 1))

# Training the Regression models on the Training set

# 1. Simple/Multiple Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
# 2. Polynomial Linear Regression
poly_reg = PolynomialFeatures(degree=4)
poly_reg_test = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X_train)
polynomial_model = LinearRegression()
polynomial_model.fit(X_poly, y_train)
# 3. SVR  Regression
svr_model = SVR(kernel='rbf')
svr_model.fit(standard_X, standard_y)
# 4. Decision Tree Regression
decisionTree_model = DecisionTreeRegressor(random_state=0)
decisionTree_model.fit(X_train, y_train)
# 5. Random Forest Regression
randomForest_model = RandomForestRegressor(n_estimators=10, random_state=0)
randomForest_model.fit(X_train, y_train)

# Predicting the Test set results
# 1. Simple/Multiple Linear Regression
linear_pred = linear_model.predict(X_test)
# 2. Polynomial Linear Regression
poly_pred = polynomial_model.predict(poly_reg.transform(X_test))
# 3. SVR  Regression
svr_pred = sc_y.inverse_transform(svr_model.predict(standard_testX).reshape(-1, 1))
# 4. Decision Tree Regression
dtr_pred = decisionTree_model.predict(X_test)
# 5. Random Forest Regression
rfr_pred = randomForest_model.predict(X_test)

# Evaluating the Model Performance
# 1. Simple/Multiple Linear Regression
linear_score = r2_score(y_test, linear_pred)
# 2. Polynomial Linear Regression
poly_score = r2_score(y_test, poly_pred)
# 3. SVR  Regression
svr_score = r2_score(y_test, svr_pred)
# 4. Decision Tree Regression
dtr_score = r2_score(y_test, dtr_pred)
# 5. Random Forest Regression
rfr_score = r2_score(y_test, rfr_pred)

results = {
    "Simple/Multiple Linear Regression": linear_score,
    "Polynomial Linear Regression": poly_score,
    "SVR  Regression": svr_score,
    "Decision Tree Regression": dtr_score,
    "Random Forest Regression": rfr_score
}

results = sorted(results.items(), key=lambda x: x[1], reverse=True)

for i in range(0, len(results)):
    print(f'The model in the {i+1} place is {results[i][0]} and the R^2 score is {results[i][1]}\n')
