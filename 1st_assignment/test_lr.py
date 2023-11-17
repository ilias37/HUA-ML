from sklearn import datasets
from sklearn import model_selection
import linear_regression    # Το linear_regression.py
import numpy as np
from sklearn import linear_model
from sklearn import metrics

# 3.1
dataset = datasets.fetch_california_housing()   # Δημιουργία dataset

# Χωρισμός του dataset
X_train, X_test, y_train, y_test = model_selection.train_test_split(dataset.data, dataset.target, test_size = 0.3, random_state = 42)

lr = linear_regression.LinearRegression()        # Δημιουργία μοντέλου
lr.fit(X_train, y_train)

y_hat_test, mse_test = lr.evaluate(X_test, y_test)
rmse_test = np.sqrt(mse_test)
print("3.1")
print("RMSE on test set: {}".format(rmse_test) )
print("")

# 3.2
rmse_values = []

for _ in range(20):

    X_train, X_test, y_train, y_test = model_selection.train_test_split(dataset.data, dataset.target, test_size=0.3)
    
    lr = linear_regression.LinearRegression() 
    lr.fit(X_train, y_train)

    y_hat_test, mse_test = lr.evaluate(X_train, y_train) 
    rmse_test = np.sqrt(mse_test)
    rmse_values.append(rmse_test)


rmse_values = np.array(rmse_values)

mean_rmse = np.mean(rmse_values)    # Υπολογισμός μέσου του RMSE
print("3.2")
print("Mean RMSE over {} iterations: {}".format(20, mean_rmse))

std_rmse = np.std(rmse_values)      # Υπολογισμός τυπικής απόκλισης του RMSE
print("Standard Deviation of RMSE over {} iterations: {}".format(20, std_rmse))
print("")

# 3.3
rmse_values = []

for _ in range(20):

    X_train, X_test, y_train, y_test = model_selection.train_test_split(dataset.data, dataset.target, test_size=0.3)
    
    lr = linear_model.LinearRegression()
    lr.fit(X_train, y_train)
    
    y_hat = lr.predict(X_test)
   
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_hat)) # Υπολογισμός του RMSE
    rmse_values.append(rmse)

rmse_values = np.array(rmse_values)

mean_rmse = np.mean(rmse_values)
print("Mean of RMSE: {}".format(mean_rmse))

std_rmse = np.std(rmse_values)
print("Standard Deviation of RMSE: {}".format(std_rmse))