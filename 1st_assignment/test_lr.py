from sklearn import datasets
from sklearn import model_selection
import linear_regression
import numpy as np
from sklearn import linear_model
from sklearn import metrics

# 3.1
dataset = datasets.fetch_california_housing()   # Δημιουργία dataset

# Χωρισμός του dataset
X_train, X_test, y_train, y_test = model_selection.train_test_split(dataset.data, dataset.target, test_size = 0.3, random_state = 42)

lr = linear_regression.LinearRegression()        # Δημιουργία μοντέλου
lr.fit(X_train, y_train)

y_hat_train, mse_train = lr.evaluate(X_train, y_train) 
rmse_train = np.sqrt(mse_train)
print("Root Mean Squared Error (RMSE) on training set: {:.10f}".format(rmse_train) )

y_hat_test, mse_test = lr.evaluate(X_test, y_test)
rmse_test = np.sqrt(mse_test)
print("Root Mean Squared Error (RMSE) on test set: {:.10f}".format(rmse_test) )
print("")

# 3.2
rmse_values = []

for _ in range(20):

    X_train, X_test, y_train, y_test = model_selection.train_test_split(dataset.data, dataset.target, test_size=0.3, random_state=42)
    
    lr = linear_regression.LinearRegression() 
    lr.fit(X_train, y_train)

    y_hat_test, mse_test = lr.evaluate(X_train, y_train) 
    rmse_test = np.sqrt(mse_test)
    rmse_values.append(rmse_test)


rmse_values = np.array(rmse_values)

mean_rmse = np.mean(rmse_values)    # Υπολογισμός μέσου του RMSE
print("Mean RMSE over {} iterations: {}".format(20, mean_rmse))

std_rmse = np.std(rmse_values)      # Υπολογισμός τυπικής απόκλισης του RMSE
print("Standard Deviation of RMSE over {} iterations: {}".format(20, std_rmse))
print("")

# 3.3
rmse_values = []

for _ in range(20):

    X_train, X_test, y_train, y_test = model_selection.train_test_split(dataset.data, dataset.target, test_size=0.3, random_state=42)
    
    lr = linear_model.LinearRegression()
    lr.fit(X_train, y_train)
    
    y_hat = lr.predict(X_test)
   
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_hat)) # Υπολογισμός του RMSE
    rmse_values.append(rmse)


mean_rmse = np.mean(rmse_values)
print("Mean RMSE: {}".format(mean_rmse))

std_rmse = np.std(rmse_values)
print("RMSE Variance: {}".format(std_rmse))