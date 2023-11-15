import numpy as np

# 2.0
class LinearRegression:
    
    def __init__(self):
        self.w = None # Βάρος του μοντέλου
        self.b = None # Όρος μεροληψίας του μοντέλου

    # 2.1
    def fit(self, X, y):

        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):    # Έλεγχος ότι τα X και y είναι numpy arrays 
            raise ValueError("X and y should be numpy arrays") 

        if X.shape[0] != y.shape[0]:        # 'Ελεγχος αν οι μεταξύ τους διαστάσεις είναι συμβατές               
            raise ValueError("The number of rows in X and y should be the same")

        ones_column = np.ones((X.shape[0], 1))
        X = np.concatenate( (X, ones_column), axis=1)

        theta = np.linalg.inv( X.T.dot(X) ).dot(X.T).dot(y)    # Υπολογισμός των παραμέτρων θ

        # Αποθήκευση στις ιδιότητες της κλάσης        
        self.w = theta[:-1]
        self.b = theta[-1]

    # 2.2
    def predict(self, X):

        if self.w is None or self.b is None:        # Σφάλμα αν το μοντέλο δεν έχει εκπαιδευτεί
            raise ValueError("Model has not been trained")
        
        return np.dot(X, self.w) + self.b             # y^ = Xw + b

    # 2.3
    def evaluate(self, X, y):

        if self.w is None or self.b is None:        # Σφάλμα αν το μοντέλο δεν έχει εκπαιδευτεί
            raise ValueError("Model has not been trained")

        y_hat = self.predict(X)
        N = X.shape[0]
        mse = np.dot( (y_hat - y).T, (y_hat - y) ) / N   # MSE = 1/N (y^ − y)T (y^ − y) 

        return (y_hat, mse)