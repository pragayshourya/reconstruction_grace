'''
Various evaluation done for time series forecasting are:
(1) Root Mean Squared Error (RMSE)
(2) Correlation
(3) R2 score (Coefficient of Determination)
(4) NSE (Nash-Sutcliffe Efficiency)
(5) NRMSE (Normalised Root Mean Squared Error)

The above evaluation was done for both training and testing.
'''

#Imports:
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import math

#Defining class:
class evaluation():
    def __init__(self, y, yhat):
        '''
        Initialise the evaluation class with true and predicted values

        Args:
        -   y (numpy.ndarray): Actual values of the Target variable
        -   yhat (numpy.ndarray): Predicted values of the target variable
        '''
        self.y = y
        self.yhat = yhat

    def nse(self, ypred, y):
        '''
        Calculates NSE between actual and predicted target variable.

        Args:
        -   y (numpy.ndarray): Actual values of the Target variable
        -   yhat (numpy.ndarray): Predicted values of the target variable

        Returns:
        -   NSE (float): Nash Sutcliffe Efficiency.
        '''
        return 1-(np.sum((y-ypred)**2)/np.sum((y-np.mean(y))**2))
    
    def correlation(self, y, yhat):
        '''
        Calculates correlation coefficient between actual and predicted target variable.

        Args:
        -   y (numpy.ndarray): Actual values of the Target variable
        -   yhat (numpy.ndarray): Predicted values of the target variable

        Returns:
        -   Correlation (float): Pearson Correlation Coefficient.
        '''
        return np.corrcoef(y, yhat)[1,0]
    
    def r2score(self, y, yhat):
        '''
        Calculates Coefficient of Determination (R2 score) between actual and predicted target variable.

        Args:
        -   y (numpy.ndarray): Actual values of the Target variable
        -   yhat (numpy.ndarray): Predicted values of the target variable

        Returns:
        -   r2_score (float): Coefficient of Determination (R2 score).
        '''
        return r2_score(y, yhat)
    
    def rmse(self, y, yhat):
        '''
        Calculates Root Mean Squared Error between actual and predicted target variable.

        Args:
        -   y (numpy.ndarray): Actual values of the Target variable
        -   yhat (numpy.ndarray): Predicted values of the target variable

        Returns:
        -   rmse (float): Root Mean Squared Error.
        '''
        return math.sqrt(mean_squared_error(y, yhat))
    
    def nrmse_range(self, y, yhat):
        '''
        Calculates Normalised Root Mean Squared Error between actual and predicted target variable.

        Args:
        -   y (numpy.ndarray): Actual values of the Target variable
        -   yhat (numpy.ndarray): Predicted values of the target variable

        Returns:
        -   nrmse_range (float): Normalised Root Mean Squared Error (NRMSE Range).
        '''
        return math.sqrt(mean_squared_error(y, yhat))/(max(y) - min(y))
    
    def return_results(self):
        '''
        Calculate and return a dictionary containing varioaus evaluation metrics.

        Returns:
        -   result_dict (dict): Dictionary containing varioaus evaluation metrics.
        '''
        result_dict = {'Correlation': self.correlation(self.y.reshape(-1), self.yhat.reshape(-1)),
                        'R2 score': self.r2score(self.y.reshape(-1), self.yhat.reshape(-1)),
                        'RMSE': self.rmse(self.y.reshape(-1), self.yhat.reshape(-1)),
                        'NRMSE': self.nrmse_range(self.yhat.reshape(-1), self.y.reshape(-1)),
                        'NSE': self.nse(self.yhat.reshape(-1), self.y.reshape(-1))
                    }
        return result_dict

