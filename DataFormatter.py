'''
Data Preparation for Deep Learning Training.
The following class handles the preprocessing of datasets into suitable 3D format for deep learning training.

1. convert_dataframe: Converts the input dataframe into features (x) and target variable (y), as well prepares
a variable x_pred which is the missing months observation for GRACE-TWS. 

2. split_data: Splits the features (x) and target variable (y) into training, validation, and prediction input datasets (x_pred).

'''


#Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



class DataFormatter:
    def __init__(self, df, look_back=1, remove_nan=True, shuffle=True, return_data=False):
        '''
        Initializes the DataFormatter class attributes.

        Parameters:
        - df: str
            The path to the input dataframe.

        - look_back: int
            Number of time steps to look back for creating sequences.

        - remove_nan: bool
            Whether to remove NaN values from the dataset.

        - shuffle: bool
            Whether to shuffle the data.

        - return_data: bool
            Whether to return processes data.       
        '''
        self.df = pd.read_csv(df)  # Load dataset
        self.look_back = look_back  # Number of time steps to look back
        self.shuffle = shuffle  # Whether to shuffle the data
        self.remove_nan = remove_nan  # Whether to remove NaN values
        self.return_data = return_data  # Whether to return processed data
        self.df.set_index('Date', inplace = True, drop = True)
    def convert_dataframe(self, return_data=True):
        '''
        Convert input dataframe into desirable 3D format for deep learning training.

        Parameters:
        - return_data: bool
            Whether to return processed data.

        Returns:
        If return_data is True, returns processed data 

        '''
        look_back = self.look_back
        col_lst = list(self.df.columns)
        x = np.zeros((1, look_back, len(col_lst) - 1))
        
        for i in range(0, self.df.shape[0] - look_back):
            # Reshape and convert dataframe to 3D format for deep learning
            x = np.append(x, np.array(self.df.iloc[i:i + look_back, 0:(len(col_lst) - 1)]).reshape(1, look_back, (len(col_lst) - 1)), axis=0)
        
        y = np.zeros((1, 1))
        
        for i in range(look_back, self.df.shape[0]):
            # Prepare target variable (y) for training
            y = np.append(y, np.array(self.df.iloc[i, len(col_lst) - 1]).reshape(1, 1), axis=0)
        
        x_pred = np.array([x[i] for i in np.where(np.isnan(y.reshape(-1)))]).squeeze()
        
        if self.remove_nan:
            # Remove rows with NaN values from the data
            x = np.delete(x, np.where(np.isnan(y)), axis=0)
            y = np.delete(y, np.where(np.isnan(y)), axis=0)
        
        self.x = x
        self.y = y
        
        if return_data:
            # Return the processed data if specified
            return x, y, x_pred
        return None

    def split_data(self, validation_split=0.1, show_split=False, random_state=None):
        """
        Splits the features (x) and target variable (y) into training and validation sets.

        Parameters:
        - validation_split: float
            Proportion of data to keep in the validation set.
        - show_split: bool
            Whether to display the shapes of the split datasets.
        - random_state: int
            Random seed for reproducibility.

        Returns:
        Returns the split datasets (x_train, x_val, y_train, y_val).
        """
        x_train, x_val, y_train, y_val = train_test_split(self.x, self.y, test_size=validation_split, shuffle=self.shuffle, random_state=random_state)
        
        if show_split:
            # Display shapes of the split datasets
            print("Shapes of the splits are as follows:", x_train.shape, y_train.shape, x_val.shape, y_val.shape, sep='\n')
        
        return x_train, x_val, y_train, y_val
