'''
This module defines deep learning model. 
It will return a deep learning model which can be directly used by Random Search.
'''

# Imports
import tensorflow as tf
from keras import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, Flatten
from keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')


class deep_learning_model():
    
    def __init__(self, look_back, n_features, max_cnn_layers=3, max_lstm_layers=2, max_dense_layers=3):
        '''
        Initialize the deep learning model with the specified parameters.
        
        Args:
        - look_back (int): Number of time steps to look back in input sequences.
        - n_features (int): Number of input features.
        - max_cnn_layers (int): Maximum number of CNN layers.
        - max_lstm_layers (int): Maximum number of LSTM layers.
        - max_dense_layers (int): Maximum number of Dense layers.
        '''
        self.look_back = look_back
        self.n_features = n_features
        self.max_cnn_layers = max_cnn_layers
        self.max_lstm_layers = max_lstm_layers
        self.max_dense_layers = max_dense_layers

    def get_model_hyp(self, hp):
        '''
        Construct and compile the deep learning model with hyperparameter tuning.
        
        Args:
        - hp (keras_tuner.HyperParameters): Hyperparameters for tuning the model architecture.
        
        Returns:
        - model (keras.Model): Constructed and compiled CNN-LSTM model.
        '''
        model = Sequential()

        # Add CNN layers with hyperparameter tuning
        for cnn_layer in range(hp.Int('num_cnn_layers', min_value=1, max_value=self.max_cnn_layers)):
            model.add(Conv1D(filters=hp.Int(f'cnn_units_{cnn_layer}',
                                             min_value=32,
                                             max_value=256,
                                             step=32),
                             kernel_size=1,
                             activation='relu',
                             input_shape=(self.look_back, self.n_features)))
            model.add(MaxPooling1D(pool_size=1))
        
        model.add(BatchNormalization())
        
        # Add LSTM layers with hyperparameter tuning
        for lstm_layer in range(hp.Int('num_lstm_layers', min_value=1, max_value=self.max_lstm_layers)):
            model.add(LSTM(hp.Int(f'lstm_units_{lstm_layer}',
                                  min_value=32,
                                  max_value=128,
                                  step=32),
                           return_sequences=True if lstm_layer < self.max_lstm_layers - 1 else False))
        
        # Add Dropout layers with hyperparameter tuning
        for dropout_layer in range(hp.Int('num_dropout_layers', min_value=1, max_value=3)):
            model.add(Dropout(hp.Float(f'dropout_rate_{dropout_layer}',
                                       min_value=0.1,
                                       max_value=0.3,
                                       step=0.1)))
        
        # Add Dense layers with hyperparameter tuning
        for dense_layer in range(hp.Int('num_dense_layers', min_value=1, max_value=self.max_dense_layers)):
            model.add(Dense(hp.Int(f'dense_units_{dense_layer}',
                                   min_value=32,
                                   max_value=512,
                                   step=32)))
        
        model.add(Flatten())
        # Add Output layer
        model.add(Dense(1))
        
        # Compile the model with tuned learning rate
        model.compile(optimizer=Adam(
            learning_rate=hp.Float('lr', min_value=1e-4, max_value=1e-2, step=1e-1)),
            loss='mae',
            metrics=['mae'])
        
        return model