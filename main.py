'''
Multiprocessing for training various grids.
'''

#Imports:
import os 
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='1,2,3,4,5,6,7'
import pandas as pd
import numpy as np
import eval as ev
import tensorflow as tf
import DataFormatter as prep
from sklearn.preprocessing import MinMaxScaler
import deep_learning_model as dl_model
from glob import glob
from tqdm import tqdm
import multiprocessing as mp

import time
from math import isnan
import warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

if os.path.exists('./TIME SERIES') is not True:
    os.makedirs('./TIME SERIES')
if os.path.exists('./output') is not True:
    os.makedirs('./output')
if os.path.exists('./Train') is not True:
    os.makedirs('./Train')
if os.path.exists('./Test') is not True:
    os.makedirs('./Test')
def training_cnn_lstm(file):


    
    if os.path.exists('./TIME SERIES/{}.csv'.format(file.split('/')[-1][:-4])):
      return None
    
    import keras_tuner as kt
    import tensorflow as tf
    from keras import backend as K
    from keras.callbacks import EarlyStopping
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)
    look_back = 3

    df = pd.read_csv(file)
    df.set_index('Date', inplace = True, drop = True)
    if df.isna().all().any():
        return
    preprocess = prep.DataFormatter(file, look_back = look_back, remove_nan = True, shuffle = True, return_data = True)
    x, y, X_pred = preprocess.convert_dataframe()
    X_train, X_val, y_train , y_val = preprocess.split_data(show_split = False, validation_split=0.2)
    temp_tr, temp_val = y_train, y_val
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_pred = scaler.transform(X_pred.reshape(-1, X_pred.shape[-1])).reshape(X_pred.shape)
    y_train = scaler.fit_transform(y_train)
    y_val = scaler.transform(y_val)
    
    model = dl_model.deep_learning_model(look_back = X_train.shape[1], n_features = X_train.shape[2])
    
    tuner = kt.RandomSearch(
            model.get_model_hyp,
            objective='val_loss',
            max_trials=1,
            overwrite = True,
            directory = './Models',
            project_name='{}'.format(file.split('/')[-1][:-4]))
    tuner.search(X_train, y_train, epochs=500,
                validation_data=(X_val, y_val),
                use_multiprocessing = False,
                workers=-1, verbose = 1,
                callbacks=[EarlyStopping(monitor='val_loss', patience=15)]
                )
    best_model = tuner.get_best_models()[0]
    predicted_val = scaler.inverse_transform(best_model.predict(X_val, verbose=0))
    predicted_tr = scaler.inverse_transform(best_model.predict(X_train, verbose=0))
    y_val = scaler.inverse_transform(y_val)
    y_train = scaler.inverse_transform(y_train)
    multi_index = pd.MultiIndex.from_tuples([(file.split('/')[-1][:-4].split('_')[1], file.split('/')[-1][:-4].split('_')[0])], names = ['lon', 'lat'])
    tr_df = pd.DataFrame(ev.evaluation(y_train, predicted_tr).return_results(), index=multi_index)
    te_df = pd.DataFrame(ev.evaluation(y_val, predicted_val).return_results(), index=multi_index)
    tr_df.to_csv('./Train/{}.csv'.format(file.split('/')[-1][:-4]))
    te_df.to_csv('./Test/{}.csv'.format(file.split('/')[-1][:-4]))
    y_pred = scaler.inverse_transform(best_model.predict(X_pred, verbose=0))
    act = np.concatenate((temp_tr, temp_val))
    pred = np.concatenate((predicted_tr, predicted_val))

    # Create a DataFrame with 'GRACE-Actual' and 'GRACE-Predicted' columns
    temp = pd.DataFrame({'GRACE-Actual': act.reshape(-1), 'GRACE-Predicted': pred.reshape(-1)})

    # Merge 'temp' with 'df' to get matching dates
    merged_temp = temp.merge(df.reset_index(), left_on='GRACE-Actual', right_on=df['GRACE'], how='inner')
    merged_temp = merged_temp[['Date', 'GRACE-Actual', 'GRACE-Predicted']]
    merged_temp.set_index('Date', inplace = True)
    merged_temp.sort_index(inplace=True)
    merged_temp.index = pd.to_datetime(merged_temp.index)
    # Create the DataFrame with index ranging from start to end dates
    recon_grace = pd.DataFrame(index=pd.date_range(start='2002-04-01', end='2022-07-01', freq='MS'))
    recon_grace.index.name = 'Date'

    # Merge 'recon_grace' with 'merged_temp' using left join
    recon_grace = recon_grace.merge(merged_temp, left_index=True, right_on='Date', how='left')
    recon_grace.set_index('Date', inplace=True)

    # Identify NaN indices in 'GRACE-Predicted'
    nan_indices = recon_grace.index[recon_grace['GRACE-Predicted'].isnull()]

    # Fill NaN values in 'GRACE-Predicted' with values from 'y_pred'
    recon_grace.loc[nan_indices, 'GRACE-Predicted'] = y_pred

    recon_grace.to_csv('./TIME SERIES/{}.csv'.format(file.split('/')[-1][:-4]))
    #return ev.evaluation(y_val, predicted_val).return_results(), ev.evaluation(y_train, predicted_tr).return_results()



if __name__ == '__main__':
    os.system('clear')
    pool = mp.Pool(mp.cpu_count())
    files = glob(os.path.join('./data/COMPILED/*.csv'))
    pool.map(training_cnn_lstm, files[:1])
    os.system('clear')
