from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
import tensorflow.keras.layers

from tensorflow.keras import Input
from tensorflow.keras import Model
from tcn import TCN

import config


def create_MLP_1(n_layers=3, n_neurons=100, dropout=0.2):
    n_input = config.N_STEPS * config.N_FEATURES_IN
    
    model = Sequential()
    model.add(Dense(n_neurons, activation='relu', input_dim=n_input))
    for i in range (n_layers - 1):
        model.add(Dropout(dropout))
        model.add(Dense(n_neurons, activation='relu'))
    model.add(Dense(config.N_FEATURES_OUT))
    
    return model

def create_LSTM_1(n_layers=1, n_neurons=50, dropout=0.2):
    model = Sequential()
    if n_layers > 1:
        model.add(LSTM(n_neurons, return_sequences=True, activation='relu', input_shape=(config.N_STEPS, config.N_FEATURES_IN)))
        model.add(Dropout(0.2))
    for i in range (n_layers - 2):
        model.add(LSTM(n_neurons, return_sequences=True, activation='relu'))
        model.add(Dropout(0.2))
    model.add(LSTM(n_neurons, activation='relu'))
    model.add(Dense(config.N_FEATURES_OUT))
    
    return model

def create_TCN_1(batch_x=5, batch_y=10, nb_filters=10, kernel_size=2, dilations=[1, 2], dropout_rate=0.1):
    inputs1 = Input(batch_shape=(batch_x, batch_y, config.N_FEATURES_IN))
    outputs1 = TCN(nb_filters=nb_filters, kernel_size=kernel_size, dilations=dilations, dropout_rate=dropout_rate, use_skip_connections=True, padding='causal')(inputs1)
    outputs1 = tensorflow.keras.layers.Dense(config.N_FEATURES_OUT)(outputs1)

    model = Model(inputs=[inputs1], outputs=[outputs1])
                  
    return model