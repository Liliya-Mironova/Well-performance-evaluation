import numpy as np
import pandas as pd
import time
import datetime
import json
import pickle
import os

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

from keras.models import Sequential
from tensorflow.keras.models import load_model, model_from_json
from keras.layers import Dense, LSTM, Dropout
import tensorflow.keras.optimizers
from keras import optimizers
from keras.utils.vis_utils import plot_model
# from keras.callbacks import LearningRateScheduler

# TCN
from tensorflow.keras import Input
from tensorflow.keras import Model
from tcn import TCN

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

from utils import *
from models import *

class Model():
    def __init__(self, model_name, study_path, real_file, train_all_ratio=1/3, n_steps=5, 
                 n_features_in=6, n_features_out=3):
        
        self.study_path = study_path
        self.real_file = real_file
        self.train_all_ratio = train_all_ratio
        self.n_steps = n_steps
        self.n_features_in = n_features_in
        self.n_features_out = n_features_out
        self.n_input = n_steps * n_features_in
        self.model_name = model_name
    
    def print_info(self):
        print (f'train/all: {self.train_all_ratio}')
        print (f'n_steps: {self.n_steps}')
        print (f'n_features_in: {self.n_features_in}, n_features_out: {self.n_features_out}')
        print (f'loss metric: {self.loss}')
        print (f'layers: {self.n_layers}, neurons: {self.n_neurons}, droupout: {self.dropout}')

    def full_launch(self, n_epochs=2, tuning=True, plot_losses=True, plot_preds=True, plot_errs=False):
        self.model.train(n_epochs=n_epochs, tuning=tuning)
        self.model.predict(plot_losses=plot_losses, plot_preds=plot_preds, plot_errs=plot_errs)
        self.model.save()
        
    def prepare_data(self, real_file, ):
        df = pd.read_csv(f"{self.real_file}")
        out_names = ['BORE_OIL_VOL', 'BORE_GAS_VOL', 'BORE_WAT_VOL']
        outputs = df[out_names]
        inputs = df.drop(columns=out_names, axis=1)
        
        self.all_len = len(inputs)
        self.train_len = int(np.floor(len(outputs) * self.train_all_ratio))
        self.val_len = (self.all_len - self.train_len) // 2
        
        # split a multivariate sequence into samples 
        raw_seq1 = np.hstack((inputs[:self.train_len], outputs[:self.train_len])) # ins and outs - train
        val = np.hstack((inputs[self.train_len:self.train_len + self.val_len], 
                         outputs[self.train_len:self.train_len + self.val_len]))  # ins and outs - val
        self.true_data = np.hstack((inputs, outputs))                             # ins and outs - all

        # normalize the data
        self.scaler = MinMaxScaler()
        self.raw_seq1_norm = self.scaler.fit_transform(raw_seq1) # norm all
        val_norm = self.scaler.transform(val)
        self.true_data_norm = self.scaler.transform(self.true_data)

        self.X, self.y = split_sequence(self.raw_seq1_norm, self.n_steps, self.n_features_in, self.n_features_out)
        self.X_val, self.y_val = split_sequence(val_norm, self.n_steps, self.n_features_in, self.n_features_out)
        self.X_all, _ = split_sequence(self.true_data_norm, self.n_steps, self.n_features_in, self.n_features_out)
        
        if 'MLP' in self.model_name or 'XGB' in self.model_name:
            # flatten input
            self.X_reshaped = self.X.reshape((self.X.shape[0], self.n_input))
            self.X_val_reshaped = self.X_val.reshape((self.X_val.shape[0], self.n_input))
            self.X_all_reshaped = self.X_all.reshape((self.X_all.shape[0], self.n_input))
        elif 'LSTM' in self.model_name or 'TCN' in self.model_name:
            self.X_reshaped = self.X
            self.X_val_reshaped = self.X_val
            self.X_all_reshaped = self.X_all
            
    def train(self, n_epochs=50, tuning=True, verbose=1, batch_size=32):
        self.n_epochs = n_epochs
        self.tuning = tuning
        
        if 'XGB' not in self.model_name:
            if self.prefix != '-init':
                for layer in self.model.layers[:-1]:
                    layer.trainable = self.tuning
            self.model.compile(loss=self.loss, optimizer=self.optimizer)
        
        # train the model
        start_time = time.time()

        if 'TCN' in self.model_name:
            shape_train = self.X_reshaped.shape[0] - self.X_reshaped.shape[0] % self.batch_x
            shape_test = self.X_val_reshaped.shape[0] - self.X_val_reshaped.shape[0] % self.batch_x
            self.model_inf = self.model.fit(self.X_reshaped[:shape_train], self.y[:shape_train], 
                                       epochs=self.n_epochs, verbose=verbose, 
                                       validation_data=(self.X_val_reshaped[:shape_test], self.y_val[:shape_test]))
        elif 'XGB' in self.model_name:
            self.model.fit(self.X_reshaped, self.y)
        else:
            if self.X_val_reshaped.shape[0] > 0:
                self.model_inf = self.model.fit(self.X_reshaped, self.y, epochs=self.n_epochs, 
                                                verbose=verbose, batch_size=batch_size, 
                                                validation_data=(self.X_val_reshaped, self.y_val))
            else:
                self.model_inf = self.model.fit(self.X_reshaped, self.y, epochs=self.n_epochs, 
                                                verbose=verbose, batch_size=batch_size)  

        self.delta_time = time.time() - start_time
        self.cur_time = datetime.datetime.now()
            
    def predict(self, plot_losses=True, plot_preds=True, plot_errs=False, print_res=False):
        # obtain the model results: make a prediction and calculate an error

        y_pred_norm = self.model.predict(self.X_all_reshaped)
        table_norm = np.hstack((self.true_data_norm[:, :-self.n_features_out], 
                                np.vstack((self.raw_seq1_norm[:self.n_steps, -self.n_features_out:], y_pred_norm))))
        table = self.scaler.inverse_transform(table_norm) # denormalize all
        preds = table[:, -self.n_features_out:]
        errs = error(preds, self.true_data[:, -self.n_features_out:]) # calc error between denormalized y_pred and real y

        data = {}
        data["model_name"] = self.model_name
        if 'XGB' not in self.model_name:
            data["loss"] = self.model_inf.history['loss']
            if self.X_val_reshaped.shape[0] > 0:
                data["val_loss"] = self.model_inf.history['val_loss']
                data["val_size"] = self.train_len + self.val_len
        data["train_size"] = self.train_len
        data["preds"] = preds.tolist()
        data["g_truth"] = self.true_data[:, -self.n_features_out:].tolist()
        data["errors"] = errs.tolist()
        data["training_time"] = self.delta_time
        data["cur_time"] = str(self.cur_time)

        if plot_losses:
            if 'XGB' not in self.model_name:
                if self.X_val_reshaped.shape[0] > 0:
                    plot_train_val_loss(data['loss'], data['val_loss'], model_name=self.model_name)
                else:
                    plot_loss(data['loss'], full_screen=True, model_name=self.model_name)
        if print_res:
            print_results(data)
        if plot_preds:
            plot_pred(data)
        if plot_errs:
            plot_err(data)
        
        print ('here')
        if 'XGB' not in self.model_name:
            return np.mean(data["errors"][data["val_size"]:]), float(str(data["training_time"])[:4])
        else:
            return np.mean(data["errors"][data["train_size"]:]), float(str(data["training_time"])[:4])
            
    def save(self, prefix):
        if 'XGB' in self.model_name:
            pickle.dump(self.model, open(f'{self.study_path}/{self.model_name}{prefix}.hdf', "wb"))
        elif 'TCN' in self.model_name:
            # get model as json string and save to file
            model_as_json = self.model.to_json()
            with open(f'{self.study_path}/{self.model_name}{prefix}.json', "w") as json_file:
                json_file.write(model_as_json)

            # save weights to file (for this format, need h5py installed)
            self.model.save_weights(f'{self.study_path}/{self.model_name}{prefix}.h5')
        else:
            self.model.save(f'{self.study_path}/{self.model_name}{prefix}', save_format='h5')

        print (f'{self.study_path}/{self.model_name}{prefix}.hdf saved')

class Init(Model):
    prefix = '-init'
    
    def __init__(self, model_name, study_path, real_file, train_all_ratio=1/3, n_steps=5, 
                 n_features_in=6, n_features_out=3, loss='mae', lr=1e-3, n_layers=3, 
                 n_neurons=50, dropout=0.2, max_depth=5, reg_alpha=1):
        
        super().__init__(model_name, study_path, real_file, train_all_ratio=train_all_ratio, 
                 n_steps=n_steps, n_features_in=n_features_in, n_features_out=n_features_out)
        
        self.loss = loss
        self.lr = lr
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.dropout = dropout
        self.max_depth = max_depth
        self.reg_alpha = reg_alpha
        
        self.prepare_data()
        self.create_model()
        
    def print_info(self):
        super().print_info()
        
    def prepare_data(self):
        super().prepare_data()
            
    def create_model(self):
        # load model
        model_path = f'{self.study_path}/{self.model_name}.hdf'

        if 'MLP' in self.model_name:
            self.model = create_MLP_1(n_layers=self.n_layers, n_neurons=self.n_neurons, dropout=self.dropout)

        elif 'LSTM' in self.model_name:
            self.model = create_LSTM_1(n_layers=self.n_layers, n_neurons=self.n_neurons, dropout=self.dropout)
            
        elif 'TCN' in self.model_name:
            self.batch_x = 16
            self.model = create_TCN_1(batch_x=self.batch_x, batch_y=self.X_reshaped.shape[1], 
                                 nb_filters=50, kernel_size=5, dilations=[1, 2], dropout_rate=self.dropout)

        elif 'XGB' in self.model_name:
            self.model = MultiOutputRegressor(XGBRegressor(n_estimators=self.n_neurons, 
                                                           max_depth=self.max_depth, reg_alpha=self.reg_alpha))
        
        if 'XGB' not in self.model_name:
            self.optimizer = tensorflow.keras.optimizers.Adam(lr=self.lr)

    def train(self, n_epochs=50, tuning=True, verbose=1, batch_size=32):
        super().train(n_epochs=n_epochs, tuning=tuning, verbose=verbose, batch_size=batch_size)
        
    def predict(self, plot_losses=True, plot_preds=True, plot_errs=False, print_res=False):
        super().predict(plot_losses=plot_losses, plot_preds=plot_preds, plot_errs=plot_errs, print_res=print_res)
        
    def save(self):
        super.save(self.prefix)
        
#     def full_launch(n_epochs=2, tuning=True, plot_losses=True, plot_preds=True, plot_errs=False):
#         self.model.train(n_epochs=n_epochs, tuning=tuning)
#         self.model.predict(plot_losses=plot_losses, plot_preds=plot_preds, plot_errs=plot_errs)
#         self.model.save()
        

class TL(Model):
    prefix = '-transf'
    
    def __init__(self, model_name, study_path, real_file, train_all_ratio=1/3, n_steps=5, 
                 n_features_in=6, n_features_out=3, loss='mae', lr=1e-3):
        
        super().__init__(model_name, study_path, real_file, train_all_ratio=train_all_ratio, 
                 n_steps=n_steps, n_features_in=n_features_in, n_features_out=n_features_out)
        
        self.loss = loss
        self.lr = lr
        
        self.prepare_data()
        self.create_model()
        
    def prepare_data(self):
        super().prepare_data()
            
    def create_model(self):
        # load model
        model_path = f'{self.study_path}/{self.model_name}.hdf'

        if 'MLP' in self.model_name or 'LSTM' in self.model_name:
            self.model = load_model(model_path)

        elif 'TCN' in self.model_name:
            self.batch_x = 17 # 30
            loaded_json = open(f'{model_path[:-4]}.json', 'r').read()
            self.model = model_from_json(loaded_json, custom_objects={'TCN': TCN})

        elif 'XGB' in self.model_name:
            self.model = pickle.load(open(model_path, 'rb'))
            
        if 'XGB' not in self.model_name:
            self.optimizer = tensorflow.keras.optimizers.Adam(lr=self.lr)
            
    def train(self, n_epochs=50, tuning=True, verbose=1, batch_size=32): # tuning=True!
        self.n_epochs = n_epochs
        self.tuning = tuning
        
        if 'XGB' not in self.model_name:
            if self.prefix != '-init':
                for layer in self.model.layers[:-1]:
                    layer.trainable = self.tuning
            self.model.compile(loss=self.loss, optimizer=self.optimizer)
        
        # train the model
        start_time = time.time()

        if 'TCN' in self.model_name:
            shape_train = self.X_reshaped.shape[0] - self.X_reshaped.shape[0] % self.batch_x
            shape_test = self.X_val_reshaped.shape[0] - self.X_val_reshaped.shape[0] % self.batch_x
            self.model_inf = self.model.fit(self.X_reshaped[:shape_train], self.y[:shape_train], 
                                       epochs=self.n_epochs, verbose=verbose, 
                                       validation_data=(self.X_val_reshaped[:shape_test], self.y_val[:shape_test]))
        elif 'XGB' in self.model_name:
            self.model.fit(self.X_reshaped, self.y)
        else:
            if self.X_val_reshaped.shape[0] > 0:
                self.model_inf = self.model.fit(self.X_reshaped, self.y, epochs=self.n_epochs, 
                                                verbose=verbose, batch_size=batch_size, 
                                                validation_data=(self.X_val_reshaped, self.y_val))
            else:
                self.model_inf = self.model.fit(self.X_reshaped, self.y, epochs=self.n_epochs, 
                                                verbose=verbose, batch_size=batch_size)  

        self.delta_time = time.time() - start_time
        self.cur_time = datetime.datetime.now()
        
    def predict(self, plot_losses=True, plot_preds=True, plot_errs=False, print_res=False):
        # obtain the model results: make a prediction and calculate an error

        y_pred_norm = self.model.predict(self.X_all_reshaped)
        table_norm = np.hstack((self.true_data_norm[:, :-self.n_features_out], 
                                np.vstack((self.raw_seq1_norm[:self.n_steps, -self.n_features_out:], y_pred_norm))))
        table = self.scaler.inverse_transform(table_norm) # denormalize all
        preds = table[:, -self.n_features_out:]
        errs = error(preds, self.true_data[:, -self.n_features_out:]) # calc error between denormalized y_pred and real y

        data = {}
        data["model_name"] = self.model_name
        if 'XGB' not in self.model_name:
            data["loss"] = self.model_inf.history['loss']
            if self.X_val_reshaped.shape[0] > 0:
                data["val_loss"] = self.model_inf.history['val_loss']
                data["val_size"] = self.train_len + self.val_len
        data["train_size"] = self.train_len
        data["preds"] = preds.tolist()
        data["g_truth"] = self.true_data[:, -self.n_features_out:].tolist()
        data["errors"] = errs.tolist()
        data["training_time"] = self.delta_time
        data["cur_time"] = str(self.cur_time)

        if plot_losses:
            if 'XGB' not in self.model_name:
                if self.X_val_reshaped.shape[0] > 0:
                    plot_train_val_loss(data['loss'], data['val_loss'], model_name=self.model_name)
                else:
                    plot_loss(data['loss'], full_screen=True, model_name=self.model_name)
        if print_res:
            print_results(data)
        if plot_preds:
            plot_pred(data)
        if plot_errs:
            plot_err(data)
            
        if 'XGB' not in self.model_name:
            return np.mean(data["errors"][data["val_size"]:]), float(str(data["training_time"])[:4])
        else:
            return np.mean(data["errors"][data["train_size"]:]), float(str(data["training_time"])[:4])
        
    def save(self):
        self.prefix = '-tuning' if self.tuning else '-transf'
        super().save(self.prefix)
        
    def full_launch(n_epochs=2, tuning=False, plot_losses=True, plot_preds=True, plot_errs=False):
        self.model.train(n_epochs=n_epochs, tuning=tuning)
        self.model.predict(plot_losses=plot_losses, plot_preds=plot_preds, plot_errs=plot_errs)
        self.model.save()
        

class Gen(Model):
    prefix = ''
    
    model = None
    optimizer = None
    
    study_path = ""
    model_name = None
    loss = None
    lr = None
    n_epochs = None
    train_all_ratio = None
    
    n_layers = None
    n_neurons = None
    dropout = None
    n_input = None
    
    delta_time, cur_time = None, None
    
    # new
    gen_dfs_path = None
    inputss, outputss = [], []
    X_train, y_train = None, None
    X_valid, y_valid = None, None
    X_all_reshaped_s, raw_seq1_norm_s, true_data_norm_s, true_data_s = [], [], [], []
    scalers = []
    train_lens = []
    
    def __init__(self, model_name, study_path, real_file, gen_dfs_path, tpl_path, train_all_ratio=1/3, n_steps=5, 
                 n_features_in=6, n_features_out=3, loss='mae', lr=1e-3, n_layers=3, 
                 n_neurons=50, dropout=0.2, max_depth=5, reg_alpha=1):
        
        self.study_path = study_path
        self.real_file = real_file
        self.gen_dfs_path = gen_dfs_path
        self.train_all_ratio = train_all_ratio
        self.n_steps = n_steps
        self.n_features_in = n_features_in
        self.n_features_out = n_features_out
        self.n_input = n_steps * n_features_in
        self.tpl_path = tpl_path
            
        self.model_name = model_name
        self.loss = loss
        self.lr = lr
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.dropout = dropout
        
        self.max_depth = max_depth
        self.reg_alpha = reg_alpha
        
        self.prepare_data()
        self.create_model()
        
    def print_info(self):
        print (f'train/all: {self.train_all_ratio}')
        print (f'n_steps: {self.n_steps}')
        print (f'n_features_in: {self.n_features_in}, n_features_out: {self.n_features_out}')
        print (f'loss metric: {self.loss}')
        print (f'layers: {self.n_layers}, neurons: {self.n_neurons}, droupout: {self.dropout}')
        
    def prepare_data(self):
        # load datasets
        out_names = ['BORE_OIL_VOL', 'BORE_GAS_VOL', 'BORE_WAT_VOL']

        for file in os.listdir(self.gen_dfs_path):
            df = pd.read_csv(f'{self.gen_dfs_path}/{file}')
            if df.shape[0] >= 50:
                outputs = df[out_names]
                inputs = df.drop(columns=out_names, axis=1)

                self.inputss.append(inputs)
                self.outputss.append(outputs)
                        
        for i, (inputs, outputs) in enumerate(zip(self.inputss, self.outputss)):
            all_len = len(inputs)
            train_len = int(np.floor(len(outputs) * self.train_all_ratio))
            val_len = (all_len - train_len) // 2
            self.train_lens.append(train_len)

            # split a multivariate sequence into samples
            raw_seq1 = np.hstack((inputs[:train_len], outputs[:train_len])) # ins and outs - train
            val = np.hstack((inputs[train_len:train_len + val_len], outputs[train_len:train_len + val_len])) # ins and outs - val
            true_data = np.hstack((inputs, outputs)) # ins and outs - all

            # normalize the data
            scaler = MinMaxScaler()
            raw_seq1_norm = scaler.fit_transform(raw_seq1) # norm all
            val_norm = scaler.transform(val)
            true_data_norm = scaler.transform(true_data)
            self.scalers.append(scaler)
            X, y = split_sequence(raw_seq1_norm, self.n_steps, self.n_features_in, self.n_features_out)
            X_val, y_val = split_sequence(val_norm, self.n_steps, self.n_features_in, self.n_features_out)
            X_all, _ = split_sequence(true_data_norm, self.n_steps, self.n_features_in, self.n_features_out)

            # flatten input, if needed
            if 'MLP' in self.model_name or 'XGB' in self.model_name:
                X_reshaped = X.reshape((X.shape[0], self.n_input))
                X_val_reshaped = X_val.reshape((X_val.shape[0], self.n_input))
                X_all_reshaped = X_all.reshape((X_all.shape[0], self.n_input))
            else:
                X_reshaped = X
                X_val_reshaped = X_val
                X_all_reshaped = X_all

            # append
            if i == 0:
                self.X_train = X_reshaped
                self.y_train = y
            elif i < len(self.inputss) - 10:
                self.X_train = np.vstack((self.X_train, X_reshaped))
                self.y_train = np.vstack((self.y_train, y))

                if X_val_reshaped.shape[0] != 0:
                    if self.X_valid is None:
                        self.X_valid = X_val_reshaped
                        self.y_valid = y_val
                    else:
                        self.X_valid = np.vstack((self.X_valid, X_val_reshaped))
                        self.y_valid = np.vstack((self.y_valid, y_val))
            else:
                self.X_all_reshaped_s.append(X_all_reshaped)
                self.raw_seq1_norm_s.append(raw_seq1_norm)
                self.true_data_norm_s.append(true_data_norm)
                self.true_data_s.append(true_data)

        self.X_train, self.y_train = shuffle(self.X_train, self.y_train)
        
        # print number of valid files
        tpl_files = list_ext(self.tpl_path, 'tpl')
        print (f'{len(self.inputss)} out of {len(os.listdir(self.gen_dfs_path))} ({np.round(len(self.inputss)/len(os.listdir(self.gen_dfs_path))*100, 1)}%) without errors')
        print (f'which means')
        print (f'{len(self.inputss)} out of {len(tpl_files)} ({np.round(len(self.inputss)/len(tpl_files)*100, 1)}%) without errors total')

    def create_model(self):
        # load model
        model_path = f'{self.study_path}/{self.model_name}.hdf'

        if 'MLP' in self.model_name:
            self.model = create_MLP_1(n_layers=self.n_layers, n_neurons=self.n_neurons, dropout=self.dropout)

        elif 'LSTM' in self.model_name:
            self.model = create_LSTM_1(n_layers=self.n_layers, n_neurons=self.n_neurons, dropout=self.dropout)
            
        elif 'TCN' in self.model_name:
            self.batch_x = 30
            self.model = create_TCN_1(batch_x=self.batch_x, batch_y=self.X_train.shape[1], 
                                 nb_filters=10, kernel_size=5, dilations=[1, 2], dropout_rate=self.dropout)

        if 'XGB' in self.model_name:
            self.model = MultiOutputRegressor(XGBRegressor(n_estimators=self.n_neurons, 
                                                           max_depth=self.max_depth, reg_alpha=self.reg_alpha))
        else:
            self.optimizer = tensorflow.keras.optimizers.Adam(lr=self.lr)
                            
    def train(self, n_epochs=50, verbose=1, batch_size=32):
        self.n_epochs = n_epochs
        
        if 'XGB' not in self.model_name:
            self.model.compile(loss=self.loss, optimizer=self.optimizer)
        
        # train the model
        start_time = time.time()

        if 'TCN' in self.model_name:
            shape_train = self.X_train.shape[0] - self.X_train.shape[0] % self.batch_x
            shape_test = self.X_valid.shape[0] - self.X_valid.shape[0] % self.batch_x
            self.model_inf = self.model.fit(self.X_train[:shape_train], self.y_train[:shape_train], 
                                       epochs=self.n_epochs, verbose=verbose, 
                                       validation_data=(self.X_valid[:shape_test], self.y_valid[:shape_test]))
        elif 'XGB' in self.model_name:
            self.model.fit(self.X_train, self.y_train)
        else:
            if self.X_valid.shape[0] > 0:
                self.model_inf = self.model.fit(self.X_train, self.y_train, epochs=self.n_epochs, 
                                                verbose=verbose, batch_size=batch_size, 
                                                validation_data=(self.X_valid, self.y_valid))
            else:
                self.model_inf = self.model.fit(self.X_train, self.y_train, epochs=self.n_epochs, 
                                                verbose=verbose, batch_size=batch_size)  

        self.delta_time = time.time() - start_time
        self.cur_time = datetime.datetime.now()
        
    def predict(self, plot_losses=True, plot_preds=True, plot_errs=False, print_res=False):
        # obtain the model results: make a prediction and calculate an error

        if plot_losses:
            if 'XGB' not in self.model_name:
                loss = self.model_inf.history['loss']
                val_loss = self.model_inf.history['val_loss']
                plot_train_val_loss(loss, val_loss, model_name=self.model_name)
        
        for i, (X_all_reshaped, raw_seq1_norm, true_data_norm, true_data) in enumerate(
            zip(self.X_all_reshaped_s, self.raw_seq1_norm_s, self.true_data_norm_s, self.true_data_s)):
                
            # obtain the model results: make a prediction and calculate an error
            y_pred_norm = self.model.predict(X_all_reshaped)
            table_norm = np.hstack((true_data_norm[:, :-self.n_features_out], 
                                    np.vstack((raw_seq1_norm[:self.n_steps, -self.n_features_out:], y_pred_norm))))
            table = self.scalers[i].inverse_transform(table_norm) # denormalize all
            preds = table[:, -self.n_features_out:]
            errs = error(preds, true_data[:, -self.n_features_out:]) # calc error between denormalized y_pred and real y

            data = {}
            data["model_name"] = self.model_name
            if 'XGB' not in self.model_name:
                data["loss"] = self.model_inf.history['loss']
            data["train_size"] = self.train_lens[i]
            data["preds"] = preds.tolist()
            data["g_truth"] = true_data[:, -self.n_features_out:].tolist()
            data["errors"] = errs.tolist()
            data["training_time"] = self.delta_time
            data["cur_time"] = str(self.cur_time)

            if print_res:
                print_results(data)
            if plot_preds:
                plot_pred(data)
            if plot_errs:
                plot_err(data)

#         if 'XGB' not in self.model_name:
#             return np.mean(data["errors"][data["val_size"]:]), str(data["training_time"])[:4]
#         else:
        print (f"Train error:      %.3f" % (np.mean(data["errors"][:data["train_size"]])))
        print (f"Test error:       %.3f" % (np.mean(data["errors"][data["train_size"]:])))
        print (f"Training time:    %.3f s\n" % (float(str(data["training_time"])[:4])))
        return np.mean(data["errors"][data["train_size"]:]), float(str(data["training_time"])[:4])
        
    def save(self):
#         self.prefix = '-tuning' if self.tuning else '-transf'

        if 'XGB' in self.model_name:
            pickle.dump(self.model, open(f'{self.study_path}/{self.model_name}{self.prefix}.hdf', "wb"))
        elif 'TCN' in self.model_name:
            # get model as json string and save to file
            model_as_json = self.model.to_json()
            with open(f'{self.study_path}/{self.model_name}{self.prefix}.json', "w") as json_file:
                json_file.write(model_as_json)

            # save weights to file (for this format, need h5py installed)
            self.model.save_weights(f'{self.study_path}/{self.model_name}{self.prefix}.h5')
        else:
            self.model.save(f'{self.study_path}/{self.model_name}{self.prefix}')#, save_format='h5')

        print (f'{self.study_path}/{self.model_name}{self.prefix}.hdf saved')
        
    def full_launch(n_epochs=2, tuning=True, plot_losses=True, plot_preds=True, plot_errs=False):
        self.model.train(n_epochs=n_epochs, tuning=tuning)
        self.model.predict(plot_losses=plot_losses, plot_preds=plot_preds, plot_errs=plot_errs)
        self.model.save()