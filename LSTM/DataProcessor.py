

import pandas_datareader as dr

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

import numpy as np

import json
import math
from Options.csvworkaround import CSVWorkAround

class DataProcessor:



    def __init__(self, ticker, start_date, end_date, configs):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.configs = configs

    def getDataSet(self, window_size):
        csv = CSVWorkAround(self.ticker, '2020-04-01', '2020-06-28')
        df = csv.read_data()
        data = df
        data = data.drop(['Date', 'Close'], axis=1)
        print('OG Data Len: ' + str(len(data)))
        return data

    def prepareData(self):
        window_size = self.configs['data']['sequence_length']

        ##determine test_size
        data = self.getDataSet(window_size)
        split_size = self.configs['data']['train_test_split']

        cut_off = len(data) % window_size
        data = data[cut_off: ]
        data = data.filter(['Adj Close'])
        close = data.filter(['Adj Close'])

        training_len = float(len(data) * float(split_size))
        training_len = window_size * round(training_len/window_size) - window_size

        x_train = data[0:training_len: ]
        x_test = data[training_len:]
        y_train = close[0: training_len]
        y_test = close[training_len: ]

        scaler = MinMaxScaler(feature_range=(0, 1))

        x_train = scaler.fit_transform(x_train)
        x_test = scaler.fit_transform(x_test)
        y_train = scaler.fit_transform(y_train)
        y_test = scaler.fit_transform(y_test)

        x_train = np.array(x_train).reshape(int(x_train.shape[0]/window_size), window_size, x_train.shape[1])
        x_test = np.array(x_test).reshape(int(x_test.shape[0]/window_size), window_size, x_test.shape[1])
        y_train = np.array(y_train).reshape(int(y_train.shape[0]/window_size), window_size, y_train.shape[1])
        y_test = np.array(y_test).reshape(int(y_test.shape[0]/window_size), window_size, y_test.shape[1])


        print(x_train.shape)
        print(x_test.shape)
        print(y_train.shape)
        print(y_test.shape)


        self.x_train_scaled = x_train
        self.x_test_scaled = x_test
        self.y_train_scaled = y_train
        self.y_test_scaled = y_test
        self.scaler = scaler
        self.close_dataset = close





