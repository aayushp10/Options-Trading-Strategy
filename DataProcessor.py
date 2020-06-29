

import pandas_datareader as dr

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

import numpy as np

import json

from Options.csvworkaround import CSVWorkAround

class DataProcessor:



    def __init__(self, ticker, start_date, end_date, configs):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.configs = configs

    def getDataSet(self):
        csv = CSVWorkAround(self.ticker, '2020-04-01', '2020-06-28')
        df = csv.read_data()
        data = df.filter(['Adj Close'])
        print(data)
        return data

    def prepareData(self):
        data = self.getDataSet()
        close = data['Adj Close']

        split_size = self.configs['data']['train_test_split']
        X_train, X_test, y_train, y_test = train_test_split(data, close, test_size=split_size)


        X_train = np.array(X_train).reshape(-1,1)
        X_test = np.array(X_test).reshape(-1,1)
        y_train = np.array(y_train).reshape(-1,1)
        y_test = np.array(y_test).reshape(-1,1)

        scaler = MinMaxScaler(feature_range=(0, 1))

        self.x_train_scaled = scaler.fit_transform(X_train)
        self.x_test_scaled = scaler.fit_transform(X_test)
        self.y_train_scaled = scaler.fit_transform(y_train)
        self.y_test_scaled = scaler.fit_transform(y_test)
        self.scaler = scaler
        self.close_dataset = close





