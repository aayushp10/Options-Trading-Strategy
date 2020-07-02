
from Options.DataProcessor import DataProcessor
from Options.LSTMModel import LSTMModel
from Options.csvworkaround import CSVWorkAround
import json
import numpy as np
ticker = 'AAPL'
configs = json.load(open('config.json', 'r'))


data = DataProcessor(ticker, '2020-04-01', '2020-06-28', configs)
data.prepareData()

model = LSTMModel(ticker, configs)
model.build_model()

# data.x_train_scaled = np.reshape(data.x_train_scaled, (data.x_train_scaled.shape[0], data.x_train_scaled.shape[1] * data.x_train_scaled.shape[2]))
data.y_test_scaled = np.reshape(data.y_test_scaled, (data.y_test_scaled.shape[0], data.y_test_scaled.shape[1] * data.y_test_scaled.shape[2]))

model.train(data.x_train_scaled, data.y_train_scaled)
model.test(data, 30)
#
#
