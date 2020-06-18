#Long Short-Term Memory Model for closing stock price


import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import pickle

plt.style.use('fivethirtyeight')


def test(model, data):
    #create the data sets x_Test and y_test

    scaled_data = data['scaled_data']
    training_data_len = data['training_data_len']

    scaled_test_data = scaled_data[training_data_len - 12:, :]

    scaler = data['scaler']
    x_test = []
    y_test = data['close_dataset'][training_data_len:, :]
    for i in range(12, len(scaled_test_data)):
        x_test.append(scaled_test_data[i-12: i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    #get predicted values for x_test

    predictions = model.predict(x_test)

    predictions = scaler.inverse_transform(predictions)

    # get the root mean squared error (RMSE) -> lower values are better
    rmse = np.sqrt(np.mean(predictions - y_test)**2)
    print(rmse)

    train = data['close_dataset'][:training_data_len]
    valid = data['close_dataset'][training_data_len:]
    for i in range(0,valid.size):
        valid[i] = predictions[i]

    output = train.tolist()
    output.extend(valid)
    plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.xlabel('Date', fontsize='18')
    plt.ylabel('Close price $USD', fontsize='18')
    plt.plot(output, color='orange', label='Predicted Price', lw=2)
    plt.plot(train, color='blue', label='Original Price', lw=2)
    # plt.plot(valid[[Close',' 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()


def getDataSet():
    df = web.DataReader('TSLA', data_source="yahoo", start='2016-04-12', end='2020-07-31')
    adj_close = df.filter(['Adj Close'])
    close_dataset = adj_close.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_dataset)
    training_data_len = math.ceil(len(scaled_data)-12)
    training_data = scaled_data[0:training_data_len, :]
    testing_data = scaled_data[training_data_len - 12: , :]
    #return [training_data, testing_data]
    return {'scaled_data': scaled_data, 'scaler': scaler, 'training_data_len':training_data_len, 'close_dataset': close_dataset }

try:
    model = pickle.load(open('x', 'rb'))
    data = getDataSet()
    test(model, data)

except FileNotFoundError:


    data = getDataSet()
    scaled_data = data['scaled_data']
    training_data_len = data['training_data_len']
    training_data = scaled_data[0:training_data_len, :]

    x_train = []
    y_train = []

    for i in range(20, len(training_data)):
        x_train.append((training_data[i-12: i, 0]))
        y_train.append((training_data[i, 0]))

    #convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    #reshape data (LSTM expects 3d data : number of steps, time steps, features
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    #Build the LSTM Model

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2]) ) )
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))


    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(x_train, y_train, batch_size=1, epochs=10)
    model_name = '2020-06-16'
    pickle.dump(model, open(model_name, 'wb'))
    test(model, data)







#
# plt.figure(figsize=(16,8))
# plt.title('Adj Closing History')
# plt.plot(df['Adj Close'])
# plt.xlabel('Date')
# plt.ylabel('Adj Close')
# plt.show()
