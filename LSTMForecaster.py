#Long Short-Term Memory Model for closing stock price


import math
import pandas_datareader as web
import quandl
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import pickle

plt.style.use('fivethirtyeight')


def test(model, data):
    #create the data sets x_Test and y_test

    scaled_data = data['scaled_data']
    training_data_len = data['training_data_len']

    scaler = data['scaler']
    data['close_dataset'].append([0,1,2,3,4,5,6,7,8,9,10,11])
    y_test = data['close_dataset'][training_data_len:, :]


    #get predicted values for x_test

    predictions = model.predict([0,1,2,3,4,5,6,7,8,9,10,11])
    print(predictions)
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
    print(df.columns)

    df = df[['Open', 'High', 'Low', 'Adj Close', 'Volume', 'Close']]
    df['HL_PCT'] = (df['High'] - df['Low']) / df['Low']
    df['PCT_change'] = (df['Close'] - df['Open']) / df['Open']
    df = df[['Adj Close', 'HL_PCT', 'PCT_change', 'Volume']]

    forecast_col = 'Adj Close'
    df.dropna(inplace=True)
    forecast_out = int(math.ceil(12))  ## predict 12d out which is daily



    df['forecast'] = df[forecast_col].shift(-forecast_out)

    X = np.array(df.drop(['forecast'], 1))  ##features

    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.tran(X)

    X = X[:-forecast_out]
    X_lately = X[-forecast_out:]

    df.dropna(inplace=True)
    y = np.array(df['forecast'])

    X_train = df[:-forecast_out, :]
    X_test = df[-forecast_out:, :]


    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)


    print(df)
    # training_data_len = math.ceil(len(scaled_data))
    # training_data = scaled_data[0:training_data_len, :]
    # testing_data = scaled_data[training_data_len: , :]
    # #return [training_data, testing_data]
    # return {'scaled_data': scaled_data, 'scaler': scaler, 'training_data_len':training_data_len, 'close_dataset': close_dataset }

try:
    getDataSet()
    # model = pickle.load(open('x', 'rb'))
    # data = getDataSet()
    # test(model, data)

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
