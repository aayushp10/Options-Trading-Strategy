import pandas as pd
import pandas_datareader as dr

import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# model building libraries
import math
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, BatchNormalization, Dropout, Activation, Embedding, LeakyReLU, Embedding, \
    Flatten
from keras.activations import relu, sigmoid
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

import matplotlib.pyplot as plt


def get_data(ticker):
    return dr.DataReader(ticker, 'yahoo', start="2019-02-01", end="2020-02-01")


def process_data(data, n_features):
    dataX, dataY = [], []

    for i in range(len(data) - n_features - 1):
        a = data[i: (i + n_features), 0]
        dataX.append(a)
        dataY.append(data[i + n_features, 0])

    return np.array(dataX), np.array(dataY)


if __name__ == '__main__':
    stock_data = get_data('AMZN')
    stock_closings = stock_data['Adj Close']
    stock_closings = np.array(stock_closings)

    '''
    Feature Scaler is applied so that we figure out theta (weigth) faster. 
    On a positively parabolic curve, theta is the global minima.

    *If a label is a categorical variable then we have use LabelEncoder*

    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelEncoder_1 = LabelEncoder
    x[:,col] = labelEncoder_1.fit_transform(x[:1])

    x = onehotencoder.fit_tranform(x).toArray()
    '''

    scaler = MinMaxScaler(feature_range=(0, 1))
    fitted_closings = scaler.fit_transform(stock_closings[:, np.newaxis])

    train = fitted_closings[0:int(len(fitted_closings) * 0.8)]
    test = fitted_closings[int(len(fitted_closings) * 0.8):]

    train = train.reshape(len(train), 1)
    test = test.reshape(len(test), 1)

    n_features = 2
    trainX, trainY = process_data(train, n_features)
    testX, testY = process_data(test, n_features)

    trainX = trainX.reshape(trainX.shape[0], 1, trainX.shape[1])
    testX = testX.reshape(testX.shape[0], 1, testX.shape[1])

    filepath = "stock_weights.hdf5"

    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=1)

    checkpoint = ModelCheckpoint(filepath, monitor='val loss', verbose=1, save_best_only=True, mode='max')

    model = Sequential()
    model.add(GRU(256, input_shape=(1, n_features), return_sequences=True))

    model.add(Dropout(0.4))
    model.add(LSTM(256))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))

    # lr = learning rate
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0005), metrics=['mean_squared_error'])

    history = model.fit(trainX, trainY, epochs=100, batch_size=128, callbacks=[checkpoint, lr_reduce],
                        validation_data=(testX, testY))

    pred = model.predict(testX)

    pred = scaler.inverse_transform(pred)

    testY = testY.reshape(testY.shape[0], 1)
    testY = scaler.inverse_transform(testY)

    print(pred[:10], testY[:10])







    plt.rcParams["figure.figsize"] = (10, 7)
    plt.plot(testY, 'b')
    plt.plot(pred, 'r')

    plt.xlabel('Time')
    plt.ylabel('Stock Price')

    plt.title('Check the accuracy of the model with time')

    plt.grid(True)
    plt.show()
    # print(trainX.shape)
    # print(trainY.shape)
    # print(testX.shape)
    # print(testY.shape)

    '''
    How to determine what layers to apply:

    Hyperparameter Optimization




    from keras.wrappers.scikit_learn import KerasClassifier
    from sklearn.model_selection import GridSearchCV



    '''


    def create_model(layers, activation):
        model = Sequential()
        for i, nodes in enumerate(layers):
            if i == 0:
                model.add(Dense(nodes, input_dim=trainX.shape[1]))
                model.add(Activation(activation))
            else:
                model.add(Dense(nodes))
                model.add(Activation(activation))
        model.add(Dense(1))
        # loss is binary_crossentropy beca
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model


    '''

    https://www.youtube.com/watch?v=Bc2dWI3vnE0

    '''
