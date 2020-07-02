import math
import pandas_datareader as web
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import pickle
import os

from pandas.plotting import register_matplotlib_converters
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM

register_matplotlib_converters()

#Download ticker data from Yahoo Finance and store it in a dataframe called stock
ticker = input('Enter the ticker of the stock\n').upper()
start_date = dt.datetime(2016, 4, 12)
end_date = dt.datetime(2020, 6, 16)
stock = web.DataReader(f'{ticker}', 'yahoo', start=start_date, end=end_date)
stockPrice = stock.reset_index()['Adj Close']
plt.style.use('ggplot')
plt.plot(stockPrice, color = '#4169E1', lw=2)
#plt.show()


#Scale the training data so it is between 0 and 1
scaler = MinMaxScaler(feature_range=(0,1))
stockPrice = scaler.fit_transform(np.array(stockPrice).reshape(-1,1))

#Identify Training Data and Testing Data
trainingSize = int(len(stockPrice)*.80)
testingSize = len(stockPrice) - trainingSize
trainingData, testingData = stockPrice[0:trainingSize,:], stockPrice[trainingSize:len(stockPrice),:1]

#Create a function to create the dataset for X_train and Y_train 
#where the time step represents how many days the next point depends on
def createDataset(dataset, timeStep=1):
    X_train, Y_train = [], []
    for data in range(len(dataset) - timeStep - 1):
        X_train.append(dataset[data:(data + timeStep), 0])
        Y_train.append(dataset[data + timeStep, 0])
    return np.array(X_train), np.array(Y_train)


timeStep = 20
X_train, Y_train = createDataset(trainingData, timeStep)
X_test, Y_test = createDataset(testingData, timeStep)

#Reshape the input to be 3-D with [samples, timeSteps, features], which is how it needs to be formatted for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

#Create LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, Y_train, validation_data=(X_test,Y_test),epochs=5,batch_size=1)

#Create a prediction and check performance metrics
trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)

#Transform it back to original price form using inverse_transform
trainPredict = scaler.inverse_transform(trainPredict)
testPredict = scaler.inverse_transform(testPredict)

#Calculate root mean squared error for the train and test
rmseTrain = np.sqrt(mean_squared_error(Y_train, trainPredict))
rmseTest = np.sqrt(mean_squared_error(Y_test, testPredict))
print(rmseTrain, rmseTest)

"""
temp_input=list(x_input)
temp_input=temp_input[0].tolist()
lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1

print(lst_output)
"""