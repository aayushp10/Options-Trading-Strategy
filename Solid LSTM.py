import pandas_datareader as pdr
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from numpy import newaxis
#ticker = input("Enter a ticker\n")
#df = pdr.DataReader(f'{ticker}', 'yahoo', start='2005-01-01', end='2020-01-14')
#df.to_csv(f'{ticker}.csv')
df = pd.read_csv('AAPL.csv')
df1 = df.reset_index()['Adj Close']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1,1))

training_size = int(len(df1)*0.85)
test_size = len(df1)-training_size
train_data, test_data = df1[0:training_size, :], df1[training_size:len(df1), :1]

def create_dataset(dataset, time_step=1):
    dataX = []
    dataY = []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i: (i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)


X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step,1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=128, verbose=1)

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(test_predict, y_test))
print(rmse)

look_back = 100
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict

#plt.plot(scaler.inverse_transform(df1), color='blue')
# plt.plot(trainPredictPlot, color='orange')
# plt.plot(testPredictPlot, color='green')
# plt.show()

days_out = 100
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
prediction_seqs = []
for i in range(int(X_train.shape[0]/days_out)):

    curr_frame = (X_train[:, :])[(i)*days_out]
    print('current frame \n')
    print(curr_frame)

    predicted = []
    for j in range(days_out):

        a = model.predict(curr_frame[newaxis,:, :], verbose=1)
        a = a[0,0]
        predicted.append(a)
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [look_back-2], predicted[-1], axis=0)

    prediction_seqs.append(predicted)
print(len(prediction_seqs))


x_input = test_data[len(test_data)-look_back:].reshape(1,-1)


temp_input = list(x_input)
temp_input = temp_input[0].tolist()

lst_output = []
n_steps = look_back

i = 0
while (i<days_out):
    if (len(temp_input)>n_steps):
        x_input=np.array(temp_input[1:])
        x_input=x_input.reshape(1, -1)
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose = 1)
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        lst_output.extend(yhat.tolist())
        i = i+1

    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=1)
        temp_input.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i = i+1

day_new = np.arange(1, look_back+1)
day_pred = np.arange(look_back+1, look_back+1+days_out)

df3 = scaler.inverse_transform(df1)
df3 = df3.tolist()

fig = plt.figure(facecolor='white')
ax = fig.add_subplot(111)
ax.plot(df3, label='True Data')
# Pad the list of predictions to shift it in the graph to it's correct start
for i, data in enumerate(prediction_seqs):
    print('yo')
    prediction_seqs[i] = scaler.inverse_transform(np.array(data).reshape((-1,1)))
for i, data in enumerate(prediction_seqs):
    padding = [None for p in range(i * days_out)]
    padding = np.array(padding).reshape((-1,1))
    pltData = np.concatenate((padding, data))
    plt.plot(pltData, label='Prediction')
    plt.legend()
plt.show()
#plt.plot(df3, color = 'orange')
#plt.plot(scaler.inverse_transform(lst_output), color = 'blue')
#plt.show()
