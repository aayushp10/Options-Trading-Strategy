import math
import numpy as np
import datetime as dt
from numpy import newaxis
from keras.layers import Dense, Activation, Dropout, LSTM, Flatten
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
import pickle

class LSTMModel():
    def __init__(self, name, configs):
        self.configs = configs
        try:
            self.modelName = name
            self.model = pickle.load(open(name, 'rb'))
        except:
            self.model = Sequential()

    def build_model(self):

        for layer in self.configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None


            if layer['type'] == 'flatten':
                self.model.add(Flatten())
            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation = activation))
            if layer['type'] == 'lstm':
                self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))


        self.model.compile(loss=self.configs['model']['loss'], optimizer=self.configs['model']['optimizer'])


    def train(self, x, y):

        epochs = self.configs['training']['epochs']
        batch_size = self.configs['training']['batch_size']
        callbacks = [EarlyStopping(monitor='val_loss', patience=2)]


        self.model.fit(x,y,epochs=epochs, batch_size=batch_size, callbacks=callbacks)
        pickle.dump(self.model, open(self.modelName, 'wb'))


    def test(self, processedData, shift):
        x_test = []
        y_test = processedData.y_test
        for i in range(shift, len(processedData.x_test) + shift):
            x_test.append(processedData.x_test[i-shift: i, 0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

        #get predicted values
        predictions = self.model.predict(x_test)
        predictions = processedData.scaler.inverse_transform(predictions)

        #get the root mean squared error (RMSE)

        rmse = np.sqrt(mean_squared_error(predictions, y_test))
        print(rmse)


    def predict(self, days_out, x_input):
        temp_input = list(x_input)
        # temp_input = temp_input[0].tolist()
        lst_output = []
        n_steps = self.configs['data']['sequence_length']
        i = 0
        while (i < days_out):

            if (len(temp_input) > n_steps):
                # print(temp_input)
                x_input = np.array(temp_input[1:])
                print("{} day input {}".format(i, x_input))
                x_input = x_input.reshape(1, -1)
                x_input = x_input.reshape((1, n_steps, 1))
                # print(x_input)
                yhat = self.model.predict(x_input, verbose=0)
                print("{} day output {}".format(i, yhat))
                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                # print(temp_input)
                lst_output.extend(yhat.tolist())
                i = i + 1
            else:
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = self.model.predict(x_input, verbose=0)
                print(yhat[0])
                temp_input.extend(yhat[0].tolist())
                print(len(temp_input))
                lst_output.extend(yhat.tolist())
                i = i + 1

        return lst_output


