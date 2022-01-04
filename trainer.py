import pandas as pd
from nameko.rpc import rpc, RpcProxy
from nameko.timer import timer
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers
import numpy as np
from sklearn.model_selection import train_test_split
import datetime

WEEK = 60 * 60 * 24 * 7
DECIMALS = 6
Y_CHANGE_THRESHOLD = 0.001


class Trainer:
    name = 'trainer_service'

    y = RpcProxy('data_service')

    # TODO: Train model with multiple 5000 points datasets

    @staticmethod
    def get_model():
        """
        Definition of the Model Layers using Keras
        :return: Keras Model Object
        """

        model = Sequential()

        model.add(Dense(units=64,
                        activation='relu',
                        input_shape=(4,)))

        model.add(Dense(units=64,
                        activation='sigmoid',
                        kernel_regularizer=regularizers.l2(0.001),
                        activity_regularizer=regularizers.l1(0.001)))

        model.add(Dense(units=3,
                        activation='softmax'))

        sgd = SGD(learning_rate=0.0001)

        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

        return model

    # Retrain the model 1 time x week if runs continuously
    # @timer(interval=20)
    @rpc
    def retrain(self):
        """Retrain a model for a specific a) trading instrument, b) timeframe, c) input shape"""

        # Get historical data from data_service
        candles = self.y.get_historical_ohlc()['candles']

        X, Y = self.process(candles, type='train')

        # Train, test split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

        # Train Model
        model = self.get_model()
        fit = model.fit(X_train, Y_train, batch_size=32, epochs=100, verbose=True)
        score = model.evaluate(X_test, Y_test, batch_size=128)

        print(model.summary())
        print(score)

        # Save trained model to disk
        filename = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model.save(filename + '--- EUR_USD_H1')

        return model

    @rpc
    def process(self, candles, type='train'):
        """Processing candles to a format/shape consumable by the model
        :param candles: dict/list of Open, High, Low, Close prices
        :return: X: numpy.ndarray, Y: numpy.ndarray"""

        if type == 'train':

            X = np.ndarray(shape=(0, 4))
            Y = np.ndarray(shape=(0, 1))

            # Clean and process data
            previous_close = None

            for candle in candles:
                candle = candle['mid']
                candle['o'] = float(candle['o'])
                candle['h'] = float(candle['h'])
                candle['l'] = float(candle['l'])
                candle['c'] = float(candle['c'])

                X = np.append(X,
                              np.array([[
                                  # High to Open price
                                  round(candle['h'] / candle['o'] - 1, DECIMALS),
                                  # Low to Open price
                                  round(candle['l'] / candle['o'] - 1, DECIMALS),
                                  # Close to Open price
                                  round(candle['c'] / candle['o'] - 1, DECIMALS),
                                  # High to Low price
                                  round(candle['h'] / candle['l'] - 1, DECIMALS)]]),
                              axis=0)

                # Percentage variation prediction between Close and Close in the previous candle
                if previous_close is not None:
                    y = None
                    precise_prediction = round(candle['c'] / previous_close - 1, DECIMALS)

                    # Positive price change
                    if precise_prediction > Y_CHANGE_THRESHOLD:
                        y = 1
                    # Negative price change
                    elif precise_prediction < 0 - Y_CHANGE_THRESHOLD:
                        y = 2
                    # Neutral price change
                    else:
                        y = 0

                    Y = np.append(Y, np.array([[y]]))
                else:
                    Y = np.append(Y, np.array([[0]]))

                previous_close = round(candle['c'], DECIMALS)

            Y = np.delete(Y, 0)
            Y = np.append(Y, np.array([0]))
            Y = to_categorical(Y, num_classes=3)

            return X, Y

        elif type == 'predict':

            X = np.ndarray(shape=(0, 4))

            # Clean and process data
            candles['o'] = float(candles['o'])
            candles['h'] = float(candles['h'])
            candles['l'] = float(candles['l'])
            candles['c'] = float(candles['c'])

            X = np.append(X,
                          np.array([[
                              # High to Open price
                              round(candles['h'] / candles['o'] - 1, DECIMALS),
                              # Low to Open price
                              round(candles['l'] / candles['o'] - 1, DECIMALS),
                              # Close to Open price
                              round(candles['c'] / candles['o'] - 1, DECIMALS),
                              # High to Low price
                              round(candles['h'] / candles['l'] - 1, DECIMALS)]]),
                          axis=0)

            return X
