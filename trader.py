from nameko.timer import timer
from pymongo import MongoClient
from keras.models import load_model
from nameko.rpc import RpcProxy
import numpy as np
from trainer import DECIMALS

pyclient = MongoClient()
tradingDB = pyclient["Trading"]


class Trader:
    name = 'trader_service'

    y = RpcProxy('trainer_service')

    @timer(interval=10)
    def predict(self):
        newcandle_collection = tradingDB['new_candle']

        # Get the newest candle from MongoDB
        newest_candle = newcandle_collection.find({}) \
            .sort([{'candles.0.time', -1}]) \
            .limit(1)

        for nc in newest_candle:
            newest_candle = nc['candles'][0]['mid']

        print(newest_candle)

        # Process X
        X = np.ndarray(shape=(0, 4))

        # Clean and process data
        newest_candle['o'] = float(newest_candle['o'])
        newest_candle['h'] = float(newest_candle['h'])
        newest_candle['l'] = float(newest_candle['l'])
        newest_candle['c'] = float(newest_candle['c'])

        X = np.append(X,
                      np.array([[
                          # High to Open price
                          round(newest_candle['h'] / newest_candle['o'] - 1, DECIMALS),
                          # Low to Open price
                          round(newest_candle['l'] / newest_candle['o'] - 1, DECIMALS),
                          # Close to Open price
                          round(newest_candle['c'] / newest_candle['o'] - 1, DECIMALS),
                          # High to Low price
                          round(newest_candle['h'] / newest_candle['l'] - 1, DECIMALS)]]),
                      axis=0)

        # TODO: usare process x sistemare la funzione
        # X = self.y.process(newest_candle, type='predict')

        model = load_model('2022-01-03 11:41:40--- EUR_USD_H1')
        Y = model.predict(X)
        print(Y)

        # TODO: visualize predicted data (confronting with next candle close value)

