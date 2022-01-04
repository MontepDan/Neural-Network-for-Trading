from nameko.rpc import rpc
from nameko.timer import timer

import pprint
from time import time, ctime, strftime, localtime
from datetime import datetime

import oandaapi
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments

from pymongo import MongoClient

pyclient = MongoClient()
tradingDB = pyclient["Trading"]

pp = pprint.PrettyPrinter(width=41, compact=True)

access_token = oandaapi.oandaAPI
account_ID = oandaapi.accountID

client = API(access_token=access_token)

INSTRUMENT = "EUR_GBP"
GRANULARITY = "H1"

# TODO: sistemare from_date
FROM_DATE = datetime.strptime('01/01/2021 00:00:01', '%d/%m/%Y %H:%M:%S')
TO_DATE = datetime.strptime(ctime(time()), "%a %b %d %H:%M:%S %Y") # TODAY

# robo3t x controllare graficamente DB


class DataCollector:
    name = 'data_service'

    @timer(interval=5)
    def get_ohlc(self, instrument=INSTRUMENT, granularity=GRANULARITY):

        """ Get newest candle from OANDA"""

        params = {"granularity": granularity,
                  "count": 2}

        # Obtains candle data from OandaAPI

        newest_candle = instruments.InstrumentsCandles(instrument, params)  # prepara i dettagli della richiesta HTTP

        newest_candle = client.request(newest_candle)  # esegue la richiesta HTTP e la risposta viene salvata in prices

        # Insert newest candle value in tradingDB if entry is not duplicate

        newcandle_collection = tradingDB['new_candle']

        current_timestamp = newest_candle['candles'][0]['time']

        find_double = newcandle_collection.find_one({'time': current_timestamp})

        if find_double is not None:
            return False

        newcandle_collection.insert_one(newest_candle)

    @rpc
    def get_historical_ohlc(self, instrument=INSTRUMENT, granularity=GRANULARITY, count=5000):

        # TODO: sistemare from_date
        # def get_historical_ohlc(self, instrument=INSTRUMENT, granularity=GRANULARITY, from_date=FROM_DATE, to_date=TO_DATE,
        #                             count=5000):

        """ Get historical candles data from OANDA """

        params = {"granularity": granularity,
                  "count": count}
                  # "from": from_date,
                  # "to": to_date}

        # Obtains candle data from OandaAPI

        historical_candle = instruments.InstrumentsCandles(instrument, params)

        historical_candle = client.request(historical_candle)

        # Insert historical candle values in tradingDB

        histcandle_collection = tradingDB["historical_candle"]

        histcandle_collection.insert_one(historical_candle)

        del historical_candle['_id']

        return historical_candle

