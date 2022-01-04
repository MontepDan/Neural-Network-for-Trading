from data import *

cursor = tradingDB['new_candle'].find({})
for document in cursor:
    print(document)

# TODO: Creare GitHub repository con codice master e dev branches.

# nameko run data
# nameko run trainer

