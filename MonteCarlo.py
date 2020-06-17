import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import style
import os
import pandas_datareader.data as web 
import datetime as dt 

ticker = input('Enter the ticker of the stock\n').upper()
probProfit = float(input("What probability of profit are you looking for:\n"))
input_filepath = "/Users/mraay/Desktop/yahoo-downloaded-data/"
start_date = dt.datetime(2018, 1, 1)
end_date = dt.datetime(2020, 6, 10)
stockPrice = web.DataReader(f'{ticker}', 'yahoo', start_date, end_date)['Adj Close']

stockReturns = stockPrice.pct_change()
lastPrice = stockPrice.iloc[-1]

numSimulations = 101
numDays = 32

monteCarlo_df = pd.DataFrame()

for simulation in range(numSimulations):
    count = 0
    dailyVolatility = stockReturns.std()
    
    simulatedPrices = []

    price = lastPrice * (1 + np.random.normal(0, dailyVolatility))
    simulatedPrices.append(price)

    for day in range(numDays):
        if count == 32:
            break
        price = simulatedPrices[count] * (1 + np.random.normal(0, dailyVolatility))
        simulatedPrices.append(price)
        count += 1

    monteCarlo_df[simulation] = simulatedPrices

endPrices = monteCarlo_df.loc[32]
upperPrice = np.percentile(endPrices, float(100-(100-probProfit)/2))
lowerPrice = np.percentile(endPrices, float((100-probProfit)/2))
monteCarloMean = endPrices.mean()
monteCarloStd = endPrices.std()
upperStd = lastPrice + monteCarloStd
lowerStd = lastPrice - monteCarloStd
print(f'Mean: {monteCarloMean}\n',f'Standard Deviation: {monteCarloStd}\n',f'Closing Price: {lastPrice}\n',
f'1 Std Above: {upperStd}\n',f'1 Std Below: {lowerStd}\n', f'Sell {int(round(upperPrice))} call\n',
f'Sell {int(round(lowerPrice))} put')

#Backtester
""" quote = web.DataReader(f'{ticker}', 'yahoo', start=dt.datetime(2020,6,9), end=dt.datetime(2020,6,9))['Adj Close']
print(quote)

if quote.loc['2020-06-09'] > lowerPrice and quote.loc['2020-06-09'] < upperPrice:
    print('We made bread')
else:
    print("We lost bread")"""

#Plot Monte Carlo Simulation
fig = plt.figure()
fig.suptitle("Stock Monte Carlo Simulations")
plt.plot(monteCarlo_df)
plt.axhline(y=lastPrice, color='r', ls='-', lw=3)
plt.axhline(y=upperStd, color='b', ls='-', lw=3)
plt.axhline(y=lowerStd, color='b', ls='-', lw=3)
plt.axhline(y=upperPrice, color='g', ls='-', lw=3)
plt.axhline(y=lowerPrice, color='g', ls='-', lw=3)
plt.xlabel('Day')
plt.ylabel('Price')
plt.show()