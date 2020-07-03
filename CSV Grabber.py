import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import os
import pandas_datareader.data as web 
import datetime as dt 

#Download a CSV file for a stock and display it's contents in a dataframe
ticker = input('Enter the ticker of the stock\n').upper()
input_filepath = "/Users/mraay/Desktop/yahoo-downloaded-data/"
start_date = dt.datetime(2001, 2, 20)
end_date = dt.datetime(2020, 6, 29)
stock = web.DataReader(f'{ticker}', 'yahoo', start_date, end_date)
stock.to_csv(input_filepath+f'{ticker}.csv')
