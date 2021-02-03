
import pandas_datareader as dr
import pandas as pd
class CSVWorkAround:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker

        df = dr.DataReader(ticker, 'yahoo', start=start_date, end=end_date)
        df.to_csv(ticker+'Data.csv')


    def read_data(self):
        return pd.read_csv(self.ticker +'Data.csv')