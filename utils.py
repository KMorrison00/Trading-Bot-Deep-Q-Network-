# Yahoo Finance Api setup
# Author Kyle, Tyler

import yfinance as yf
import matplotlib
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader


class yFinanceDataset(Dataset):
    """Single Stock Dataset"""
    """
    Known E-Mini Futures tickers:
    S&P 500: ES=F
    NASDAQ: NQ=F
    DOW Jones: YM=F
    """

    # Default stock data we use incase no data is supplied
    default_stock_data =  {
        'tickers'  : 'ES=F',
        'period'   : '1mo',
        'interval' : '1m',
        'group_by' : 'ticker',
        'start'    : '2020-04-04',
        'end'      : '2020-04-11'
    }

    def __init__(self, stock_data=default_stock_data):
        self._stock_data = stock_data
        self._data = self._get_data()

    def _get_data(self):
        """ Return data from y-finance, convert to tensor"""

        """ARGS:
                tickers: the name of each stock separated by a space ex:"GOOG SPY AAPL"

                period: the start/end date for data ex: 
                    1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max

                interval: fetch data interval valid intervals: 
                    1m,2m,5m,15m,30m,60m,90m,1h,1d (default),5d,1wk,1mo,3mo

                group_by: group data by something ex: ticker, default is by column
        """
        data = yf.download( 
            tickers = self._stock_data['tickers'],
            start = self._stock_data['start'],
            end = self._stock_data['end'],
            interval = self._stock_data['interval'],
            group_by = self._stock_data['group_by']
        )
        data.to_csv('dataset.csv', index=True, header=False, mode='a')
        # data recieved in order: Datetime, Open, High, Low, Close, Adj Close, Volume
        return data
            
    def __len__(self):
        return len(self.data)


dataset = yFinanceDataset()  # for data checking, needed to instantiate a dataset obj
