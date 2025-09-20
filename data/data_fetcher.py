# data/data_fetcher.py
import yfinance as yf
import ccxt
from datetime import datetime, timedelta
import pandas as pd
from typing import Union, Optional
from utils.logging import logger
from utils.cache import cached

class DataFetcher:
    """Class to fetch stock and crypto data from various sources"""
    
    def __init__(self):
        self.exchange = ccxt.binance()
    
    @cached("stock_data_{symbol}_{period}_{interval}", expire=3600)
    def fetch_stock_data(self, symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        """Fetch stock data from Yahoo Finance"""
        try:
            logger.info(f"Fetching stock data for {symbol}, period: {period}, interval: {interval}")
            stock = yf.Ticker(symbol)
            df = stock.history(period=period, interval=interval)
            df = df.rename(columns={c: c.lower() for c in df.columns})
            logger.info(f"Successfully fetched {len(df)} rows of stock data for {symbol}")
            return df
        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {e}")
            raise
    
    @cached("crypto_data_{symbol}_{days}_{timeframe}", expire=1800)
    def fetch_crypto_data(self, symbol: str, days: int = 730, timeframe: str = '1d') -> pd.DataFrame:
        """Fetch cryptocurrency data from Binance"""
        try:
            logger.info(f"Fetching crypto data for {symbol}, days: {days}, timeframe: {timeframe}")
            since = self.exchange.parse8601((datetime.now() - timedelta(days=days)).isoformat())
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            logger.info(f"Successfully fetched {len(df)} rows of crypto data for {symbol}")
            return df
        except Exception as e:
            logger.error(f"Error fetching crypto data for {symbol}: {e}")
            raise