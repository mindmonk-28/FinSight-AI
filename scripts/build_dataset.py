# scripts/build_minute_dataset.py

import os
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice
from ta.utils import dropna

# Output directory for minute-level data
data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "minute")
os.makedirs(data_dir, exist_ok=True)

# List of Indian tickers (NSE) to include
indian_tickers = [
    "RELIANCE.NS", "INFY.NS", "TCS.NS", "WIPRO.NS", "HDFCBANK.NS",
    "ICICIBANK.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "ASIANPAINT.NS",
    "LT.NS", "ULTRACEMCO.NS", "MARUTI.NS", "AXISBANK.NS", "BAJFINANCE.NS",
    "NESTLEIND.NS", "TECHM.NS", "KOTAKBANK.NS", "SUNPHARMA.NS", "HCLTECH.NS"
]

def add_technical_indicators(df):
    """Add technical indicators to the dataframe"""
    try:
        # Moving averages
        df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
        df['EMA_20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
        
        # MACD
        macd = MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Histogram'] = macd.macd_diff()
        
        # RSI
        df['RSI'] = RSIIndicator(close=df['Close']).rsi()
        
        # Bollinger Bands
        bb = BollingerBands(close=df['Close'])
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Middle'] = bb.bollinger_mavg()
        df['BB_Lower'] = bb.bollinger_lband()
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # Volume indicators
        df['VWAP'] = VolumeWeightedAveragePrice(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']).volume_weighted_average_price()
        
    except Exception as e:
        print(f"Error adding technical indicators: {e}")
    
    return df

def fetch_minute_data(ticker):
    try:
        print(f"Fetching minute-level data for {ticker}...")
        data = yf.download(ticker, period="7d", interval="1m", auto_adjust=True)
        
        # Handle None return from yf.download
        if data is None or data.empty:
            print(f"No data for {ticker}, skipping.")
            return

        data = dropna(data)
        
        # Ensure we have the required columns
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in data.columns for col in required_columns):
            print(f"Missing required columns for {ticker}, skipping.")
            return
            
        # Add technical indicators
        data = add_technical_indicators(data)

        # Add extra features
        data["Return"] = data["Close"].pct_change()
        data["LogReturn"] = (1 + data["Return"]).apply(
            lambda x: pd.NA if pd.isna(x) or x <= 0 else np.log(x)
        )
        data["MA7"] = data["Close"].rolling(window=7).mean()
        data["VolumeChange"] = data["Volume"].pct_change()

        filename = os.path.join(data_dir, f"{ticker.replace('.', '_')}_minute.csv")
        data.to_csv(filename)
        print(f"Saved {filename}")
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")

# Run for all tickers
for ticker in indian_tickers:
    fetch_minute_data(ticker)
