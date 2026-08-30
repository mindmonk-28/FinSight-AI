# scripts/train_global_model.py

import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import joblib
import math

# Directory where minute-level data is stored
data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "minute")
model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(model_dir, exist_ok=True)

# Features to use for training - updated to match our technical indicators
FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA_20', 'EMA_20', 'MACD', 'MACD_Signal', 'MACD_Histogram',
    'RSI', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'Stoch_K', 'Stoch_D',
    'VWAP', 'Return', 'LogReturn', 'MA7', 'VolumeChange'
]

TARGET = 'Close'

def clean_and_prepare_data(df):
    """Clean and prepare the dataframe for training"""
    # Convert numeric columns
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert technical indicator columns to numeric
    for col in df.columns:
        if col not in numeric_columns and col != 'Target':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
    
    return df

# Loop through all minute-level CSVs
ticker_files = glob.glob(os.path.join(data_dir, "*_minute.csv"))

for file_path in ticker_files:
    try:
        print(f"Processing {os.path.basename(file_path)}...")
        
        # Read CSV with explicit date parsing
        df = pd.read_csv(file_path, index_col=0)
        df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
        
        # Clean and prepare data
        df = clean_and_prepare_data(df)
        
        # Check if required features exist
        available_features = [f for f in FEATURES if f in df.columns]
        if len(available_features) < 5:  # Need at least basic OHLCV + some indicators
            print(f"Skipping {os.path.basename(file_path)} - insufficient features")
            continue
            
        # Drop rows with NaN values
        df = df.dropna(subset=available_features + [TARGET])

        # Sort data by datetime to preserve time series order
        df = df.sort_index()

        # Predict next time step Close
        df['Target'] = df[TARGET].shift(-1)
        df = df.dropna(subset=['Target'])

        # Ensure all features are numeric
        X = df[available_features].astype(float)
        y = df['Target'].astype(float)

        if len(X) < 100:  # Need minimum data points
            print(f"Skipping {os.path.basename(file_path)} - insufficient data points ({len(X)})")
            continue

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = math.sqrt(mean_squared_error(y_test, y_pred))
        print(f"{os.path.basename(file_path)} - RMSE: {rmse:.4f}")

        # Save model
        ticker = os.path.basename(file_path).split("_minute")[0]
        joblib.dump(model, os.path.join(model_dir, f"{ticker}_xgb_minute.pkl"))

    except Exception as e:
        print(f"Error training model for {file_path}: {e}")
