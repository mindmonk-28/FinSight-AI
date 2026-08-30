# scripts/realtime_predictor.py

import time
import yfinance as yf
import numpy as np
import pandas as pd
import joblib
from ta import add_all_ta_features
from ta.utils import dropna
from xgboost import XGBRegressor
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load XGBoost model
xgb_model = joblib.load("../models/xgb_model.joblib")
# Load LSTM model
lstm_model = load_model("../models/lstm_model.h5")
# Load scaler (used during training)
scaler = joblib.load("../models/scaler.joblib")

# Ticker to monitor
ticker = "RELIANCE.NS"
interval = "1m"

# Keep last N minutes for LSTM
window_size = 30

print(f"Starting real-time monitoring for {ticker}...")

while True:
    try:
        # Fetch latest minute-level data (last 1 day for feature history)
        data = yf.download(ticker, period="1d", interval=interval)
        data = dropna(data)

        # Feature engineering
        data = add_all_ta_features(
            data, open="Open", high="High", low="Low", close="Close", volume="Volume"
        )
        data["Return"] = data["Close"].pct_change()
        data["LogReturn"] = (1 + data["Return"]).apply(
            lambda x: pd.NA if pd.isna(x) or x <= 0 else np.log(x)
        )
        data["MA7"] = data["Close"].rolling(window=7).mean()
        data["VolumeChange"] = data["Volume"].pct_change()

        # Drop rows with missing values
        data.dropna(inplace=True)

        # Extract the most recent row
        latest_features = data.iloc[-1:]

        # Scale input for both models
        input_scaled = scaler.transform(latest_features)

        # --- XGBoost prediction ---
        xgb_pred = xgb_model.predict(input_scaled)[0]

        # --- LSTM prediction ---
        if len(data) >= window_size:
            lstm_input = input_scaled[-window_size:].reshape(1, window_size, -1)
            lstm_pred = lstm_model.predict(lstm_input)[0][0]
        else:
            lstm_pred = None

        # Show predictions
        print(f"\n[{pd.Timestamp.now()}] {ticker} Predictions:")
        print(f"XGBoost: ₹{xgb_pred:.2f}")
        if lstm_pred:
            print(f"LSTM: ₹{lstm_pred:.2f}")
        else:
            print("LSTM: Not enough data yet")

    except Exception as e:
        print(f"Error in real-time prediction loop: {e}")

    time.sleep(60)  # Run every 1 minute
