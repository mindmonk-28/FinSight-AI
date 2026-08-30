import os
import joblib
import yfinance as yf
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from ta import add_all_ta_features
from ta.utils import dropna
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)



# Directories
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
TICKERS = [
    "RELIANCE.NS", "INFY.NS", "TCS.NS", "WIPRO.NS", "HDFCBANK.NS",
    "ICICIBANK.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "ASIANPAINT.NS",
    "LT.NS", "ULTRACEMCO.NS", "MARUTI.NS", "AXISBANK.NS", "BAJFINANCE.NS",
    "NESTLEIND.NS", "TECHM.NS", "KOTAKBANK.NS", "SUNPHARMA.NS", "HCLTECH.NS"
]

# Helper to fetch live data and preprocess
def get_latest_features(ticker):
    data = yf.download(ticker, period="2d", interval="1m", auto_adjust=True)
    if data.empty or len(data) < 20:
        raise ValueError(f"Not enough data for {ticker}")

    data = dropna(data)
    data = add_all_ta_features(
        data, open="Open", high="High", low="Low", close="Close", volume="Volume"
    )
    data["Return"] = data["Close"].pct_change()
    data["LogReturn"] = (1 + data["Return"]).apply(
        lambda x: pd.NA if pd.isna(x) or x <= 0 else np.log(x)
    )
    data["MA7"] = data["Close"].rolling(window=7).mean()
    data["VolumeChange"] = data["Volume"].pct_change()

    data = data.dropna()
    return data.iloc[-1:]  # Only latest row

# Homepage route
@app.route("/")
def home():
    return render_template("index.html", tickers=TICKERS)

# Predict API
@app.route("/predict", methods=["POST"])
def predict():
    ticker = request.form.get("ticker")
    if ticker not in TICKERS:
        return jsonify({"error": "Invalid ticker"}), 400

    try:
        features = get_latest_features(ticker)
        model_path = os.path.join(MODEL_DIR, f"{ticker.replace('.', '_')}_model.pkl")
        model = joblib.load(model_path)

        # Drop target columns if accidentally included
        for col in ["Close", "Return", "LogReturn"]:
            if col in features.columns:
                features = features.drop(columns=[col])

        prediction = model.predict(features)
        return jsonify({"ticker": ticker, "predicted_next_close": float(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "True").lower() in ["true", "1", "t"]
    app.run(host="0.0.0.0", port=port, debug=debug)

