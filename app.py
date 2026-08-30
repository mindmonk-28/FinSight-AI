import os
import joblib
import yfinance as yf
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from ta import add_all_ta_features
from ta.utils import dropna
from xgboost import XGBRegressor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Model directory configuration
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

TICKERS = [
    "RELIANCE.NS", "INFY.NS", "TCS.NS", "WIPRO.NS", "HDFCBANK.NS",
    "ICICIBANK.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "ASIANPAINT.NS",
    "LT.NS", "ULTRACEMCO.NS", "MARUTI.NS", "AXISBANK.NS", "BAJFINANCE.NS",
    "NESTLEIND.NS", "TECHM.NS", "KOTAKBANK.NS", "SUNPHARMA.NS", "HCLTECH.NS"
]

def fetch_and_prepare_data(ticker):
    """Fetch live ticker data from yfinance and calculate technical indicator features."""
    data = None
    attempts = [
        ("3mo", "1d"),
        ("6mo", "1d"),
        ("1mo", "1d"),
        ("7d", "15m")
    ]

    for period, interval in attempts:
        try:
            df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
            if df is not None and not df.empty and len(df) >= 15:
                clean_dict = {}
                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    if col in df.columns:
                        col_data = df[col]
                        clean_dict[col] = col_data.iloc[:, 0] if len(col_data.shape) > 1 else col_data
                data = pd.DataFrame(clean_dict, index=df.index)
                break
        except Exception:
            continue

    if data is None or data.empty or len(data) < 15:
        raise ValueError(f"Insufficient market data available for '{ticker}'.")

    data = dropna(data)
    data = add_all_ta_features(
        data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
    )
    data["Return"] = data["Close"].pct_change()
    data["LogReturn"] = (1 + data["Return"]).apply(
        lambda x: pd.NA if pd.isna(x) or x <= 0 else np.log(x)
    )
    data["MA7"] = data["Close"].rolling(window=7).mean()
    data["VolumeChange"] = data["Volume"].pct_change()
    data = data.dropna()

    if data.empty:
        raise ValueError(f"Feature calculation resulted in empty data for '{ticker}'.")

    return data

def get_or_train_model(ticker, df):
    """Load existing model or train an on-the-fly model if missing."""
    ticker_clean = ticker.replace('.', '_')
    candidate_paths = [
        os.path.join(MODEL_DIR, f"{ticker_clean}_model.pkl"),
        os.path.join(MODEL_DIR, f"{ticker_clean}_xgb_minute.pkl"),
        os.path.join(MODEL_DIR, "global_model.pkl")
    ]
    
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                return joblib.load(path)
            except Exception:
                pass

    # Train on-the-fly XGBoost model if pre-trained model file does not exist
    target = df["Close"].shift(-1)
    train_df = df.iloc[:-1].copy()
    train_df["Target"] = target.iloc[:-1]
    train_df = train_df.dropna()

    drop_cols = ["Close", "Return", "LogReturn", "Target"]
    features = [c for c in train_df.columns if c not in drop_cols]
    
    X = train_df[features].astype(float)
    y = train_df["Target"].astype(float)

    model = XGBRegressor(n_estimators=50, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X, y)

    save_path = os.path.join(MODEL_DIR, f"{ticker_clean}_model.pkl")
    try:
        joblib.dump(model, save_path)
    except Exception:
        pass

    return model

@app.route("/", methods=["GET", "POST"])
def home():
    prediction_result = None
    selected_ticker = TICKERS[0]
    error_msg = None

    if request.method == "POST":
        selected_ticker = request.form.get("ticker", selected_ticker)
        if selected_ticker not in TICKERS:
            error_msg = f"Invalid ticker symbol: {selected_ticker}"
        else:
            try:
                df = fetch_and_prepare_data(selected_ticker)
                model = get_or_train_model(selected_ticker, df)
                
                latest_row = df.iloc[-1:]
                latest_close = float(latest_row["Close"].values[0])

                drop_cols = ["Close", "Return", "LogReturn"]
                feat_cols = [c for c in latest_row.columns if c not in drop_cols]
                
                pred_val = float(model.predict(latest_row[feat_cols].astype(float))[0])
                change = pred_val - latest_close
                pct_change = (change / latest_close) * 100

                prediction_result = {
                    "ticker": selected_ticker,
                    "latest_close": f"₹{latest_close:,.2f}",
                    "predicted_close": f"₹{pred_val:,.2f}",
                    "change": f"{'+' if change >= 0 else ''}₹{change:,.2f}",
                    "pct_change": f"{'+' if pct_change >= 0 else ''}{pct_change:.2f}%",
                    "direction": "UP" if change >= 0 else "DOWN"
                }
            except Exception as e:
                error_msg = str(e)

    return render_template(
        "index.html",
        tickers=TICKERS,
        selected_ticker=selected_ticker,
        prediction=prediction_result,
        error=error_msg
    )

@app.route("/predict", methods=["POST"])
def predict():
    ticker = request.form.get("ticker") or (request.json and request.json.get("ticker"))
    if not ticker or ticker not in TICKERS:
        return jsonify({"error": "Invalid or missing ticker"}), 400

    try:
        df = fetch_and_prepare_data(ticker)
        model = get_or_train_model(ticker, df)
        
        latest_row = df.iloc[-1:]
        latest_close = float(latest_row["Close"].values[0])

        drop_cols = ["Close", "Return", "LogReturn"]
        feat_cols = [c for c in latest_row.columns if c not in drop_cols]
        
        pred_val = float(model.predict(latest_row[feat_cols].astype(float))[0])

        return jsonify({
            "ticker": ticker,
            "latest_close": round(latest_close, 2),
            "predicted_next_close": round(pred_val, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "True").lower() in ["true", "1", "t"]
    app.run(host="0.0.0.0", port=port, debug=debug)
