# 🤖 AI Finance Agent - Stock Prediction System

A comprehensive stock prediction system that uses both XGBoost and LSTM models to forecast stock prices.

## 🚀 Features

- **Dual Model Approach**: XGBoost (gradient boosting) + LSTM (deep learning)
- **Real-time Data**: Yahoo Finance and Alpha Vantage APIs
- **Technical Indicators**: Moving averages, returns, volume analysis
- **Web Interface**: Flask-based web application
- **Visualization**: Interactive stock price charts

## 📊 Technical Stack

- **Backend**: Python, Flask
- **ML Models**: XGBoost, TensorFlow/Keras LSTM
- **Data Sources**: Yahoo Finance, Alpha Vantage
- **Frontend**: HTML, CSS, Matplotlib
- **Data Processing**: Pandas, NumPy, Scikit-learn

## 🛠️ Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup Models (First Time Only)
```bash
python setup_models.py
```
This will:
- Build the master dataset from 500+ ticker symbols
- Train XGBoost and LSTM models
- Save models to `models/` directory

**Note**: This process takes 10-30 minutes depending on your system.

### 3. Test the System
```bash
python test_prediction.py
```

### 4. Run the Web Application
```bash
python app.py
```
Then visit: http://localhost:5000

## 📁 Project Structure

```
AI finance agent/
├── app.py                 # Flask web application
├── models.py             # Prediction logic
├── data_utils.py         # Data fetching and processing
├── setup_models.py       # Setup script
├── test_prediction.py    # Test script
├── requirements.txt      # Python dependencies
├── scripts/
│   ├── build_dataset.py      # Dataset creation
│   └── train_global_model.py # Model training
├── models/               # Trained models (created after setup)
├── Data/                 # Data files
│   └── Yahoo-Finance-Ticker-Symbols.csv
├── static/               # Generated charts
└── templates/
    └── index.html        # Web interface
```

## 🔮 How It Works

### Data Pipeline
1. **Data Collection**: Fetches historical data from Yahoo Finance/Alpha Vantage
2. **Feature Engineering**: Creates technical indicators:
   - Price returns and log returns
   - 7-day moving average
   - Volume change percentage
3. **Window Processing**: Uses 7-day sliding window for time series analysis

### Model Architecture
- **XGBoost**: Gradient boosting for regression
  - Features: 63 features (9 features × 7 days)
  - Output: Next day's predicted price
- **LSTM**: Deep learning sequence model
  - Input: 7-day sequence of 9 features
  - Architecture: 2 LSTM layers (64 units) + Dropout + Dense
  - Output: Next day's predicted price

### Prediction Process
1. Fetch latest 7 days of data for the ticker
2. Apply feature engineering
3. Make predictions with both models
4. Return predictions and current price

## 🎯 Usage

### Web Interface
1. Enter a stock ticker symbol (e.g., AAPL, GOOGL, MSFT)
2. Click "Predict"
3. View predictions from both models
4. See the stock price chart

### API Usage
```python
from models import predict_next_day

# Get prediction for a stock
prediction = predict_next_day("AAPL")
print(f"XGBoost: ${prediction['xgb']}")
print(f"LSTM: ${prediction['lstm']}")
print(f"Current: ${prediction['latest']}")
```

## 📈 Model Performance

The system uses two different approaches:
- **XGBoost**: Good for capturing non-linear relationships
- **LSTM**: Excellent for temporal patterns and sequences

Both models are trained on a diverse dataset of 500+ stocks to ensure generalization.

## 🔧 Customization

### Adding New Features
Edit `data_utils.py` in the `prepare_features()` function:
```python
def prepare_features(df):
    # Add your new features here
    df['RSI'] = calculate_rsi(df['Close'])
    df['Bollinger_Bands'] = calculate_bollinger_bands(df['Close'])
    return df
```

### Modifying Model Parameters
Edit `scripts/train_global_model.py`:
- XGBoost: Change `n_estimators`, `learning_rate`
- LSTM: Modify layers, units, dropout rates

### Adding New Data Sources
Edit `data_utils.py` in the `get_historical()` function to add new APIs.

## ⚠️ Important Notes

1. **API Limits**: Alpha Vantage has rate limits (5 calls/minute for free tier)
2. **Model Accuracy**: Stock prediction is inherently difficult - use predictions as guidance, not guarantees
3. **Data Freshness**: Models are trained on historical data - market conditions change
4. **Computational Requirements**: LSTM training requires significant RAM and time

## 🐛 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Run scripts from the root directory
2. **Missing Models**: Run `python setup_models.py`
3. **API Errors**: Check internet connection and API keys
4. **Memory Issues**: Reduce `MAX_TICKERS` in `build_dataset.py`

### Getting Help
- Check the console output for error messages
- Verify all dependencies are installed
- Ensure you have sufficient disk space for datasets

## 📝 License

This project is for educational purposes. Please use responsibly and do not rely solely on these predictions for financial decisions.

## 🤝 Contributing

Feel free to improve the models, add features, or enhance the web interface! 