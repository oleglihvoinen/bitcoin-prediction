# Configuration settings for the Bitcoin prediction project
import os
from datetime import datetime, timedelta

# Data configuration
DATA_CONFIG = {
    'start_date': '2015-01-01',
    'end_date': datetime.now().strftime('%Y-%m-%d'),
    'currency': 'USD',
    'data_source': 'coinbase'  # Alternative: 'binance', 'kraken'
}

# Feature engineering configuration
FEATURE_CONFIG = {
    'window_sizes': [5, 10, 20, 50],
    'technical_indicators': ['sma', 'ema', 'rsi', 'macd', 'bollinger_bands'],
    'target_column': 'close',
    'prediction_horizon': 1  # Predict 1 day ahead
}

# Model configuration
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'validation_size': 0.1,
    'cv_folds': 5
}

# LSTM configuration
LSTM_CONFIG = {
    'sequence_length': 60,
    'batch_size': 32,
    'epochs': 100,
    'patience': 15,
    'units': [50, 50, 1],
    'dropout_rate': 0.2
}

# Random Forest configuration
RF_CONFIG = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42
}

# Path configuration
PATHS = {
    'data_dir': 'data/',
    'models_dir': 'models/saved_models/',
    'plots_dir': 'plots/',
    'results_dir': 'results/'
}

# Create directories if they don't exist
for path in PATHS.values():
    os.makedirs(path, exist_ok=True)