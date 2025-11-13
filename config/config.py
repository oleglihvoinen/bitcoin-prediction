# config/config.py
import os
from datetime import datetime, timedelta

# Get the base directory (where main.py is located)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data configuration
DATA_CONFIG = {
    'start_date': '2020-01-01',
    'end_date': datetime.now().strftime('%Y-%m-%d'),
    'currency': 'USD',
    'data_source': 'coinbase',
    'data_dir': os.path.join(BASE_DIR, 'data'),
    'data_file': 'bitcoin_data.csv'
}

# Feature engineering configuration
FEATURE_CONFIG = {
    'window_sizes': [5, 10, 20],
    'technical_indicators': ['sma', 'ema', 'rsi', 'macd', 'bollinger_bands'],
    'target_column': 'close',
    'prediction_horizon': 1
}

# Model configuration
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'validation_size': 0.1,
    'cv_folds': 3  # Reduced for faster training
}

# LSTM configuration
LSTM_CONFIG = {
    'sequence_length': 30,
    'batch_size': 16,  # Reduced for memory
    'epochs': 30,      # Reduced for faster training
    'patience': 5,
    'units': [32, 32, 1],  # Smaller network
    'dropout_rate': 0.2
}

# Random Forest configuration
RF_CONFIG = {
    'n_estimators': 50,   # Reduced for faster training
    'max_depth': 8,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42
}

# Path configuration
PATHS = {
    'data_dir': os.path.join(BASE_DIR, 'data'),
    'models_dir': os.path.join(BASE_DIR, 'models', 'saved_models'),
    'plots_dir': os.path.join(BASE_DIR, 'plots'),
    'results_dir': os.path.join(BASE_DIR, 'results')
}

# Create directories if they don't exist
for path in PATHS.values():
    os.makedirs(path, exist_ok=True)

print(f"✅ Configuration loaded - Base directory: {BASE_DIR}")