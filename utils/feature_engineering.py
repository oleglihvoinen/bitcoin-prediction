import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.utils import dropna
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator

class FeatureEngineer:
    def __init__(self, config):
        self.config = config
    
    def add_technical_indicators(self, df):
        """Add technical indicators to the DataFrame"""
        print("Adding technical indicators...")
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        # Make a copy to avoid modifying original data
        df_ta = df.copy()
        
        # Simple Moving Averages
        for window in self.config['window_sizes']:
            df_ta[f'sma_{window}'] = df_ta['close'].rolling(window=window).mean()
            df_ta[f'ema_{window}'] = df_ta['close'].ewm(span=window).mean()
        
        # RSI
        df_ta['rsi_14'] = RSIIndicator(df_ta['close'], window=14).rsi()
        
        # MACD
        macd = MACD(df_ta['close'])
        df_ta['macd'] = macd.macd()
        df_ta['macd_signal'] = macd.macd_signal()
        df_ta['macd_histogram'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = BollingerBands(df_ta['close'], window=20, window_dev=2)
        df_ta['bb_upper'] = bb.bollinger_hband()
        df_ta['bb_lower'] = bb.bollinger_lband()
        df_ta['bb_middle'] = bb.bollinger_mavg()
        df_ta['bb_width'] = (df_ta['bb_upper'] - df_ta['bb_lower']) / df_ta['bb_middle']
        
        # Price-based features
        df_ta['price_change'] = df_ta['close'].pct_change()
        df_ta['price_range'] = (df_ta['high'] - df_ta['low']) / df_ta['close']
        df_ta['close_to_open'] = df_ta['close'] / df_ta['open'] - 1
        
        # Volume features
        df_ta['volume_sma'] = df_ta['volume'].rolling(window=20).mean()
        df_ta['volume_ratio'] = df_ta['volume'] / df_ta['volume_sma']
        
        # Lag features
        for lag in [1, 2, 3, 5, 7]:
            df_ta[f'close_lag_{lag}'] = df_ta['close'].shift(lag)
            df_ta[f'volume_lag_{lag}'] = df_ta['volume'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df_ta[f'rolling_std_{window}'] = df_ta['close'].rolling(window).std()
            df_ta[f'rolling_min_{window}'] = df_ta['close'].rolling(window).min()
            df_ta[f'rolling_max_{window}'] = df_ta['close'].rolling(window).max()
        
        # Target variable (future price change)
        horizon = self.config['prediction_horizon']
        df_ta['target'] = (df_ta['close'].shift(-horizon) / df_ta['close'] - 1) * 100
        
        # Drop NaN values created by indicators
        df_ta = df_ta.dropna()
        
        print(f"Feature engineering complete. Final shape: {df_ta.shape}")
        return df_ta
    
    def get_feature_columns(self, df):
        """Get list of feature columns (excluding target and basic price data)"""
        exclude_cols = ['target', 'open', 'high', 'low', 'close', 'volume']
        return [col for col in df.columns if col not in exclude_cols]
    
    def prepare_lstm_data(self, df, sequence_length=60):
        """Prepare data for LSTM model"""
        feature_cols = self.get_feature_columns(df)
        X = df[feature_cols].values
        y = df['target'].values
        
        X_seq, y_seq = [], []
        
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    # Add this method to the FeatureEngineer class in feature_engineering.py
    def add_price_prediction_features(self, df):
        """Add features specifically for price prediction (not just changes)"""
        print("Adding price prediction features...")
        
        # Make a copy to avoid modifying original data
        df_price = df.copy()
        
        # Keep the existing technical indicators
        df_price = self.add_technical_indicators(df_price)
        
        # Change target to actual future price instead of percentage change
        horizon = self.config['prediction_horizon']
        df_price['target_price'] = df_price['close'].shift(-horizon)
        
        # Remove the percentage change target if it exists
        if 'target' in df_price.columns:
            df_price = df_price.drop('target', axis=1)
        
        # Drop NaN values created by indicators and target
        df_price = df_price.dropna()
        
        print(f"Price prediction features complete. Final shape: {df_price.shape}")
        return df_price