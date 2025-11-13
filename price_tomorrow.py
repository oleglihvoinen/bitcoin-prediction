# predict_tomorrow.py - Predict tomorrow's Bitcoin price
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from config.config import *
from utils.data_loader import BitcoinDataLoader
from utils.feature_engineering import FeatureEngineer
from models.random_forest import BitcoinRandomForest
from models.lstm_model import BitcoinLSTM

class BitcoinPricePredictor:
    def __init__(self):
        self.data_loader = BitcoinDataLoader(DATA_CONFIG)
        self.feature_engineer = FeatureEngineer(FEATURE_CONFIG)
        self.rf_model = None
        self.lstm_model = None
        self.df = None
        self.feature_cols = None
        
    def load_models(self):
        """Load trained models"""
        try:
            # Load Random Forest
            self.rf_model = BitcoinRandomForest({**MODEL_CONFIG, **RF_CONFIG})
            rf_model_path = os.path.join(PATHS['models_dir'], 'random_forest_model.joblib')
            if os.path.exists(rf_model_path):
                self.rf_model.load_model(rf_model_path)
                print("✅ Random Forest model loaded")
            else:
                print("❌ Random Forest model not found")
                
            # Load LSTM
            self.lstm_model = BitcoinLSTM({**MODEL_CONFIG, **LSTM_CONFIG})
            lstm_model_path = os.path.join(PATHS['models_dir'], 'lstm_model.h5')
            lstm_scaler_path = os.path.join(PATHS['models_dir'], 'lstm_scaler.joblib')
            if os.path.exists(lstm_model_path) and os.path.exists(lstm_scaler_path):
                self.lstm_model.load_model(lstm_model_path, lstm_scaler_path)
                print("✅ LSTM model loaded")
            else:
                print("❌ LSTM model not found")
                
        except Exception as e:
            print(f"Error loading models: {e}")
            
    def prepare_prediction_data(self):
        """Prepare the latest data for prediction"""
        # Load and prepare data
        self.df = self.data_loader.load_data()
        df_features = self.feature_engineer.add_technical_indicators(self.df)
        
        # Get feature columns (excluding target and basic price data)
        exclude_cols = ['target', 'target_price', 'open', 'high', 'low', 'close', 'volume']
        self.feature_cols = [col for col in df_features.columns if col not in exclude_cols]
        
        # Get the latest sequence for prediction
        latest_data = df_features[self.feature_cols].iloc[-1:].values
        
        # For LSTM, we need a sequence
        sequence_length = LSTM_CONFIG['sequence_length']
        if len(df_features) >= sequence_length:
            lstm_sequence = df_features[self.feature_cols].iloc[-sequence_length:].values
            lstm_sequence = lstm_sequence.reshape(1, sequence_length, len(self.feature_cols))
        else:
            lstm_sequence = None
            
        return latest_data, lstm_sequence, df_features
    
    def predict_tomorrow_price(self):
        """Predict tomorrow's Bitcoin price"""
        print("🔮 Predicting Tomorrow's Bitcoin Price...")
        print("=" * 50)
        
        # Load models
        self.load_models()
        
        # Prepare data
        rf_data, lstm_data, df_features = self.prepare_prediction_data()
        
        # Get current price
        current_price = self.df['close'].iloc[-1]
        current_date = self.df.index[-1].strftime('%Y-%m-%d')
        
        print(f"📊 Current Price ({current_date}): ${current_price:,.2f}")
        print("=" * 50)
        
        predictions = {}
        
        # Random Forest Prediction
        if self.rf_model and self.rf_model.model is not None:
            try:
                # The model predicts percentage change, convert to price
                rf_prediction_change = self.rf_model.model.predict(rf_data)[0]
                rf_prediction_price = current_price * (1 + rf_prediction_change/100)
                predictions['Random Forest'] = {
                    'price': rf_prediction_price,
                    'change_percent': rf_prediction_change,
                    'change_dollar': rf_prediction_price - current_price
                }
            except Exception as e:
                print(f"❌ Random Forest prediction failed: {e}")
        
        # LSTM Prediction
        if self.lstm_model and self.lstm_model.model is not None and lstm_data is not None:
            try:
                # LSTM also predicts percentage change
                lstm_prediction_change = self.lstm_model.predict(lstm_data)[0]
                lstm_prediction_price = current_price * (1 + lstm_prediction_change/100)
                predictions['LSTM'] = {
                    'price': lstm_prediction_price,
                    'change_percent': lstm_prediction_change,
                    'change_dollar': lstm_prediction_price - current_price
                }
            except Exception as e:
                print(f"❌ LSTM prediction failed: {e}")
        
        # Simple moving average prediction (baseline)
        if 'sma_7' in df_features.columns:
            sma_7 = df_features['sma_7'].iloc[-1]
            sma_21 = df_features['sma_21'].iloc[-1] if 'sma_21' in df_features.columns else current_price
            
            # Simple trend-based prediction
            if sma_7 > sma_21:  # Uptrend
                sma_prediction = current_price * 1.005  # +0.5%
            else:  # Downtrend
                sma_prediction = current_price * 0.995  # -0.5%
                
            predictions['SMA Trend'] = {
                'price': sma_prediction,
                'change_percent': (sma_prediction/current_price - 1) * 100,
                'change_dollar': sma_prediction - current_price
            }
        
        # Display predictions
        print("\n🎯 TOMORROW'S PRICE PREDICTIONS:")
        print("=" * 50)
        
        for model_name, pred in predictions.items():
            change_icon = "📈" if pred['change_dollar'] > 0 else "📉"
            print(f"{model_name:15} {change_icon} ${pred['price']:>10,.2f} "
                  f"({pred['change_percent']:+.2f}%)")
        
        # Calculate average prediction
        if predictions:
            avg_price = np.mean([pred['price'] for pred in predictions.values()])
            avg_change_percent = (avg_price/current_price - 1) * 100
            avg_change_dollar = avg_price - current_price
            change_icon = "📈" if avg_change_dollar > 0 else "📉"
            
            print("=" * 50)
            print(f"{'AVERAGE':15} {change_icon} ${avg_price:>10,.2f} "
                  f"({avg_change_percent:+.2f}%)")
            
            # Confidence estimate (based on model agreement)
            price_std = np.std([pred['price'] for pred in predictions.values()])
            confidence = max(0, 100 - (price_std / current_price * 1000))
            
            print(f"{'CONFIDENCE':15}   {confidence:.1f}%")
        
        print("=" * 50)
        
        # Additional insights
        print("\n💡 MARKET INSIGHTS:")
        
        # RSI analysis
        if 'rsi_14' in df_features.columns:
            rsi = df_features['rsi_14'].iloc[-1]
            if rsi > 70:
                print("⚠️  RSI indicates OVERBOUGHT conditions")
            elif rsi < 30:
                print("⚠️  RSI indicates OVERSOLD conditions")
            else:
                print("✅ RSI in normal range")
        
        # Volume analysis
        current_volume = self.df['volume'].iloc[-1]
        avg_volume = self.df['volume'].tail(20).mean()
        volume_ratio = current_volume / avg_volume
        if volume_ratio > 1.5:
            print("📊 High trading volume detected")
        elif volume_ratio < 0.7:
            print("📊 Low trading volume detected")
        
        return predictions, current_price

def main():
    predictor = BitcoinPricePredictor()
    predictions, current_price = predictor.predict_tomorrow_price()
    
    # Save prediction to file
    if predictions:
        prediction_date = datetime.now().strftime('%Y-%m-%d')
        prediction_data = {
            'date': prediction_date,
            'current_price': current_price,
            'predictions': predictions,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save as JSON
        import json
        prediction_file = os.path.join(PATHS['results_dir'], 'latest_prediction.json')
        with open(prediction_file, 'w') as f:
            json.dump(prediction_data, f, indent=2)
        
        print(f"\n💾 Prediction saved to: {prediction_file}")

if __name__ == "__main__":
    main()