import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

class BitcoinLSTM:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.sequence_length = config['sequence_length']
    
    def prepare_sequences(self, df):
        """Prepare sequences for LSTM"""
        from utils.feature_engineering import FeatureEngineer
        
        feature_engineer = FeatureEngineer(self.config)
        feature_cols = feature_engineer.get_feature_columns(df)
        
        X = df[feature_cols].values
        y = df['target'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        X_seq, y_seq = [], []
        
        for i in range(self.sequence_length, len(X_scaled)):
            X_seq.append(X_scaled[i-self.sequence_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq), feature_cols
    
    def build_model(self, input_shape):
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(self.config['dropout_rate']),
            LSTM(units=50, return_sequences=True),
            Dropout(self.config['dropout_rate']),
            LSTM(units=50, return_sequences=False),
            Dropout(self.config['dropout_rate']),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, df):
        """Train LSTM model"""
        print("Training LSTM model...")
        
        X, y, feature_cols = self.prepare_sequences(df)
        
        # Split data (time series split)
        split_idx = int(len(X) * (1 - self.config['test_size']))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Build model
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        print(self.model.summary())
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config['patience'],
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.0001
        )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_split=self.config['validation_size'],
            callbacks=[early_stopping, reduce_lr],
            verbose=1,
            shuffle=False
        )
        
        # Make predictions
        y_pred = self.model.predict(X_test).flatten()
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
        
        print(f"LSTM Training Complete")
        print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        return metrics, history, y_test, y_pred
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X).flatten()
    
    def save_model(self, model_path, scaler_path):
        """Save trained model and scaler"""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load_model(self, model_path, scaler_path):
        """Load trained model and scaler"""
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        print(f"Model loaded from {model_path}")
        print(f"Scaler loaded from {scaler_path}")