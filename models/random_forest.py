# models/random_forest.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import sys

# Add the utils directory to the path so we can import FeatureEngineer
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.feature_engineering import FeatureEngineer

class BitcoinRandomForest:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.feature_importance = None
    
    def prepare_features(self, df):
        """Prepare features for Random Forest"""
        feature_engineer = FeatureEngineer(self.config)
        feature_cols = feature_engineer.get_feature_columns(df)
        
        X = df[feature_cols]
        y = df['target']
        
        return X, y, feature_cols
    
    def train(self, df):
        """Train Random Forest model"""
        print("Training Random Forest model...")
        
        X, y, feature_cols = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['test_size'], 
            random_state=self.config['random_state'],
            shuffle=False  # Important for time series data
        )
        
        # Use simpler hyperparameters for faster training
        rf = RandomForestRegressor(
            n_estimators=self.config.get('n_estimators', 50),
            max_depth=self.config.get('max_depth', 10),
            min_samples_split=self.config.get('min_samples_split', 5),
            min_samples_leaf=self.config.get('min_samples_leaf', 2),
            random_state=self.config.get('random_state', 42),
            n_jobs=-1  # Use all available cores
        )
        
        # Fit the model
        rf.fit(X_train, y_train)
        
        self.model = rf
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
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
        
        print(f"✅ Random Forest Training Complete")
        print(f"   MAE: {mae:.4f}%, RMSE: {rmse:.4f}%, R²: {r2:.4f}")
        print(f"   Top 5 features: {list(self.feature_importance['feature'].head(5))}")
        
        return metrics, y_test, y_pred
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def save_model(self, filepath):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        joblib.dump(self.model, filepath)
        print(f"💾 Random Forest model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        self.model = joblib.load(filepath)
        print(f"📂 Random Forest model loaded from {filepath}")