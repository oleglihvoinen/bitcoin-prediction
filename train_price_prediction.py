# train_price_predictor.py - Train models for price prediction
from config.config import *
from utils.data_loader import BitcoinDataLoader
from utils.feature_engineering import FeatureEngineer
from models.random_forest import BitcoinRandomForest
from models.lstm_model import BitcoinLSTM
import os

def train_price_models():
    """Train models specifically for price prediction"""
    print("=== Training Price Prediction Models ===")
    
    # Load data
    data_loader = BitcoinDataLoader(DATA_CONFIG)
    df = data_loader.load_data()
    
    # Create feature engineer with price prediction
    feature_engineer = FeatureEngineer(FEATURE_CONFIG)
    df_features = feature_engineer.add_technical_indicators(df)
    
    # Modify target to be actual price instead of percentage change
    horizon = FEATURE_CONFIG['prediction_horizon']
    df_features['target_price'] = df_features['close'].shift(-horizon)
    
    # Drop rows with NaN target
    df_features = df_features.dropna()
    
    print(f"Training data shape: {df_features.shape}")
    
    # Train Random Forest for price prediction
    print("\n1. Training Random Forest for price prediction...")
    rf_model = BitcoinRandomForest({**MODEL_CONFIG, **RF_CONFIG})
    
    # Modify the target for RF training
    feature_cols = feature_engineer.get_feature_columns(df_features)
    X = df_features[feature_cols]
    y = df_features['target_price']  # Actual price target
    
    # We need to modify the RF model to handle price prediction
    # For now, let's create a simple version
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=MODEL_CONFIG['test_size'], 
        random_state=MODEL_CONFIG['random_state'], shuffle=False
    )
    
    rf_price_model = RandomForestRegressor(
        n_estimators=RF_CONFIG['n_estimators'],
        max_depth=RF_CONFIG['max_depth'],
        random_state=RF_CONFIG['random_state']
    )
    
    rf_price_model.fit(X_train, y_train)
    y_pred = rf_price_model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"Price Prediction - MAE: ${mae:.2f}, RMSE: ${rmse:.2f}")
    
    # Save the price prediction model
    import joblib
    price_model_path = os.path.join(PATHS['models_dir'], 'price_rf_model.joblib')
    joblib.dump(rf_price_model, price_model_path)
    print(f"Price prediction model saved to: {price_model_path}")
    
    return rf_price_model

if __name__ == "__main__":
    train_price_models()