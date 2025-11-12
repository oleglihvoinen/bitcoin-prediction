import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config.config import DATA_CONFIG, FEATURE_CONFIG, MODEL_CONFIG, LSTM_CONFIG, RF_CONFIG, PATHS
from utils.data_loader import BitcoinDataLoader
from utils.feature_engineering import FeatureEngineer
from utils.visualization import BitcoinVisualizer
from models.random_forest import BitcoinRandomForest
from models.lstm_model import BitcoinLSTM
from models.model_evaluation import ModelEvaluator
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=== Bitcoin Price Prediction Project ===")
    
    # Step 1: Load data
    print("\n1. Loading data...")
    data_loader = BitcoinDataLoader(DATA_CONFIG)
    df = data_loader.load_data()
    
    # Step 2: Feature engineering
    print("\n2. Engineering features...")
    feature_engineer = FeatureEngineer(FEATURE_CONFIG)
    df_features = feature_engineer.add_technical_indicators(df)
    
    # Step 3: Visualize data
    print("\n3. Creating visualizations...")
    visualizer = BitcoinVisualizer(PATHS['plots_dir'])
    visualizer.plot_price_history(df_features)
    visualizer.plot_technical_indicators(df_features)
    visualizer.plot_correlation_heatmap(df_features)
    
    # Step 4: Train Random Forest model
    print("\n4. Training Random Forest model...")
    rf_model = BitcoinRandomForest({**MODEL_CONFIG, **RF_CONFIG})
    rf_metrics, rf_y_test, rf_y_pred = rf_model.train(df_features)
    
    # Save RF model
    rf_model.save_model(f"{PATHS['models_dir']}random_forest_model.joblib")
    
    # Step 5: Train LSTM model
    print("\n5. Training LSTM model...")
    lstm_model = BitcoinLSTM({**MODEL_CONFIG, **LSTM_CONFIG})
    lstm_metrics, lstm_history, lstm_y_test, lstm_y_pred = lstm_model.train(df_features)
    
    # Save LSTM model
    lstm_model.save_model(
        f"{PATHS['models_dir']}lstm_model.h5",
        f"{PATHS['models_dir']}lstm_scaler.joblib"
    )
    
    # Step 6: Model evaluation and comparison
    print("\n6. Evaluating models...")
    evaluator = ModelEvaluator(PATHS['plots_dir'])
    
    # Compare models
    models_metrics = {
        'Random Forest': rf_metrics,
        'LSTM': lstm_metrics
    }
    
    evaluator.compare_models(models_metrics)
    evaluator.plot_predictions_comparison(
        rf_y_test, rf_y_pred, lstm_y_test, lstm_y_pred
    )
    
    # Plot feature importance for Random Forest
    evaluator.plot_feature_importance(rf_model.feature_importance)
    
    # Plot training history for LSTM
    evaluator.plot_training_history(lstm_history)
    
    print("\n=== Project Complete ===")
    print(f"Results saved to: {PATHS['plots_dir']}")
    print(f"Models saved to: {PATHS['models_dir']}")

if __name__ == "__main__":
    main()