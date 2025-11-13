# models/model_evaluation.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class ModelEvaluator:
    def __init__(self, plots_dir='plots/'):
        self.plots_dir = plots_dir
        os.makedirs(plots_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
    
    def compare_models(self, models_metrics, save=True):
        """Compare performance of different models"""
        metrics_df = pd.DataFrame(models_metrics).T
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # MAE comparison
        axes[0, 0].bar(metrics_df.index, metrics_df['mae'], color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Mean Absolute Error (MAE)')
        axes[0, 0].set_ylabel('MAE (%)')
        for i, v in enumerate(metrics_df['mae']):
            axes[0, 0].text(i, v, f'{v:.3f}', ha='center', va='bottom')
        
        # RMSE comparison
        axes[0, 1].bar(metrics_df.index, metrics_df['rmse'], color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Root Mean Squared Error (RMSE)')
        axes[0, 1].set_ylabel('RMSE (%)')
        for i, v in enumerate(metrics_df['rmse']):
            axes[0, 1].text(i, v, f'{v:.3f}', ha='center', va='bottom')
        
        # R² comparison
        axes[1, 0].bar(metrics_df.index, metrics_df['r2'], color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('R² Score')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].set_ylim(0, 1)
        for i, v in enumerate(metrics_df['r2']):
            axes[1, 0].text(i, v, f'{v:.3f}', ha='center', va='bottom')
        
        # Combined metrics radar chart (simplified)
        models = metrics_df.index
        metrics_to_plot = ['mae', 'rmse']
        
        # Normalize metrics for radar chart (lower is better)
        normalized_metrics = metrics_df[metrics_to_plot].copy()
        for metric in metrics_to_plot:
            normalized_metrics[metric] = 1 - (normalized_metrics[metric] / normalized_metrics[metric].max())
        
        x = np.arange(len(models))
        width = 0.35
        
        for i, metric in enumerate(metrics_to_plot):
            axes[1, 1].bar(x + i*width, normalized_metrics[metric], width, label=metric)
        
        axes[1, 1].set_title('Normalized Performance (Higher is Better)')
        axes[1, 1].set_xticks(x + width/2)
        axes[1, 1].set_xticklabels(models)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save:
            filename = os.path.join(self.plots_dir, 'model_comparison.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved model comparison to {filename}")
        
        plt.show()
        
        # Print metrics table
        print("\n" + "="*50)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*50)
        print(metrics_df.round(4))
        
        return metrics_df
    
    def plot_predictions_comparison(self, rf_y_test, rf_y_pred, lstm_y_test, lstm_y_pred, save=True):
        """Compare predictions from different models"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Predictions Comparison: Random Forest vs LSTM', fontsize=16, fontweight='bold')
        
        # Random Forest predictions
        axes[0, 0].scatter(rf_y_test, rf_y_pred, alpha=0.6, color='blue')
        axes[0, 0].plot([rf_y_test.min(), rf_y_test.max()], [rf_y_test.min(), rf_y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual')
        axes[0, 0].set_ylabel('Predicted')
        axes[0, 0].set_title('Random Forest: Actual vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # LSTM predictions
        axes[0, 1].scatter(lstm_y_test, lstm_y_pred, alpha=0.6, color='green')
        axes[0, 1].plot([lstm_y_test.min(), lstm_y_test.max()], [lstm_y_test.min(), lstm_y_test.max()], 'r--', lw=2)
        axes[0, 1].set_xlabel('Actual')
        axes[0, 1].set_ylabel('Predicted')
        axes[0, 1].set_title('LSTM: Actual vs Predicted')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Time series comparison (first 100 points)
        n_points = min(100, len(rf_y_test))
        indices = range(n_points)
        
        axes[1, 0].plot(indices, rf_y_test[:n_points], label='Actual', color='black', linewidth=2)
        axes[1, 0].plot(indices, rf_y_pred[:n_points], label='RF Predicted', color='blue', linestyle='--')
        axes[1, 0].set_xlabel('Time Index')
        axes[1, 0].set_ylabel('Price Change (%)')
        axes[1, 0].set_title('Random Forest: Time Series Prediction')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(indices, lstm_y_test[:n_points], label='Actual', color='black', linewidth=2)
        axes[1, 1].plot(indices, lstm_y_pred[:n_points], label='LSTM Predicted', color='green', linestyle='--')
        axes[1, 1].set_xlabel('Time Index')
        axes[1, 1].set_ylabel('Price Change (%)')
        axes[1, 1].set_title('LSTM: Time Series Prediction')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = os.path.join(self.plots_dir, 'predictions_comparison.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved predictions comparison to {filename}")
        
        plt.show()
        return fig
    
    def plot_feature_importance(self, feature_importance_df, top_n=15, save=True):
        """Plot feature importance from Random Forest"""
        if feature_importance_df is None:
            print("No feature importance data available")
            return
        
        # Get top N features
        top_features = feature_importance_df.head(top_n)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(top_features)), top_features['importance'], align='center')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Most Important Features (Random Forest)', fontweight='bold', fontsize=14)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center')
        
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save:
            filename = os.path.join(self.plots_dir, 'feature_importance.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved feature importance plot to {filename}")
        
        plt.show()
        return plt.gcf()
    
    def plot_training_history(self, history, save=True):
        """Plot LSTM training history"""
        if history is None:
            print("No training history data available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('LSTM Training History', fontsize=16, fontweight='bold')
        
        # Loss plot
        axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
        if 'val_loss' in history.history:
            axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_title('Model Loss')
        axes[0].set_ylabel('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MAE plot
        if 'mae' in history.history:
            axes[1].plot(history.history['mae'], label='Training MAE', linewidth=2)
            if 'val_mae' in history.history:
                axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
            axes[1].set_title('Model MAE')
            axes[1].set_ylabel('MAE')
            axes[1].set_xlabel('Epoch')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = os.path.join(self.plots_dir, 'training_history.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved training history to {filename}")
        
        plt.show()
        return fig