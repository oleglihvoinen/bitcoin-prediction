# utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from datetime import datetime

class BitcoinVisualizer:
    def __init__(self, plots_dir='plots/'):
        self.plots_dir = plots_dir
        os.makedirs(plots_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_price_history(self, df, save=True):
        """Plot Bitcoin price history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Bitcoin Price History and Volume', fontsize=16, fontweight='bold')
        
        # Price plot
        axes[0, 0].plot(df.index, df['close'], linewidth=2, color='#FF9500')
        axes[0, 0].set_title('Bitcoin Closing Price', fontweight='bold')
        axes[0, 0].set_ylabel('Price (USD)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Volume plot
        axes[0, 1].bar(df.index, df['volume'], alpha=0.7, color='#007AFF')
        axes[0, 1].set_title('Trading Volume', fontweight='bold')
        axes[0, 1].set_ylabel('Volume')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Daily returns
        daily_returns = df['close'].pct_change() * 100
        axes[1, 0].hist(daily_returns.dropna(), bins=50, alpha=0.7, color='#34C759')
        axes[1, 0].set_title('Daily Returns Distribution', fontweight='bold')
        axes[1, 0].set_xlabel('Daily Return (%)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Price range
        price_range = (df['high'] - df['low']) / df['close'] * 100
        axes[1, 1].plot(df.index, price_range, alpha=0.7, color='#AF52DE')
        axes[1, 1].set_title('Daily Price Range (% of Close)', fontweight='bold')
        axes[1, 1].set_ylabel('Price Range (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = os.path.join(self.plots_dir, 'price_history.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved price history plot to {filename}")
        
        plt.show()
        return fig
    
    def plot_technical_indicators(self, df, save=True):
        """Plot technical indicators"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Technical Indicators', fontsize=16, fontweight='bold')
        
        # Moving averages
        if 'sma_20' in df.columns and 'ema_20' in df.columns:
            axes[0, 0].plot(df.index, df['close'], label='Close Price', linewidth=2, color='black', alpha=0.7)
            axes[0, 0].plot(df.index, df['sma_20'], label='SMA 20', linewidth=1.5, color='blue')
            axes[0, 0].plot(df.index, df['ema_20'], label='EMA 20', linewidth=1.5, color='red')
            axes[0, 0].set_title('Moving Averages', fontweight='bold')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # RSI
        if 'rsi_14' in df.columns:
            axes[0, 1].plot(df.index, df['rsi_14'], linewidth=2, color='purple')
            axes[0, 1].axhline(70, linestyle='--', alpha=0.7, color='red', label='Overbought')
            axes[0, 1].axhline(30, linestyle='--', alpha=0.7, color='green', label='Oversold')
            axes[0, 1].set_title('RSI (14 periods)', fontweight='bold')
            axes[0, 1].set_ylim(0, 100)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # MACD
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            axes[1, 0].plot(df.index, df['macd'], label='MACD', linewidth=1.5, color='blue')
            axes[1, 0].plot(df.index, df['macd_signal'], label='Signal', linewidth=1.5, color='red')
            axes[1, 0].set_title('MACD', fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # MACD histogram
            axes2 = axes[1, 0].twinx()
            colors = ['green' if x >= 0 else 'red' for x in df['macd_histogram']]
            axes2.bar(df.index, df['macd_histogram'], alpha=0.3, color=colors)
            axes2.set_ylabel('Histogram')
        
        # Bollinger Bands
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            axes[1, 1].plot(df.index, df['close'], label='Close Price', color='black', alpha=0.7)
            axes[1, 1].plot(df.index, df['bb_upper'], label='Upper Band', linestyle='--', color='red')
            axes[1, 1].plot(df.index, df['bb_lower'], label='Lower Band', linestyle='--', color='green')
            axes[1, 1].plot(df.index, df['bb_middle'], label='Middle Band', linestyle='--', color='blue')
            axes[1, 1].fill_between(df.index, df['bb_upper'], df['bb_lower'], alpha=0.2)
            axes[1, 1].set_title('Bollinger Bands', fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = os.path.join(self.plots_dir, 'technical_indicators.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved technical indicators plot to {filename}")
        
        plt.show()
        return fig
    
    def plot_correlation_heatmap(self, df, save=True):
        """Plot correlation heatmap of features"""
        # Select only numeric columns and drop highly correlated ones
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Plot only correlations with target if it exists
        if 'target' in corr_matrix.columns:
            target_corr = corr_matrix['target'].sort_values(ascending=False)
            
            # Plot top correlated features
            top_features = target_corr.head(15)
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(top_features.to_frame(), 
                       annot=True, cmap='coolwarm', center=0,
                       fmt='.2f', linewidths=0.5)
            plt.title('Top Features Correlated with Target', fontweight='bold', fontsize=14)
            plt.tight_layout()
            
            if save:
                filename = os.path.join(self.plots_dir, 'correlation_heatmap.png')
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Saved correlation heatmap to {filename}")
            
            plt.show()
        
        return corr_matrix
    
    def plot_prediction_results(self, y_true, y_pred, model_name, save=True):
        """Plot prediction results"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle(f'{model_name} - Prediction Results', fontsize=16, fontweight='bold')
        
        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.6)
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual Values')
        axes[0].set_ylabel('Predicted Values')
        axes[0].set_title('Actual vs Predicted')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.6)
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('Predicted Values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residuals Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = os.path.join(self.plots_dir, f'{model_name.lower().replace(" ", "_")}_predictions.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved {model_name} predictions plot to {filename}")
        
        plt.show()
        return fig