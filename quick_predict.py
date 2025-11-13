# quick_predict.py - Simple tomorrow prediction
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def quick_tomorrow_prediction():
    """Quick prediction without complex models"""
    print("🚀 QUICK BITCOIN TOMORROW PREDICTION")
    print("=" * 40)
    
    # Check if data exists
    data_file = 'data/bitcoin_data.csv'
    if not os.path.exists(data_file):
        print("❌ No data found. Run main.py first to download data.")
        return
    
    # Load data
    df = pd.read_csv(data_file, index_col='date', parse_dates=True)
    current_price = df['close'].iloc[-1]
    current_date = df.index[-1].strftime('%Y-%m-%d')
    
    print(f"📊 Current Price ({current_date}): ${current_price:,.2f}")
    print("=" * 40)
    
    # Simple predictions based on recent trends
    recent_prices = df['close'].tail(10)
    
    # Method 1: Simple moving average
    sma_5 = recent_prices.tail(5).mean()
    sma_10 = recent_prices.mean()
    
    # Method 2: Recent momentum
    last_3_days = recent_prices.tail(3)
    momentum = (last_3_days.iloc[-1] / last_3_days.iloc[0] - 1) * 100
    
    # Method 3: Price channels
    high_5 = recent_prices.tail(5).max()
    low_5 = recent_prices.tail(5).min()
    
    # Generate predictions
    predictions = {}
    
    # Bullish prediction (if above SMA)
    if current_price > sma_5:
        bull_pred = current_price * 1.008  # +0.8%
    else:
        bull_pred = current_price * 1.002  # +0.2%
    
    predictions['Bullish Estimate'] = bull_pred
    
    # Bearish prediction
    if current_price < sma_5:
        bear_pred = current_price * 0.995  # -0.5%
    else:
        bear_pred = current_price * 0.998  # -0.2%
    
    predictions['Bearish Estimate'] = bear_pred
    
    # Neutral prediction (average)
    neutral_pred = (bull_pred + bear_pred) / 2
    predictions['Neutral Estimate'] = neutral_pred
    
    # Momentum-based prediction
    if momentum > 1:
        mom_pred = current_price * 1.01  # +1.0%
    elif momentum < -1:
        mom_pred = current_price * 0.99  # -1.0%
    else:
        mom_pred = current_price * 1.002  # +0.2%
    
    predictions['Momentum Based'] = mom_pred
    
    # Display predictions
    print("\n🎯 TOMORROW'S PRICE ESTIMATES:")
    print("=" * 40)
    
    for method, price in predictions.items():
        change = (price / current_price - 1) * 100
        change_icon = "📈" if change > 0 else "📉"
        print(f"{method:18} {change_icon} ${price:>10,.2f} ({change:+.2f}%)")
    
    # Average prediction
    avg_pred = np.mean(list(predictions.values()))
    avg_change = (avg_pred / current_price - 1) * 100
    avg_icon = "📈" if avg_change > 0 else "📉"
    
    print("=" * 40)
    print(f"{'AVERAGE ESTIMATE':18} {avg_icon} ${avg_pred:>10,.2f} ({avg_change:+.2f}%)")
    
    # Additional context
    print(f"\n💡 CONTEXT:")
    print(f"5-day SMA: ${sma_5:,.2f} ({'Above' if current_price > sma_5 else 'Below'} current)")
    print(f"10-day SMA: ${sma_10:,.2f}")
    print(f"3-day momentum: {momentum:+.2f}%")
    print(f"5-day range: ${low_5:,.2f} - ${high_5:,.2f}")
    
    print("\n⚠️  DISCLAIMER: This is a simple estimate, not financial advice!")
    print("   Cryptocurrency markets are highly volatile.")

if __name__ == "__main__":
    quick_tomorrow_prediction()