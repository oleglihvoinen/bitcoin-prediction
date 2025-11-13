# test_data.py - Test data loading and basic functionality
import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(__file__))

try:
    from config.config import DATA_CONFIG
    from utils.data_loader import BitcoinDataLoader
    
    print("🧪 TESTING DATA LOADING")
    print("=" * 40)
    
    # Test data loader
    data_loader = BitcoinDataLoader(DATA_CONFIG)
    df = data_loader.load_data()
    
    print(f"✅ SUCCESS! Data loaded:")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"   Current price: ${df['close'].iloc[-1]:,.2f}")
    print(f"   Data types:\n{df.dtypes}")
    
    # Show first few rows
    print(f"\n📊 First 5 rows:")
    print(df.head())
    
    # Show last few rows  
    print(f"\n📊 Last 5 rows:")
    print(df.tail())
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()