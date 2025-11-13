# clean_data.py - Clean up corrupted data and start fresh
import os
import pandas as pd
import shutil
from config.config import DATA_CONFIG, PATHS

def clean_and_reset():
    """Clean up all data and start fresh"""
    print("🧹 CLEANING UP BITCOIN DATA")
    print("=" * 50)
    
    # Remove data file if it exists
    data_file = os.path.join(DATA_CONFIG['data_dir'], DATA_CONFIG['data_file'])
    if os.path.exists(data_file):
        print(f"Removing existing data file: {data_file}")
        os.remove(data_file)
    
    # Remove models directory
    models_dir = PATHS['models_dir']
    if os.path.exists(models_dir):
        print(f"Removing models directory: {models_dir}")
        shutil.rmtree(models_dir)
    
    # Remove plots directory  
    plots_dir = PATHS['plots_dir']
    if os.path.exists(plots_dir):
        print(f"Removing plots directory: {plots_dir}")
        shutil.rmtree(plots_dir)
    
    # Remove results directory
    results_dir = PATHS['results_dir']
    if os.path.exists(results_dir):
        print(f"Removing results directory: {results_dir}")
        shutil.rmtree(results_dir)
    
    # Recreate directories
    for path in [DATA_CONFIG['data_dir'], models_dir, plots_dir, results_dir]:
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")
    
    print("✅ Cleanup complete! Ready for fresh start.")
    
    # Test data loading
    print("\n🧪 Testing data loading...")
    from utils.data_loader import BitcoinDataLoader
    data_loader = BitcoinDataLoader(DATA_CONFIG)
    df = data_loader.load_data()
    
    print(f"✅ Data loaded successfully!")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    print(f"   Current price: ${df['close'].iloc[-1]:,.2f}")

if __name__ == "__main__":
    clean_and_reset()