# utils/data_loader.py
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import os

class BitcoinDataLoader:
    def __init__(self, config):
        self.config = config
        self.data_dir = config.get('data_dir', 'data/')
        self.data_file = os.path.join(self.data_dir, config.get('data_file', 'bitcoin_data.csv'))
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
    
    def fetch_historical_data(self):
        """Fetch historical Bitcoin data from CoinGecko API"""
        print("Fetching historical Bitcoin data from CoinGecko...")
        
        try:
            # CoinGecko API endpoint for historical data
            url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
            
            # Calculate days between start and end date
            start_date = datetime.strptime(self.config['start_date'], '%Y-%m-%d')
            end_date = datetime.strptime(self.config['end_date'], '%Y-%m-%d')
            days = max(1, (end_date - start_date).days)
            
            params = {
                'vs_currency': self.config['currency'],
                'days': days,
                'interval': 'daily'
            }
            
            print(f"Fetching {days} days of data...")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Extract prices
            prices = data['prices']
            
            # Create DataFrame
            df = pd.DataFrame(prices, columns=['timestamp', 'close'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('date')
            df = df.drop('timestamp', axis=1)
            
            # Generate realistic OHLCV data based on close price
            np.random.seed(42)
            
            # Create OHLC data with realistic relationships
            df['open'] = df['close'].shift(1)
            # High is close price plus some random positive movement
            df['high'] = df['close'] * (1 + np.random.uniform(0.01, 0.03, len(df)))
            # Low is close price minus some random negative movement  
            df['low'] = df['close'] * (1 - np.random.uniform(0.01, 0.03, len(df)))
            # Volume with some randomness
            df['volume'] = np.random.uniform(1e9, 5e10, len(df))
            
            # Ensure high is highest and low is lowest
            df['high'] = df[['open', 'close', 'high']].max(axis=1)
            df['low'] = df[['open', 'close', 'low']].min(axis=1)
            
            # Remove first row (no open price)
            df = df.iloc[1:]
            
            print(f"✅ Successfully fetched {len(df)} days of Bitcoin data")
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data from API: {e}")
            print("Using sample data instead...")
            return self.generate_sample_data()
    
    def generate_sample_data(self):
        """Generate realistic sample Bitcoin data"""
        print("Generating sample Bitcoin data...")
        
        # Create date range
        start_date = datetime.strptime(self.config['start_date'], '%Y-%m-%d')
        end_date = datetime.strptime(self.config['end_date'], '%Y-%m-%d')
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate realistic Bitcoin price data with volatility
        np.random.seed(42)
        
        # Start with a reasonable Bitcoin price
        start_price = 30000  # Start from $30K for more realistic simulation
        
        # Create price series with Bitcoin-like volatility
        returns = np.random.normal(0.002, 0.04, len(dates))  # Higher volatility for crypto
        
        # Add some market cycles
        cycle_period = 90  # ~3 month cycles
        cycle = np.sin(2 * np.pi * np.arange(len(dates)) / cycle_period) * 0.01
        
        # Add some upward trend (crypto generally trends up long-term)
        trend = np.linspace(0, 0.001, len(dates))
        
        # Combine components
        returns = returns + cycle + trend
        
        # Generate prices
        prices = [start_price]
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)
        
        # Create DataFrame with realistic OHLCV data
        df = pd.DataFrame(index=dates)
        df['close'] = prices
        df['open'] = df['close'].shift(1)
        
        # Fill first open with close (slightly adjusted)
        df.loc[df.index[0], 'open'] = df.loc[df.index[0], 'close'] * 0.998
        
        # Generate high and low with realistic ranges (1-5% daily range)
        daily_ranges = np.random.uniform(0.01, 0.05, len(df))
        df['high'] = df['close'] * (1 + daily_ranges / 2)
        df['low'] = df['close'] * (1 - daily_ranges / 2)
        
        # Ensure high is highest and low is lowest
        df['high'] = df[['open', 'close', 'high']].max(axis=1)
        df['low'] = df[['open', 'close', 'low']].min(axis=1)
        
        # Volume that correlates with price movement volatility
        price_changes = df['close'].pct_change().abs()
        base_volume = 2e10  # $20B daily volume base
        df['volume'] = base_volume * (1 + price_changes * 10)
        df['volume'] = df['volume'].fillna(base_volume)
        
        print(f"✅ Generated sample data with {len(df)} days (from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")
        print(f"   Current price: ${df['close'].iloc[-1]:,.2f}")
        
        return df
    
    def clean_existing_data(self):
        """Clean up any corrupted data files"""
        if os.path.exists(self.data_file):
            print("Cleaning up existing data file...")
            try:
                # Try to read the file to see what's wrong
                df_test = pd.read_csv(self.data_file)
                print(f"Current file columns: {list(df_test.columns)}")
                
                # If no date column, fix it
                if 'date' not in df_test.columns:
                    print("Fixing missing 'date' column...")
                    # Create new proper data
                    df_new = self.generate_sample_data()
                    self.save_data(df_new)
                    return df_new
                else:
                    # Try to load with proper date parsing
                    return pd.read_csv(self.data_file, index_col='date', parse_dates=True)
                    
            except Exception as e:
                print(f"Error reading existing file: {e}")
                # Create fresh data
                print("Creating fresh data...")
                df_new = self.generate_sample_data()
                self.save_data(df_new)
                return df_new
        else:
            # No file exists, create fresh data
            df_new = self.generate_sample_data()
            self.save_data(df_new)
            return df_new
    
    def save_data(self, df):
        """Save data to CSV"""
        # Reset index to make 'date' a column for saving
        df_to_save = df.reset_index()
        df_to_save.to_csv(self.data_file, index=False)
        print(f"💾 Data saved to {self.data_file}")
    
    def load_data(self):
        """Load data from CSV or fetch/generate new data"""
        print("Loading Bitcoin data...")
        
        if os.path.exists(self.data_file):
            try:
                print("Found existing data file, loading...")
                df = pd.read_csv(self.data_file, parse_dates=['date'])
                df = df.set_index('date')
                print(f"✅ Loaded {len(df)} records from {self.data_file}")
                print(f"   Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
                print(f"   Current price: ${df['close'].iloc[-1]:,.2f}")
                return df
            except Exception as e:
                print(f"❌ Error loading existing data: {e}")
                print("Cleaning up and generating fresh data...")
                return self.clean_existing_data()
        else:
            print("No existing data found. Generating fresh data...")
            df = self.generate_sample_data()
            self.save_data(df)
            return df