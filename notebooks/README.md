# 📓 Jupyter Notebooks

This directory contains Jupyter notebooks for the Bitcoin Price Prediction project.

## Notebooks Overview

### 1. `01_data_exploration.ipynb`
**Data Analysis and Visualization**
- Load and explore Bitcoin price data
- Analyze price trends and volatility
- Calculate correlations and seasonality
- Data quality assessment

### 2. `02_feature_engineering.ipynb`  
**Feature Creation and Technical Indicators**
- Create 20+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Feature correlation analysis
- Feature quality checks
- Save processed data for modeling

### 3. `03_model_training.ipynb`
**Model Training and Evaluation**
- Train Random Forest and LSTM models
- Model performance comparison
- Feature importance analysis
- Save trained models

## Usage

1. Start Jupyter: `jupyter notebook`
2. Run notebooks in order (01 → 02 → 03)
3. Each notebook is self-contained with explanations
4. Modify parameters as needed for your analysis

## Requirements

- Jupyter installed: `pip install jupyter`
- All dependencies from `requirements.txt`
- Sufficient memory for data processing

## Outputs

- Processed data files in `../data/`
- Trained models in `../models/saved_models/`
- Visualizations and analysis results