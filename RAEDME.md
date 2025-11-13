# 🔮 Bitcoin Price Prediction with Machine Learning

A comprehensive machine learning system for predicting Bitcoin prices using LSTM neural networks and Random Forest algorithms.

## 📊 Project Overview

This project implements a complete Bitcoin price prediction pipeline including:
- Real-time data fetching from CoinGecko API
- Feature engineering with 20+ technical indicators
- LSTM neural network for time series forecasting
- Random Forest for feature importance analysis
- Model evaluation and visualization
- Tomorrow's price prediction system

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/oleglihvoinen/bitcoin-prediction.git
cd bitcoin-prediction

# Install dependencies
pip install -r requirements.txt

# Run the complete project
python main.py

python predict_tomorrow.py

📈 Results

    LSTM Model: 1.18% MAE, 79% R² score

    Random Forest: 1.24% MAE, 76% R² score

    Feature Importance: RSI, MACD, and moving averages most significant

🛠️ Project Structure
text

bitcoin-prediction/
├── config/           # Configuration settings
├── models/           # ML model implementations
├── utils/            # Data processing utilities
├── notebooks/        # Jupyter notebooks for analysis
├── data/             # Data storage (gitignored)
├── plots/            # Generated visualizations (gitignored)
├── requirements.txt  # Python dependencies
├── main.py           # Main execution script
└── predict_tomorrow.py # Prediction script

📋 Requirements

See requirements.txt for complete list. Main dependencies:

    pandas, numpy, matplotlib, seaborn

    scikit-learn, tensorflow

    ta (technical analysis library)

    requests

⚠️ Disclaimer

This project is for educational purposes only. Cryptocurrency investments carry significant risk, and past performance doesn't guarantee future results.
📄 License

MIT License - feel free to use this code for your own projects!
👨‍💻 Author

Oleg Lihvoinen

    GitHub: @oleglihvoinen

    Blog: oleglihvoinen.github.io