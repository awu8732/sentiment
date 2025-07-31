# News Sentiment Trading Project

A machine learning project that predicts stock price movements using news sentiment analysis and Temporal Fusion Transformers.

## Project Overview

This project combines natural language processing (NLP) with quantitative finance to:
- Collect financial news from multiple sources (NewsAPI, Alpha Vantage, Reddit)
- Analyze sentiment using pre-trained models (FinBERT, VADER)
- Predict short-term stock returns using Temporal Fusion Transformers
- Simulate portfolio performance based on sentiment-driven signals

## Project Structure

```
news_sentiment_trading/
├── config/           # Configuration files
├── src/              # Source code
│   ├── data/         # Data collection and management
│   ├── sentiment/    # Sentiment analysis
│   ├── models/       # ML models (TFT, baselines)
│   ├── utils/        # Utilities and helpers
│   └── visualization/ # Dashboards and plots
├── scripts/          # Executable scripts
├── tests/            # Unit tests
├── notebooks/        # Jupyter notebooks for analysis
└── data/             # Data storage
```

## Setup Instructions

### 1. Environment Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd news_sentiment_trading

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. API Keys Configuration
Create a `.env` file in the root directory:

```bash
# Required API Keys
NEWSAPI_KEY=your_newsapi_key_here
ALPHA_VANTAGE_KEY=your_alpha_vantage_key_here

# Optional: Reddit API (for additional sentiment data)
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=NewssentimentTrader/1.0

# Database path (optional, defaults to data/financial_data.db)
DATABASE_PATH=data/financial_data.db

# Logging level (optional, defaults to INFO)
LOG_LEVEL=INFO
```

### 3. Get API Keys

#### NewsAPI (Required)
1. Visit https://newsapi.org/
2. Sign up for free account (1000 requests/day)
3. Copy your API key to `.env` file

#### Alpha Vantage (Required)
1. Visit https://www.alphavantage.co/
2. Get free API key (5 calls/minute, 500 calls/day)
3. Copy your API key to `.env` file

#### Reddit API (Optional)
1. Visit https://www.reddit.com/prefs/apps
2. Create a new application (script type)
3. Copy client ID and secret to `.env` file

## Usage

### Data Collection
```bash
# Collect data for specific symbols
python scripts/collect_data.py --symbols AAPL GOOGL MSFT --days 30

# Collect data for entire sector
python scripts/collect_data.py --sector technology --days 30

# Collect all available symbols
python scripts/collect_data.py --all --days 30

# Incremental update (last few hours)
python scripts/collect_data.py --symbols AAPL --incremental
```

### Data Summary
```bash
# View data collection summary and quality metrics
python scripts/data_summary.py
```

### Pipeline Status
```bash
# Run full pipeline (data collection + processing)
python scripts/run_pipeline.py --symbols AAPL GOOGL --days 30
```

## Data Sources

- **Stock Prices**: Yahoo Finance (via yfinance)
- **Financial News**: NewsAPI, Alpha Vantage News & Sentiment
- **Social Sentiment**: Reddit (r/stocks, r/investing, etc.)

## Key Features

- **Multi-source data collection** with automatic rate limiting
- **Robust error handling** and logging
- **SQLite database** for efficient data storage
- **Modular architecture** for easy extension
- **Configuration management** via environment variables
- **Data quality validation** and monitoring

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_collectors.py
```

## Contributing

1. Follow the existing code structure and naming conventions
2. Add tests for new functionality
3. Update documentation for any API changes
4. Use type hints for better code clarity

## License

This project is for educational purposes. Please respect API rate limits and terms of service.
